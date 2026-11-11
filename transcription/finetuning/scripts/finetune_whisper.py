"""Fine-tuning de Whisper sur la darija (adiren7/darija_speech_to_text).

Deux modes :
- complet (par défaut) : conseillé pour whisper-small (T4 Colab ou RTX 8 Go) ;
- LoRA (--lora)        : pour les gros modèles (medium, large-v3-turbo).

Exemples :
  # Test rapide de bout en bout (quelques minutes, CPU possible)
  python finetune_whisper.py --base-model openai/whisper-tiny --streaming \
      --max-train-samples 32 --eval-samples 8 --max-steps 4 --eval-steps 2 --output-dir outputs/smoke

  # Entraînement réel de whisper-small
  python finetune_whisper.py --base-model openai/whisper-small --output-dir outputs/whisper-small-darija

  # LoRA sur large-v3-turbo
  python finetune_whisper.py --base-model openai/whisper-large-v3-turbo --lora \
      --batch-size 4 --grad-accum 4 --output-dir outputs/turbo-darija-lora
"""
import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from data import SAMPLE_RATE, clip_duration, decode_audio, load_darija
from text_norm import clean_label, normalize

MAX_LABEL_TOKENS = 444  # limite du décodeur Whisper (448) moins les jetons de préfixe


class NoiseAugmenter:
    """Mélange un bruit de fond (fichiers .wav d'un dossier, ex. ESC-50) à un SNR aléatoire."""

    def __init__(self, noise_dir: str, prob: float, snr_db: tuple[float, float]):
        self.files = sorted(Path(noise_dir).rglob("*.wav"))
        if not self.files:
            raise FileNotFoundError(f"aucun .wav dans {noise_dir}")
        self.prob, self.snr_db = prob, snr_db

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if random.random() > self.prob:
            return audio
        noise, sr = sf.read(random.choice(self.files), dtype="float32", always_2d=True)
        noise = noise.mean(axis=1)
        if sr != SAMPLE_RATE:
            from scipy.signal import resample_poly

            noise = resample_poly(noise, SAMPLE_RATE, sr).astype(np.float32)
        noise = np.resize(noise, audio.shape)  # répète ou coupe à la longueur du clip
        snr = random.uniform(*self.snr_db)
        p_sig, p_noise = np.mean(audio**2) + 1e-10, np.mean(noise**2) + 1e-10
        return audio + noise * np.sqrt(p_sig / (p_noise * 10 ** (snr / 10)))


@dataclass
class DataCollator:
    """Décode l'audio, calcule les log-Mel et tokenise les transcriptions à la volée."""

    processor: WhisperProcessor
    decoder_start_token_id: int
    augment: NoiseAugmenter | None = None

    def __call__(self, batch):
        audios = [decode_audio(ex["audio"]) for ex in batch]
        if self.augment:  # jamais sur la validation : le WER doit rester comparable
            audios = [a if ex.get("is_eval") else self.augment(a) for ex, a in zip(batch, audios)]
        features = self.processor.feature_extractor(
            audios, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features
        tok = self.processor.tokenizer(
            [clean_label(ex["text"]) for ex in batch], padding=True, return_tensors="pt"
        )
        labels = tok.input_ids.masked_fill(tok.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]  # le modèle ajoute lui-même <|startoftranscript|>
        return {"input_features": features, "labels": labels}


def filter_dataset(ds, tokenizer, min_seconds: float):
    def keep(ex):
        text = clean_label(ex["text"])
        return (
            len(text) >= 2
            and clip_duration(ex) >= min_seconds
            and len(tokenizer(text).input_ids) <= MAX_LABEL_TOKENS
        )

    before = len(ds)
    ds = ds.filter(keep)
    print(f"  filtrage : {before} -> {len(ds)} exemples")
    return ds


def build_compute_metrics(tokenizer):
    def compute_metrics(pred):
        label_ids = np.where(pred.label_ids != -100, pred.label_ids, tokenizer.pad_token_id)
        hyps = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps) if normalize(r)]
        refs, hyps = zip(*pairs)
        return {"wer": jiwer.wer(list(refs), list(hyps)), "cer": jiwer.cer(list(refs), list(hyps))}

    return compute_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="openai/whisper-small")
    p.add_argument("--output-dir", default="outputs/whisper-small-darija")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=None, help="défaut : 1e-5 (complet), 1e-3 (LoRA)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=200, help="taille du sous-ensemble de validation suivi pendant l'entraînement")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--streaming", action="store_true", help="test rapide : ne télécharge que les premiers exemples")
    p.add_argument("--noise-dir", default=None, help="dossier de .wav de bruit (ex. ESC-50/audio) pour l'augmentation")
    p.add_argument("--noise-prob", type=float, default=0.3)
    p.add_argument("--snr-db", type=float, nargs=2, default=(5.0, 20.0))
    p.add_argument("--spec-augment", action="store_true", help="masque des zones du spectrogramme (limite le surapprentissage)")
    p.add_argument("--dropout", type=float, default=0.0, help="ex. 0.1 pour limiter le surapprentissage")
    p.add_argument("--save-only-model", action="store_true", help="checkpoints sans l'état de l'optimiseur (3 fois plus légers avec le mode complet), mais --resume devient impossible")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume", action="store_true", help="reprend au dernier checkpoint (session Colab coupée)")
    p.add_argument("--push-to-hub", default=None, help="id du dépôt HF où pousser les checkpoints (ex. equipe/whisper-darija)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    cuda = torch.cuda.is_available()
    lr = args.learning_rate or (1e-3 if args.lora else 1e-5)
    print(f"Base : {args.base_model} | {'LoRA' if args.lora else 'complet'} | lr {lr} | GPU : {cuda}")

    processor = WhisperProcessor.from_pretrained(args.base_model, language="arabic", task="transcribe")
    # Poids en float32 + calcul en float16 mixte (fp16=True) : une base en fp16 casse generate() à l'évaluation
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.float32, dropout=args.dropout, apply_spec_augment=args.spec_augment,
        mask_time_prob=0.05, mask_feature_prob=0.05,
    )
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    if args.lora:
        from peft import LoraConfig, get_peft_model

        model.model.encoder.conv1.register_forward_hook(lambda _m, _i, out: out.requires_grad_(True))
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        ))
        model.print_trainable_parameters()

    print("Chargement des données...")
    train = load_darija("train", args.max_train_samples, streaming=args.streaming, seed=args.seed)
    val = load_darija("validation", args.eval_samples, streaming=args.streaming, seed=args.seed)
    train = filter_dataset(train, processor.tokenizer, args.min_seconds)
    val = filter_dataset(val, processor.tokenizer, args.min_seconds)
    val = val.add_column("is_eval", [True] * len(val))

    augment = NoiseAugmenter(args.noise_dir, args.noise_prob, tuple(args.snr_db)) if args.noise_dir else None
    collator = DataCollator(processor, model.config.decoder_start_token_id, augment)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=cuda,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=2,
        save_only_model=args.save_only_model,
        logging_steps=max(1, min(25, args.eval_steps)),
        predict_with_generate=True,
        generation_max_length=448,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,  # le collator a besoin des colonnes brutes
        label_names=["labels"],
        dataloader_num_workers=args.num_workers,
        report_to="none",
        seed=args.seed,
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
        hub_strategy="checkpoint",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor.tokenizer),
        processing_class=processor.feature_extractor,
    )
    trainer.train(resume_from_checkpoint=args.resume or None)

    out = Path(args.output_dir)
    print("Meilleur checkpoint :", trainer.state.best_model_checkpoint, "| WER :", trainer.state.best_metric)
    final_dir = out / "final"
    if args.lora:
        model.save_pretrained(out / "adapter")
        model = model.merge_and_unload()  # modèle complet, prêt pour la conversion CTranslate2
    model.generation_config.forced_decoder_ids = None
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Modèle final enregistré dans {final_dir}")
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
