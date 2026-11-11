"""Fine-tuning LoRA de Qwen3-ASR sur la darija (adiren7/darija_speech_to_text).

Qwen3-ASR est un modèle audio -> texte de type « LLM » : la cible est écrite dans le tour
assistant au format du modèle, « language Arabic<asr_text>... », et la loss est calculée
par le modèle (audio et padding masqués automatiquement).

Le meilleur checkpoint est choisi sur la loss de validation ; le WER/CER se mesure
ensuite avec eval_asr.py --backend qwen.

Exemples :
  # Test rapide de bout en bout
  python finetune_qwen_asr.py --streaming --max-train-samples 16 --eval-samples 4 \
      --max-steps 4 --eval-steps 2 --batch-size 2 --output-dir outputs/smoke_qwen

  # Entraînement réel (T4 16 Go)
  python finetune_qwen_asr.py --base-model Qwen/Qwen3-ASR-0.6B-hf --output-dir outputs/qwen3-asr-0.6b-darija
"""
import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, Trainer, TrainingArguments

from data import clip_duration, decode_audio, load_darija
from finetune_whisper import NoiseAugmenter
from text_norm import clean_label

LANGUAGE = "Arabic"  # pas de « darija » dans Qwen3-ASR : on part de l'arabe


@dataclass
class QwenASRCollator:
    processor: AutoProcessor
    augment: NoiseAugmenter | None = None

    def __call__(self, batch):
        conversations = []
        for ex in batch:
            audio = decode_audio(ex["audio"])
            if self.augment and not ex.get("is_eval"):  # jamais sur la validation
                audio = self.augment(audio)
            conversations.append([
                {"role": "user", "content": [{"type": "audio", "audio": audio}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": f"language {LANGUAGE}<asr_text>{clean_label(ex['text'])}"}
                ]},
            ])
        return self.processor.apply_chat_template(
            conversations, tokenize=True, return_dict=True, processor_kwargs={"output_labels": True}
        )


def filter_dataset(ds, min_seconds: float, max_seconds: float = 30.0):
    before = len(ds)
    ds = ds.filter(lambda ex: len(clean_label(ex["text"])) >= 2 and min_seconds <= clip_duration(ex) <= max_seconds + 0.5)
    print(f"  filtrage : {before} -> {len(ds)} exemples")
    return ds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen3-ASR-0.6B-hf", help="ou Qwen/Qwen3-ASR-1.7B-hf")
    p.add_argument("--output-dir", default="outputs/qwen3-asr-darija")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-targets", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"],
                   help="couches adaptées (encodeur audio et décodeur texte)")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--eval-samples", type=int, default=100)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--streaming", action="store_true", help="test rapide : ne télécharge que les premiers exemples")
    p.add_argument("--noise-dir", default=None, help="dossier de .wav de bruit (ex. ESC-50/audio)")
    p.add_argument("--noise-prob", type=float, default=0.3)
    p.add_argument("--snr-db", type=float, nargs=2, default=(5.0, 20.0))
    p.add_argument("--fp32", action="store_true", help="désactive le float16 mixte (si la loss devient nan)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--push-to-hub", default=None, help="id du dépôt HF pour les checkpoints")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    cuda = torch.cuda.is_available()
    print(f"Base : {args.base_model} | LoRA r={args.lora_r} | lr {args.learning_rate} | GPU : {cuda}")

    processor = AutoProcessor.from_pretrained(args.base_model)
    # Poids en float32 + calcul en float16 mixte : le T4 ne gère pas le bfloat16
    model = AutoModelForMultimodalLM.from_pretrained(args.base_model, dtype=torch.float32)

    from peft import LoraConfig, get_peft_model

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, target_modules=args.lora_targets,
    ))
    model.print_trainable_parameters()

    print("Chargement des données...")
    train = filter_dataset(load_darija("train", args.max_train_samples, streaming=args.streaming, seed=args.seed), args.min_seconds)
    val = filter_dataset(load_darija("validation", args.eval_samples, streaming=args.streaming, seed=args.seed), args.min_seconds)
    val = val.add_column("is_eval", [True] * len(val))

    augment = NoiseAugmenter(args.noise_dir, args.noise_prob, tuple(args.snr_db)) if args.noise_dir else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=cuda and not args.fp32,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=max(1, min(25, args.eval_steps)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.num_workers,
        report_to="none",
        seed=args.seed,
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
        hub_strategy="checkpoint",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=QwenASRCollator(processor, augment),
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    print("Meilleur checkpoint :", trainer.state.best_model_checkpoint, "| eval_loss :", trainer.state.best_metric)

    out = Path(args.output_dir)
    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()  # modèle complet, chargeable sans peft
    merged.save_pretrained(out / "final")
    processor.save_pretrained(out / "final")
    print(f"Modèle final enregistré dans {out / 'final'}")
    print(f"WER/CER : python eval_asr.py --backend qwen --model {out / 'final'}")
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
