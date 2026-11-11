"""Évalue un modèle de transcription (Whisper ou Qwen3-ASR) sur la darija : WER, CER, temps de traitement.

Exemples :
  # Qwen3-ASR sans entraînement, puis après fine-tuning
  python eval_asr.py --backend qwen --model Qwen/Qwen3-ASR-0.6B-hf
  python eval_asr.py --backend qwen --model outputs/qwen3-asr-darija/final

  # Référence sans entraînement (faster-whisper, modèle "small")
  python eval_asr.py --backend faster-whisper --model small --max-samples 200

  # Modèle public déjà ajusté sur la darija (format Hugging Face)
  python eval_asr.py --backend transformers --model ychafiqui/whisper-small-darija

  # Notre modèle après fine-tuning puis conversion CTranslate2
  python eval_asr.py --backend faster-whisper --model outputs/whisper-small-darija/ct2
"""
import argparse
import csv
import json
import time
from pathlib import Path

import jiwer

from data import SAMPLE_RATE, decode_audio, load_darija
from text_norm import normalize


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class FasterWhisperBackend:
    def __init__(self, model: str, device: str, compute_type: str | None, beam_size: int):
        from faster_whisper import WhisperModel

        compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.beam_size = beam_size

    def transcribe(self, audios):
        out = []
        for audio in audios:
            segments, _ = self.model.transcribe(
                audio, language="ar", task="transcribe", beam_size=self.beam_size,
                condition_on_previous_text=False,
            )
            out.append(" ".join(s.text.strip() for s in segments))
        return out


class TransformersBackend:
    def __init__(self, model: str, device: str, beam_size: int):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.utils import logging

        logging.set_verbosity_error()  # masque les avertissements répétés à chaque batch
        self.torch = torch
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(model, dtype=dtype).to(device).eval()
        try:
            self.processor = WhisperProcessor.from_pretrained(model)
        except (ValueError, OSError) as e:
            # Certains dépôts publics n'ont pas de tokenizer utilisable : on prend celui du Whisper
            # d'origine (même vocabulaire). 128 bandes Mel = famille large-v3 / turbo.
            base = "openai/whisper-large-v3" if self.model.config.num_mel_bins == 128 else "openai/whisper-small"
            print(f"  processor introuvable dans {model} ({type(e).__name__}) -> utilisation de {base}")
            self.processor = WhisperProcessor.from_pretrained(base)
        self.dtype = dtype
        self.beam_size = beam_size

    def transcribe(self, audios):
        feats = self.processor.feature_extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        with self.torch.inference_mode():
            ids = self.model.generate(
                feats.to(self.device, self.dtype), language="ar", task="transcribe",
                num_beams=self.beam_size, max_length=448,
            )
        return self.processor.batch_decode(ids, skip_special_tokens=True)


class QwenASRBackend:
    def __init__(self, model: str, device: str, fp32: bool = False):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        from transformers.utils import logging

        logging.set_verbosity_error()
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model)
        # Pas de bf16 sur T4 : fp16 sur GPU, fp32 sur CPU
        dtype = torch.float16 if device == "cuda" and not fp32 else torch.float32
        self.model = AutoModelForMultimodalLM.from_pretrained(model, dtype=dtype).to(device).eval()

    def transcribe(self, audios):
        inputs = self.processor.apply_transcription_request(audios, language=["Arabic"] * len(audios))
        inputs = inputs.to(self.model.device, self.model.dtype)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return self.processor.decode(out[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="nom faster-whisper (small, large-v3-turbo...), id HF ou dossier local")
    p.add_argument("--backend", choices=["faster-whisper", "transformers", "qwen"], default="faster-whisper")
    p.add_argument("--split", default="validation")
    p.add_argument("--max-samples", type=int, default=200, help="0 = tout le split")
    p.add_argument("--streaming", action="store_true", help="ne télécharge que les premiers exemples (test rapide)")
    p.add_argument("--batch-size", type=int, default=8, help="backends transformers et qwen")
    p.add_argument("--fp32", action="store_true", help="qwen : calcul en float32 si le float16 donne des sorties vides ou absurdes")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--compute-type", default=None, help="faster-whisper : float16, int8_float16, int8...")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    device = pick_device(args.device)
    ds = load_darija(args.split, args.max_samples or None, streaming=args.streaming)
    print(f"{len(ds)} exemples ({args.split}) | modèle {args.model} | {args.backend} sur {device}")

    if args.backend == "faster-whisper":
        backend = FasterWhisperBackend(args.model, device, args.compute_type, args.beam_size)
        batch_size = 1
    elif args.backend == "qwen":
        backend = QwenASRBackend(args.model, device, args.fp32)
        batch_size = args.batch_size
    else:
        backend = TransformersBackend(args.model, device, args.beam_size)
        batch_size = args.batch_size

    rows, audio_seconds, compute_seconds = [], 0.0, 0.0
    for start in range(0, len(ds), batch_size):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        audios = [decode_audio(ex["audio"]) for ex in batch]
        t0 = time.perf_counter()
        hyps = backend.transcribe(audios)
        compute_seconds += time.perf_counter() - t0
        for ex, audio, hyp in zip(batch, audios, hyps):
            audio_seconds += len(audio) / SAMPLE_RATE
            rows.append({"reference": ex["text"], "prediction": hyp,
                         "ref_norm": normalize(ex["text"]), "hyp_norm": normalize(hyp)})
        print(f"  {len(rows)}/{len(ds)}", end="\r")

    scored = [r for r in rows if r["ref_norm"]]
    refs = [r["ref_norm"] for r in scored]
    hyps = [r["hyp_norm"] for r in scored]
    metrics = {
        "model": args.model, "backend": args.backend, "device": device, "split": args.split,
        "n_samples": len(scored),
        "wer": round(jiwer.wer(refs, hyps), 4),
        "cer": round(jiwer.cer(refs, hyps), 4),
        "audio_hours": round(audio_seconds / 3600, 3),
        "rtf": round(compute_seconds / audio_seconds, 4),  # < 1 : plus rapide que le temps réel
        "seconds_per_clip": round(compute_seconds / len(rows), 3),
    }
    print("\n" + json.dumps(metrics, indent=2, ensure_ascii=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(args.model).name or args.model.replace('/', '_')}_{args.split}_{len(scored)}"
    with open(out_dir / f"{tag}_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Résultats enregistrés dans {out_dir}/{tag}_*")


if __name__ == "__main__":
    main()
