"""Transcrit un fichier audio en darija avec faster-whisper (démo et intégration dans HearAssist).

Le modèle attendu est le dossier ct2/ produit par le fine-tuning
(voir transcription/finetuning/README.md).

Exemple :
  python transcribe.py enregistrement.wav --model whisper-medium-darija-aug/ct2
"""
import argparse
import time

from faster_whisper import WhisperModel


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", help="fichier audio (wav, mp3, m4a...)")
    p.add_argument("--model", required=True, help="dossier CTranslate2 ou nom faster-whisper")
    p.add_argument("--device", default="auto")
    p.add_argument("--beam-size", type=int, default=5)
    args = p.parse_args()

    device = pick_device(args.device)
    model = WhisperModel(args.model, device=device, compute_type="float16" if device == "cuda" else "int8")
    t0 = time.perf_counter()
    segments, info = model.transcribe(
        args.audio, language="ar", beam_size=args.beam_size,
        vad_filter=True,  # Silero VAD : découpe la parole comme dans le pipeline temps réel
        condition_on_previous_text=False,
    )
    for s in segments:
        print(f"[{s.start:6.1f} -> {s.end:6.1f}] {s.text.strip()}")
    print(f"\nDurée audio : {info.duration:.1f} s | temps de calcul : {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
