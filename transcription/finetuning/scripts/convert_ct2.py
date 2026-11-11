"""Convertit un modèle Whisper ajusté (format Hugging Face) au format CTranslate2 pour faster-whisper.

Exemple :
  python convert_ct2.py --model outputs/whisper-small-darija/final --out outputs/whisper-small-darija/ct2
"""
import argparse
import shutil
from pathlib import Path

from ctranslate2.converters import TransformersConverter

FILES_FOR_FASTER_WHISPER = ["tokenizer.json", "preprocessor_config.json"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="dossier du modèle complet (sortie 'final' de finetune_whisper.py) ou id HF")
    p.add_argument("--out", required=True)
    p.add_argument("--quantization", default="float16",
                   help="float16 (GPU), int8_float16 (GPU, plus léger), int8 (CPU)")
    args = p.parse_args()

    model_dir = Path(args.model)
    copy_files = [f for f in FILES_FOR_FASTER_WHISPER if (model_dir / f).exists()] if model_dir.exists() else FILES_FOR_FASTER_WHISPER
    TransformersConverter(args.model, copy_files=copy_files).convert(args.out, quantization=args.quantization, force=True)

    # faster-whisper a besoin de tokenizer.json ; on le régénère si le dossier source ne l'avait pas
    if not (Path(args.out) / "tokenizer.json").exists():
        from transformers import WhisperProcessor

        tmp = Path(args.out) / "_tok"
        WhisperProcessor.from_pretrained(args.model).save_pretrained(tmp)
        for f in FILES_FOR_FASTER_WHISPER:
            if (tmp / f).exists():
                shutil.copy(tmp / f, Path(args.out) / f)
        shutil.rmtree(tmp)
    print(f"Modèle CTranslate2 ({args.quantization}) écrit dans {args.out}")


if __name__ == "__main__":
    main()
