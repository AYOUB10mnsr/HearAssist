"""Chargement du dataset adiren7/darija_speech_to_text.

L'audio est gardé sous forme d'octets (Audio(decode=False)) et décodé avec
soundfile au moment du batch : pas de dépendance à torchcodec/FFmpeg, et
l'augmentation par bruit peut se faire à la volée.
"""
import io
from itertools import islice
from math import gcd

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, load_dataset

DATASET_ID = "adiren7/darija_speech_to_text"
SAMPLE_RATE = 16_000


def load_darija(split: str, max_samples: int | None = None, streaming: bool = False, seed: int = 0) -> Dataset:
    """Charge un split ('train' ou 'validation').

    streaming=True ne télécharge que les max_samples premiers exemples
    (pratique pour un test rapide) ; sinon le split est téléchargé puis
    éventuellement sous-échantillonné au hasard.
    """
    if streaming:
        if not max_samples:
            raise ValueError("streaming=True demande max_samples")
        ds = load_dataset(DATASET_ID, split=split, streaming=True).cast_column("audio", Audio(decode=False))
        return Dataset.from_list(list(islice(ds, max_samples)))

    ds = load_dataset(DATASET_ID, split=split).cast_column("audio", Audio(decode=False))
    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    return ds


def decode_audio(audio: dict) -> np.ndarray:
    """Octets audio -> tableau float32 mono à 16 kHz."""
    src = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    data, sr = sf.read(src, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = gcd(sr, SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return data


def clip_duration(example: dict) -> float:
    return float(example["end_time"]) - float(example["start_time"])
