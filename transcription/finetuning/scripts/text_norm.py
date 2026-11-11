"""Normalisation du texte darija (écriture arabe).

Deux niveaux :
- clean_label()  : nettoyage léger appliqué aux transcriptions d'entraînement.
                   On garde la ponctuation utile à l'affichage (. ? !).
- normalize()    : normalisation stricte appliquée aux références ET aux
                   prédictions avant tout calcul de WER / CER.
"""
import re
import unicodedata

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ALEF = re.compile("[آأإٱ]")  # آ أ إ ٱ -> ا
_PUNCT = re.compile(r"[^\w\s]|_")
_SPACES = re.compile(r"\s+")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)


def clean_label(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = text.translate(_DIGITS)
    return _SPACES.sub(" ", text).strip()


def normalize(text: str, fold_ya: bool = True, fold_ta_marbuta: bool = True) -> str:
    text = clean_label(text).lower()
    text = _ALEF.sub("ا", text)
    if fold_ya:
        text = text.replace("ى", "ي")  # ى -> ي
    if fold_ta_marbuta:
        text = text.replace("ة", "ه")  # ة -> ه
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
