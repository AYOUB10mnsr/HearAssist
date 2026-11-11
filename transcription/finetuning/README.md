# Fine-tuning de Whisper sur la darija

Adaptation de Whisper à la darija marocaine sur le dataset Hugging Face
[`adiren7/darija_speech_to_text`](https://huggingface.co/datasets/adiren7/darija_speech_to_text),
téléchargé directement au moment de l'exécution. **Aucune donnée audio n'est versionnée dans ce dépôt.**

## Contenu

| Chemin | Rôle |
|---|---|
| `scripts/data.py` | Chargement du dataset et décodage de l'audio (16 kHz, mono) |
| `scripts/text_norm.py` | Nettoyage des transcriptions et normalisation avant le calcul du WER/CER |
| `scripts/finetune_whisper.py` | Fine-tuning de Whisper : complet ou LoRA, bruit de fond, SpecAugment, dropout |
| `scripts/finetune_qwen_asr.py` | Fine-tuning LoRA de Qwen3-ASR, pour comparaison |
| `scripts/convert_ct2.py` | Conversion du modèle au format CTranslate2 pour `faster-whisper` |
| `scripts/eval_asr.py` | WER, CER, vitesse, et export des transcriptions en CSV |
| `scripts/requirements.txt` | Dépendances Python |
| `notebooks/whisper_small_darija_finetuning.ipynb` | `whisper-small` avec augmentation |
| `notebooks/whisper_medium_darija_finetuning.ipynb` | `whisper-medium` avec augmentation |
| `notebooks/qwen3asr_darija_finetuning.ipynb` | Référence et fine-tuning de Qwen3-ASR |
| `results/whisper_finetuning_runs.csv` | Historique des entraînements |

Les modèles de référence sans entraînement sont mesurés par
`../benchmark/notebooks/whisper_darija_benchmark.ipynb` (résultats dans `../benchmark/results/whisper_baselines.csv`).
Le script de démonstration `../inference/transcribe.py` et le notebook
`../inference/whisper_darija_test_audios.ipynb` (test sur de vrais enregistrements) utilisent le modèle converti.

## Exécution sur Kaggle

1. Compresser le dossier `scripts/` en `.zip` et l'ajouter comme dataset Kaggle (nouvelle version à chaque modification du code).
2. Ouvrir un notebook de `notebooks/` sur Kaggle : **GPU T4 x2**, **Internet On**, et le dataset de code en Input.
3. Exécuter les cellules jusqu'au test rapide, puis lancer **Save Version → Save & Run All (Commit)**.
4. Récupérer `final/`, `ct2/` et `eval/` dans l'onglet **Output** de la version terminée.

Les notebooks retrouvent le code par recherche de `finetune_whisper.py` dans `/kaggle/input`, quel que soit le nom du dataset.

## Paramètres

| Paramètre | `small` | `medium` | Pourquoi |
|---|---|---|---|
| `BATCH, ACCUM` | `16, 1` | `2, 8` | Batch effectif de 16 dans les deux cas ; `medium` occupe déjà ~11,5 Go du T4 (poids, gradients, AdamW) |
| `--max-steps` | 2000 | 2000 | Le premier `small` plafonnait dès l'étape 1000 |
| `--learning-rate` | 1e-5 | 1e-5 | Valeur par défaut du mode complet |
| `--noise-dir`, `--noise-prob 0.3` | oui | oui | Bruit ESC-50 (CC BY-NC) sur 30 % des clips d'entraînement, jamais sur la validation |
| `--spec-augment`, `--dropout 0.1` | oui | oui | Limitent le surapprentissage |
| `--save-only-model` | non | oui | Checkpoints de ~3 Go au lieu de ~9 Go : `/kaggle/working` est limité à ~20 Go |

Le meilleur checkpoint est choisi sur le WER de 200 clips de validation fixes (graine 42).

## Évaluation

- Normalisation identique pour les références et les prédictions (`text_norm.normalize`) : suppression des diacritiques et de la ponctuation, unification des alef, ى → ي et ة → ه.
- WER intermédiaire : calculé par le Trainer pendant l'entraînement (décodage glouton). Il sert à choisir le checkpoint, pas à comparer avec les références.
- Score final : `eval_asr.py` avec `faster-whisper` (beam 5), sur 200 clips (comparaison avec `whisper_baselines.csv`) puis sur toute la validation (1 065 clips).
- Les transcriptions sont exportées dans `eval/*_predictions.csv`, colonnes `reference`, `prediction`, `ref_norm`, `hyp_norm`.

## Résultats

| Modèle | WER | CER | Remarque |
|---|---:|---:|---|
| `whisper-small` sans entraînement | 84,7 % | 43,4 % | `eval_asr.py`, 200 clips |
| `ychafiqui/whisper-small-darija` | 33,1 % | 14,9 % | Modèle public, `eval_asr.py`, 200 clips |
| `whisper-small` fine-tuné, v1 | 39,2 % | 17,3 % | WER du Trainer, étape 2000 ; modèle perdu |
| `whisper-medium` fine-tuné, essai 1 | 36,9 % | 17,2 % | WER du Trainer à l'étape 750, run interrompu (disque plein) |

Le WER du Trainer et celui de `eval_asr.py` ne sont pas strictement comparables (décodage différent).
Le score final de `whisper-medium` sera ajouté après un run complet.
