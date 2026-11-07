# HearAssist

**Application d’assistance auditive combinant transcription de la parole et détection des sons environnants.**

HearAssist vise à rendre les informations sonores accessibles grâce à une interface qui affiche les paroles prononcées et les événements sonores détectés. Le projet porte notamment sur l’adaptation de Whisper à la **darija marocaine** et sur l’intégration de ces fonctionnalités dans une application React.

> **Projet en cours de développement.** Les benchmarks, l’adaptation des modèles et leur intégration dans l’application constituent des étapes distinctes.

## Objectifs

- Transcrire la parole en darija marocaine.
- Évaluer l’apport du fine-tuning de Whisper sur la qualité de transcription.
- Reconnaître des sons utiles au quotidien, comme les sirènes, les klaxons et les sonneries.
- Traiter un même flux audio avec les deux modèles en parallèle.
- Afficher les transcriptions et les alertes sonores dans une interface accessible.
- Mesurer la précision, les fausses détections et le délai d’affichage.

## Organisation du projet

Le dépôt comprend trois composants principaux.

| Composant | Responsabilité |
|---|---|
| `transcription/` | Benchmark des modèles de transcription, fine-tuning et inférence |
| `detection_sonore/` | Benchmark des modèles sonores et implémentation de la détection |
| `interface/` | Application React et présentation des résultats à l’utilisateur |

```text
HearAssist/
├── README.md
├── .gitignore
│
├── transcription/
│   ├── README.md
│   ├── benchmark/
│   │   ├── notebooks/
│   │   └── results/
│   ├── finetuning/
│   │   ├── notebooks/
│   │   ├── scripts/
│   │   └── results/
│   └── inference/
│
├── detection_sonore/
│   ├── README.md
│   ├── benchmark/
│   │   ├── notebooks/
│   │   └── results/
│   └── inference/
│
└── interface/
    └── README.md
```

Les dossiers seront complétés progressivement au fil du développement.

## Transcription de la parole

Ce composant regroupe les expériences de reconnaissance vocale et l’adaptation du modèle à la darija marocaine.

Le travail comprend les étapes suivantes.

1. Comparer les modèles de transcription sur un échantillon commun.
2. Sélectionner un checkpoint de départ.
3. Préparer les données d’entraînement, de validation et de test.
4. Effectuer le fine-tuning de Whisper.
5. Comparer le modèle adapté à ses références.
6. Développer son utilisation sur un flux audio continu.

L’évaluation utilise notamment le **WER**, taux d’erreur sur les mots, et le **CER**, taux d’erreur sur les caractères. Les données utilisées et les conventions de normalisation doivent être documentées pour chaque expérience.

## Détection sonore

Ce composant évalue les modèles de reconnaissance des sons environnants et prépare leur utilisation dans HearAssist.

Le premier benchmark compare **YAMNet** et **EfficientAT**, avec le checkpoint `mn10_as`, sur un sous-ensemble d’ESC-50.

Les huit catégories cibles sont les suivantes.

- Aboiement.
- Pleurs de bébé.
- Coups à la porte.
- Sirène.
- Klaxon.
- Sonnerie de réveil.
- Bris de verre.
- Moteur.

Le protocole comprend **480 fichiers**.

| Partition | Sons cibles | Sons de fond | Total |
|---|---:|---:|---:|
| Calibration des seuils | 128 | 64 | 192 |
| Test | 192 | 96 | 288 |

Les résultats comprennent les prédictions, les métriques par catégorie, les matrices de confusion, les seuils et les paramètres d’exécution.



## Interface React

L’interface prévue permettra de présenter les résultats des deux composants.

Les fonctionnalités envisagées comprennent les éléments suivants.

- Activation et arrêt de l’écoute.
- Affichage progressif des transcriptions.
- Signalement visuel des sons détectés.
- Historique horodaté des événements.
- Présentation des états d’écoute et de traitement.

L’intégration devra gérer les transcriptions provisoires, leurs corrections et le regroupement des alertes répétées.

## Fonctionnement prévu

Le microphone fournira un flux audio commun aux deux traitements. Chaque composant analysera les portions audio selon ses besoins.

Les résultats seront associés aux horodatages de capture, puis transmis à l’interface React. Cette organisation permettra d’afficher une transcription pendant qu’un événement sonore est détecté.

Le délai utilisateur sera mesuré sur la chaîne complète, depuis la capture jusqu’à l’affichage.

## Exécution des expériences

Les notebooks sont destinés aux expérimentations sur Kaggle. Chaque expérience doit préciser ses dépendances, ses données, ses checkpoints et ses paramètres.

Pour reproduire un benchmark, ouvrir son notebook, configurer l’environnement indiqué et exécuter les cellules dans l’ordre. Les résultats exportés sont conservés dans le dossier `results/` correspondant.

Le développement de l’inférence en continu et de l’interface sera réalisé dans les composants dédiés du dépôt.


## Prochaines étapes

- Évaluer la première version de Whisper fine-tunée.
- Ajuster les seuils de détection sonore.
- Comparer les modèles sur des fenêtres audio courtes.
- Ajouter des enregistrements représentatifs des conditions d’utilisation.
- Implémenter le traitement du microphone en continu.
- Connecter les modèles à l’application React.
- Mesurer la qualité et les délais avec les deux traitements actifs.
