# YAMNet — Inférence sonore HearAssist

Module de traitement incrémental d'audio PCM mono float32 à 16 kHz.
Le modèle YAMNet officiel est téléchargé par le notebook. Les paramètres sont dans config/yamnet_settings.json.

## Utilisation

```python
import json
from yamnet_detector import YamnetDetector, EventMerger

settings = json.load(open('config/yamnet_settings.json', encoding='utf-8'))
detector = YamnetDetector('yamnet.tflite', hop_seconds=settings['hop_seconds'])
merger = EventMerger(settings['thresholds'], settings['off_ratio'], settings['release_windows'])
# pcm_chunk provient de votre source audio, déjà rééchantillonnée à 16 kHz.
# for window in detector.feed(pcm_chunk):
#     messages = merger.update(window)
# À la fin du flux, analyser detector.flush(), puis merger.flush(durée_audio).
```

Créer une instance par flux. reset() permet une nouvelle session.
Le notebook fournit un exemple complet, y compris la fermeture et les exports.
Les événements sound_start et sound_end ont un event_id commun au sein d'une session.
L'intégration à l'interface devra ajouter un identifiant de session.

## Protocole et limites

Fenêtres natives du modèle, pas de 0,48 seconde, huit catégories cibles.
Seuils calibrés sur les folds 1–2 d'ESC-50, test sur 3–5.
F2 est utilisé par défaut pour privilégier le rappel.
Les labels sont à l'échelle du fichier. Les frontières d'événements sont des estimations par fenêtres.
Les détections sur les quatre catégories de fond ne représentent pas toutes les fausses alertes du monde réel.
La mesure CPU exclut la capture, le réseau et l'affichage React.
Le flux de démonstration est simulé à partir de fichiers.

## Sources

- YAMNet https://developers.google.com/edge/mediapipe/solutions/audio/audio_classifier
- ESC-50 https://github.com/karolpiczak/ESC-50

Consulter les licences des données et du modèle. ESC-50 est principalement sous CC BY-NC 3.0.
