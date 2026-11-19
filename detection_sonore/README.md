## Évaluation du pipeline de détection sonore YAMNet

### Objectif

Cette expérimentation évalue une première version du pipeline de détection sonore de HearAssist. Elle utilise YAMNet préentraîné pour analyser progressivement l’audio, détecter huit catégories de sons et regrouper les détections en événements horodatés.

Les poids du modèle ne sont pas modifiés. Le travail porte sur le traitement audio, la calibration des seuils et la génération des alertes.

### Protocole expérimental

L’évaluation utilise 480 fichiers du dataset ESC-50 :

- **192 fichiers de calibration** — folds 1 et 2, utilisés pour déterminer les seuils.
- **288 fichiers de test** — folds 3, 4 et 5, utilisés pour évaluer le pipeline.

Le test comprend :

- **192 fichiers cibles** : 24 exemples pour chacune des huit catégories.
- **96 fichiers de fond** : pluie, applaudissements, pas et frappes au clavier.

Aucun chevauchement de sources audio n’a été relevé entre les partitions de calibration et de test.

Les fichiers sont convertis en mono à 16 kHz, puis transmis au détecteur par blocs de 100 ms. YAMNet analyse des fenêtres de 0,975 seconde avec un pas de 0,48 seconde.

Le score de présence d’une catégorie dans un fichier correspond au score maximal obtenu sur ses fenêtres. Les seuils sont calibrés individuellement en maximisant le F2, qui privilégie le rappel.

### Résultats globaux

| Indicateur | Résultat |
|---|---:|
| Fichiers cibles dont le son attendu est détecté | 185 / 192 |
| Fichiers cibles dont le son attendu est manqué | 7 / 192 |
| Rappel macro | 96,35 % |
| Précision macro | 84,20 % |
| F1 macro | 89,03 % |
| Fichiers de fond déclenchant au moins une alerte | 13 / 96 |
| Proportion de fichiers de fond avec alerte | 13,54 % |

Les métriques macro correspondent à la moyenne des métriques des huit catégories.

Le rappel élevé montre que le pipeline retrouve la majorité des sons ciblés. Cependant, une détection correcte peut s’accompagner d’alertes supplémentaires : 27 fichiers cibles déclenchent au moins une catégorie différente de leur étiquette de référence.

### Résultats par catégorie

| Catégorie | Détections correctes | Sons manqués | Précision | Fausses alertes |
|---|---:|---:|---:|---:|
| Aboiement | 24 / 24 | 0 | 96,0 % | 1 |
| Pleurs de bébé | 24 / 24 | 0 | 100 % | 0 |
| Coups à la porte | 22 / 24 | 2 | 88,0 % | 3 |
| Sirène | 24 / 24 | 0 | 88,9 % | 3 |
| Klaxon | 24 / 24 | 0 | 61,5 % | 15 |
| Réveil | 23 / 24 | 1 | 95,8 % | 1 |
| Bris de verre | 20 / 24 | 4 | 83,3 % | 4 |
| Moteur | 24 / 24 | 0 | 60,0 % | 16 |

Les fausses alertes sont comptées par catégorie et par fichier sur l’ensemble du test, selon les annotations de référence. Elles ne représentent pas un nombre d’événements par heure.

### Analyse des erreurs

Les catégories **klaxon** et **moteur** concentrent les principales difficultés de précision. Leurs seuils de déclenchement sont tous deux fixés à 0,03125.

Ces seuils, sélectionnés avec le F2, favorisent la détection des cibles mais permettent davantage de déclenchements incorrects selon les annotations.

Parmi les cas observés :

- 11 fichiers étiquetés « sirène » déclenchent également « klaxon ».
- 5 fichiers étiquetés « pluie » déclenchent « moteur ».

Ces extraits doivent être écoutés pour distinguer les confusions du modèle des sons secondaires réellement présents dans les enregistrements.

Le **bris de verre** constitue la principale difficulté de rappel, avec 4 fichiers manqués sur 24.

### Performances de calcul

| Mesure | Résultat |
|---|---:|
| Temps moyen de calcul par fenêtre | 6,92 ms |
| 95e percentile du temps de calcul par fenêtre | 15,82 ms |

Ces mesures indiquent une marge de calcul encourageante par rapport au pas de 480 ms entre fenêtres, sur la machine utilisée.

Elles ne représentent pas la latence complète de l’application. Le système doit accumuler environ 0,975 seconde d’audio pour analyser sa première fenêtre complète. La capture microphone, le rééchantillonnage, les transferts et l’affichage ajoutent leurs propres délais.

### Limites

- Le test contient seulement 24 exemples positifs par catégorie.
- Les quatre catégories de fond ne couvrent pas tous les environnements réels.
- Les annotations indiquent la catégorie du fichier, sans délimiter précisément chaque événement sonore.
- Le flux continu est simulé à partir de fichiers ; la capture microphone reste à valider.
- Les données de test ayant déjà été examinées lors du benchmark, de nouveaux enregistrements sont nécessaires pour une validation finale indépendante.
- L’agrégation par maximum diffère de celle du benchmark précédent : les variations de métriques ne doivent pas être attribuées uniquement au modèle.

### Conclusion et prochaines étapes

YAMNet constitue une base exploitable pour un prototype de HearAssist. Le pipeline présente un rappel élevé et un temps de calcul compatible avec la poursuite des essais en flux continu sur la machine évaluée.

Les priorités sont de réduire les alertes incorrectes sur les catégories klaxon et moteur, puis d’améliorer la détection du bris de verre.

Les prochaines étapes sont :

1. Écouter et analyser les erreurs.
2. Comparer les critères F1 et F2 sur les données de calibration.
3. Tester le pipeline avec le microphone et des conversations en darija.
4. Mesurer les fausses alertes par heure et la latence sur de nouveaux enregistrements continus annotés.
5. Intégrer la détection sonore avec la transcription et l’interface React.

Cette version est conservée comme référence V1. Les résultats disponibles ne justifient pas encore, à eux seuls, le recours à un fine-tuning.
