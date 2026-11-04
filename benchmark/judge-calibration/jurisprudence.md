# Jurisprudence — calibration du juge

Journal des écarts au protocole (`PROTOCOLE.md`) et des décisions prises pendant la phase à annotateur unique. Une entrée par écart, datée, écrite au moment où il est constaté et jamais retouchée ensuite. Ces écarts sont repris dans les limites de l'article.

## Écarts au protocole

| Date | Écart | Traitement |
|---|---|---|
| 2026-11-04 | **Rythme d'annotation (§4).** La première session a porté sur 30 échantillons (environ 26 min d'annotation effective, terminée à 9 h 24). Les 90 échantillons restants ont été annotés en une seule séquence d'environ 1 h 30 (dernière écriture à 10 h 53), au lieu d'une session d'environ 30 échantillons par demi-journée au maximum. | Risque de fatigue et de dérive en fin de séquence. Le temps passé sur chaque échantillon est enregistré dans `labels.jsonl` (champ `seconds` : total 115 min, médiane 29 s, 13 échantillons sous 15 s, 5 au-delà de 3 min). La dérive éventuelle est mesurée par la seconde passe (au plus tôt le 2026-11-18). Écart déclaré dans les limites. |
| 2026-11-04 | **Consultation d'un ouvrage de référence** sur Kubernetes (Perré & Gouigoux, 2019) pendant l'annotation, pour la compréhension du domaine. | Les décisions de faithfulness reposent sur le contexte affiché seul, conformément aux consignes v2.0 (« pas par vos propres connaissances de l'outil ») : l'ouvrage a servi à comprendre les notions, pas à valider une affirmation. Ce n'est pas la documentation du corpus. Écart déclaré dans les limites. |

## Gel des annotations

La première passe (`labels.jsonl`, 120 échantillons, consignes v2.0) est figée avant la levée de l'aveugle et avant tout run de juge. SHA-256 : `c57e7cecc0ac9bbed897be23105554cadade1df306a8e92ee0978979dc5c8e25`.
