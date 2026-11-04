# Jurisprudence — calibration du juge

Journal des écarts au protocole (`PROTOCOLE.md`) et des décisions prises pendant la phase à annotateur unique. Une entrée par écart, datée, écrite au moment où il est constaté et jamais retouchée ensuite. Ces écarts sont repris dans les limites de l'article.

## Écarts au protocole

| Date | Écart | Traitement |
|---|---|---|
| 2026-11-04 | **Rythme d'annotation (§4).** La première session a porté sur 30 échantillons (environ 26 min d'annotation effective, terminée à 9 h 24). Les 90 échantillons restants ont été annotés en une seule séquence d'environ 1 h 30 (dernière écriture à 10 h 53), au lieu d'une session d'environ 30 échantillons par demi-journée au maximum. | Risque de fatigue et de dérive en fin de séquence. Le temps passé sur chaque échantillon est enregistré dans `labels.jsonl` (champ `seconds` : total 115 min, médiane 29 s, 13 échantillons sous 15 s, 5 au-delà de 3 min). La dérive éventuelle est mesurée par la seconde passe (au plus tôt le 2026-11-18). Écart déclaré dans les limites. |
| 2026-11-04 | **Consultation d'un ouvrage de référence** sur Kubernetes (Perré & Gouigoux, 2019) pendant l'annotation, pour la compréhension du domaine. | Les décisions de faithfulness reposent sur le contexte affiché seul, conformément aux consignes v2.0 (« pas par vos propres connaissances de l'outil ») : l'ouvrage a servi à comprendre les notions, pas à valider une affirmation. Ce n'est pas la documentation du corpus. Écart déclaré dans les limites. |
| 2026-11-04 | **Essai préalable des juges, 4 appels** (1 échantillon × 2 juges × 2 critères), après la levée de l'aveugle, avec ragnarok-ai 1.12.1.post1. Il a montré que Prometheus 2 ne suit pas le format « Score: N » des prompts de `LLMJudge` et que le parseur substituait alors un score par défaut de 3 : les scores Prometheus ne mesuraient rien. | Appels supprimés, non utilisés. Aucune note ni explication de juge n'a été montrée à l'annotateur ; l'échantillon de l'essai n'appartient pas à la seconde passe. A conduit à la correction du juge (entrée suivante). |
| 2026-11-04 | **Changement d'instrument avant tout résultat.** ragnarok-ai 1.12.2 (branche `fix/prometheus-judge`, commit `4a7bae9`) : Prometheus 2 reçoit le gabarit officiel de prometheus-eval 0.1.20 sans réponse de référence et un parseur « [RESULT] N » ; les autres juges gardent « Score: N » ; un échec de lecture devient une erreur explicite et comptée, jamais un score par défaut. | Le run des juges utilise cette version ; le commit exact, la version et les digests des modèles sont enregistrés dans `results/run_info.json`. Les scores Prometheus de toute version antérieure sont écartés. |
| 2026-11-04 | **Vérification du parsing, 20 appels** (5 échantillons × 2 juges × 2 critères) avec ragnarok-ai 1.12.2 : lecture réussie pour 20 appels sur 20, aucun contexte tronqué. | Seul le taux de lecture a été communiqué à l'annotateur, jamais les notes. Appels supprimés, non utilisés ; le run complet repart de zéro. |

## Gel des annotations

La première passe (`labels.jsonl`, 120 échantillons, consignes v2.0) est figée avant la levée de l'aveugle et avant tout run de juge. SHA-256 : `c57e7cecc0ac9bbed897be23105554cadade1df306a8e92ee0978979dc5c8e25`.
