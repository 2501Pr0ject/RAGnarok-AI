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

## Classement des désaccords juge/humain (figé le 2026-11-05, après F et G du split test)

Arbre de décision, appliqué dans l'ordre pour chaque juge en désaccord :

1. ANNOT : mon étiquette est fausse au regard des consignes v2.0 et de cette jurisprudence.
2. Abstention justifiée (le contexte ne contient pas l'information) :
   - ABST : en faithfulness, le juge la pénalise alors qu'elle n'affirme rien de non appuyé.
   - DEF : en relevance, la grille du juge (« la réponse traite-t-elle la question ? ») ne prévoit pas qu'une abstention justifiée soit un PASS ; écart de définition, pas erreur de raisonnement.
3. Humain FAIL, juge PASS :
   - CONTRA : la réponse affirme quelque chose que le contexte contredit.
   - INV : la réponse affirme quelque chose d'absent du contexte.
4. Humain PASS, juge FAIL, hors abstention :
   - LECTURE : le juge déclare non appuyée une affirmation présente dans le contexte.
   - CRIT : le juge sanctionne un autre critère (exactitude, complétude, style).
5. AUTRE : inclassable, avec commentaire.

Une seule catégorie par ligne. Priorité : ANNOT > CONTRA > INV > ABST > DEF > LECTURE > CRIT > AUTRE.

Précisions :
- Question à prémisse fausse : l'abstention est justifiée si le contexte ne permet pas de corriger la prémisse ; ANNOT si le contexte la réfute.
- Attribuer un passage du contexte à un autre objet que celui qu'il décrit, quand le contexte distingue les deux, relève de CONTRA (ex. p183 : signature de re.Pattern.split présentée comme celle de str.split).
- Le commentaire cite une ou plusieurs phrases complètes de l'explication du juge, mot pour mot.
- (a posteriori, 2026-11-05, découvert au classement) Réponse contenant un passage substantiel dans une autre langue que celle de la question : relevance FAIL. Cas p042 : la note d'annotation mentionne le passage chinois, mais aussi la contradiction ; motif ambigu entre cette règle et la règle « prémisse fausse = relevance PASS ». Classé AUTRE.
- (a posteriori, 2026-11-05) Une abstention reste justifiée si l'information n'est que déductible, sans être explicite dans le contexte (p093, p193 ; p101 classé ANNOT par cohérence).
- (a posteriori, 2026-11-05, découvert au classement) L'aperçu de 1 500 caractères de `label.py` peut masquer la preuve : dans p010, la commande d'activation Windows figure dans le contexte à la position 2 206, et la note d'annotation la dit absente. Étiquette ANNOT. Pour la seconde passe, afficher le contexte complet (`c`) avant tout verdict de faithfulness.
- (a posteriori, 2026-11-05) p198 relevance FAIL (motivé par l'inexactitude des bénéfices) et p050 relevance PASS (réponse sur l'usage de Python, pas sur son installation) : étiquettes contraires aux consignes v2.0 (l'exactitude ne se juge pas en relevance ; répondre à une autre question = FAIL). Reclassés ANNOT.

## Méthode de classement des désaccords

Split test, configurations F et G, puis A et B (p042, p097, p020) : classées par l'annotateur, qui a figé l'arbre de décision à cette occasion. Configurations B à E restantes (59 lignes, lignes 27 à 85 de `results/categories.csv`) : pré-classement automatisé selon l'arbre figé, chaque ligne citant mot pour mot l'explication du juge ; 7 lignes marquées « À confirmer » (p010, p050, p198, p101) tranchées par l'annotateur le 2026-11-05, toutes ANNOT, après vérification du contexte complet et des notes d'annotation (voir ci-dessus) ; p036 (CONTRA) et p093 (ABST) revérifiées sur les fragments.

Vérification par échantillon, deux tirages, tous deux rapportés :

1. **Tirage seed 2026** (2026-11-05) : 10 lignes parmi les 52 pré-classées hors « À confirmer » (lignes 33, 34, 41, 51, 63, 66, 69, 70, 72, 83 ; `results/verification_echantillon_seed2026.csv`). Remplacé à la demande de l'annotateur pour élargir le périmètre aux 59 lignes, ANNOT comprises.
2. **Tirage seed 1105** (2026-11-05) : 10 lignes parmi les lignes 27 à 85 (27, 28, 32, 34, 36, 41, 44, 60, 68, 75 ; `results/verification_echantillon.csv` et `.md`). Deux lignes communes aux deux tirages (p093/mistral/faithfulness, p112/prometheus/relevance).

Les 10 lignes de chaque tirage ont été classées par l'annotateur sur papier, à l'aveugle (sans consulter la catégorie ni le commentaire pré-classés), puis comparées au pré-classement automatisé : accord auteur / pré-classement de 10/10 pour chaque tirage, soit 18 lignes distinctes sur les 59. Les feuilles papier n'ont pas été transcrites : les catégories conservées dans `categories.csv` sont celles du pré-classement, identiques. Seuil de solidité fixé à 9/10 avant la vérification ; le classement est considéré vérifié. Le classement du split dev, exploratoire, suivra la même procédure.
