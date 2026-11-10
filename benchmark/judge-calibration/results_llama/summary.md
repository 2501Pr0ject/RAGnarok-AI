# Calibration du juge : résultats

Seuil par défaut 0.7 (score Prometheus ≥ 4/5). Seuil optimisé sur le split dev, mesuré sur le split test. Température 0.0, seed 42, num_ctx 8192.

## faithfulness

Échecs de lecture exclus :

| Juge | n test | Accord (IC 95 %) | Kappa | Faux acceptés | Faux rejetés | Seuil optimisé (dev) | Kappa test au seuil optimisé |
|---|---|---|---|---|---|---|---|
| llama | 62 | 61% (49%-72%) | -0.09 (pire que le hasard) | 88% | 20% | 0.25 | -0.14 |

Échecs de lecture comptés comme FAIL (seuil par défaut) :

| Juge | n test | dont échecs de lecture | Accord (IC 95 %) | Kappa | Faux acceptés | Faux rejetés |
|---|---|---|---|---|---|---|
| llama | 62 | 0 | 61% (49%-72%) | -0.09 (pire que le hasard) | 88% | 20% |

Référence : un juge qui répond toujours PASS obtiendrait 73% d'accord sur ce split test.

Par origine (seuil par défaut) :

| Juge | Origine | n | Accord | Kappa |
|---|---|---|---|---|
| llama | A-baseline | 8 | 62% | -0.20 |
| llama | B-bad_chunking | 9 | 33% | -0.42 |
| llama | C-weak_embedding | 13 | 69% | 0.16 |
| llama | D-low_k | 9 | 44% | -0.36 |
| llama | E-noisy_context | 9 | 78% | 0.40 |
| llama | F-broken_retrieval | 7 | 100% | n/a |
| llama | G-hallucination_prone | 7 | 43% | 0.00 |

Analyse exploratoire Q2, taux de PASS par origine (tous splits, seuil par défaut) :

| Juge | Origine | n | PASS humain | PASS juge | Score moyen du juge |
|---|---|---|---|---|---|
| llama | A-baseline | 17 | 59% | 88% | 0.91 |
| llama | B-bad_chunking | 17 | 59% | 88% | 0.88 |
| llama | C-weak_embedding | 18 | 78% | 89% | 0.90 |
| llama | D-low_k | 17 | 76% | 71% | 0.72 |
| llama | E-noisy_context | 17 | 76% | 82% | 0.87 |
| llama | F-broken_retrieval | 17 | 100% | 94% | 0.94 |
| llama | G-hallucination_prone | 16 | 38% | 100% | 1.00 |

## relevance

Échecs de lecture exclus :

| Juge | n test | Accord (IC 95 %) | Kappa | Faux acceptés | Faux rejetés | Seuil optimisé (dev) | Kappa test au seuil optimisé |
|---|---|---|---|---|---|---|---|
| llama | 62 | 65% (52%-75%) | -0.12 (pire que le hasard) | 100% | 31% | 0.5 | -0.12 |

Échecs de lecture comptés comme FAIL (seuil par défaut) :

| Juge | n test | dont échecs de lecture | Accord (IC 95 %) | Kappa | Faux acceptés | Faux rejetés |
|---|---|---|---|---|---|---|
| llama | 62 | 0 | 65% (52%-75%) | -0.12 (pire que le hasard) | 100% | 31% |

Référence : un juge qui répond toujours PASS obtiendrait 94% d'accord sur ce split test.

Par origine (seuil par défaut) :

| Juge | Origine | n | Accord | Kappa |
|---|---|---|---|---|
| llama | A-baseline | 8 | 62% | -0.20 |
| llama | B-bad_chunking | 9 | 56% | -0.20 |
| llama | C-weak_embedding | 13 | 77% | -0.11 |
| llama | D-low_k | 9 | 56% | n/a |
| llama | E-noisy_context | 9 | 78% | n/a |
| llama | F-broken_retrieval | 7 | 14% | n/a |
| llama | G-hallucination_prone | 7 | 100% | n/a |

Analyse exploratoire Q2, taux de PASS par origine (tous splits, seuil par défaut) :

| Juge | Origine | n | PASS humain | PASS juge | Score moyen du juge |
|---|---|---|---|---|---|
| llama | A-baseline | 17 | 88% | 88% | 0.85 |
| llama | B-bad_chunking | 17 | 94% | 71% | 0.66 |
| llama | C-weak_embedding | 18 | 89% | 83% | 0.85 |
| llama | D-low_k | 17 | 100% | 65% | 0.68 |
| llama | E-noisy_context | 17 | 100% | 82% | 0.82 |
| llama | F-broken_retrieval | 17 | 100% | 12% | 0.12 |
| llama | G-hallucination_prone | 17 | 100% | 100% | 1.00 |

## Fonctionnement

| Juge | Critère | Local | Appels | Erreurs | Échecs de parsing | Contexte possiblement tronqué | Latence médiane | Temps total | Stabilité (n) |
|---|---|---|---|---|---|---|---|---|---|
| llama | faithfulness | oui | 120 | 0 | 1% | 0 | 19.2 s | 39 min | n/a (0) |
| llama | relevance | oui | 120 | 0 | 0% | 0 | 11.1 s | 23 min | n/a (0) |
