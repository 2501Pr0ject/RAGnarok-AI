# Calibration du juge : résultats

Seuil par défaut 0.7 (score Prometheus ≥ 4/5). Seuil optimisé sur le split dev, mesuré sur le split test. Température 0.0, seed 42, num_ctx 8192.

## faithfulness

Échecs de lecture exclus :

| Juge | n test | Accord (IC 95 %) | Kappa | Faux acceptés | Faux rejetés | Seuil optimisé (dev) | Kappa test au seuil optimisé |
|---|---|---|---|---|---|---|---|
| mistral | 62 | 55% (43%-67%) | -0.18 (pire que le hasard) | 88% | 29% | 0.25 | -0.17 |
| prometheus | 62 | 52% (39%-64%) | -0.17 (pire que le hasard) | 82% | 36% | 1.0 | 0.02 |

Échecs de lecture comptés comme FAIL (seuil par défaut) :

| Juge | n test | dont échecs de lecture | Accord (IC 95 %) | Kappa | Faux acceptés | Faux rejetés |
|---|---|---|---|---|---|---|
| mistral | 62 | 0 | 55% (43%-67%) | -0.18 (pire que le hasard) | 88% | 29% |
| prometheus | 62 | 0 | 52% (39%-64%) | -0.17 (pire que le hasard) | 82% | 36% |

Référence : un juge qui répond toujours PASS obtiendrait 73% d'accord sur ce split test.

Par origine (seuil par défaut) :

| Juge | Origine | n | Accord | Kappa |
|---|---|---|---|---|
| mistral | A-baseline | 8 | 62% | -0.20 |
| mistral | B-bad_chunking | 9 | 33% | -0.42 |
| mistral | C-weak_embedding | 13 | 46% | -0.18 |
| mistral | D-low_k | 9 | 78% | 0.00 |
| mistral | E-noisy_context | 9 | 78% | 0.40 |
| mistral | F-broken_retrieval | 7 | 43% | n/a |
| mistral | G-hallucination_prone | 7 | 43% | 0.00 |
| prometheus | A-baseline | 8 | 62% | -0.20 |
| prometheus | B-bad_chunking | 9 | 33% | -0.35 |
| prometheus | C-weak_embedding | 13 | 54% | 0.05 |
| prometheus | D-low_k | 9 | 56% | -0.29 |
| prometheus | E-noisy_context | 9 | 67% | -0.17 |
| prometheus | F-broken_retrieval | 7 | 43% | n/a |
| prometheus | G-hallucination_prone | 7 | 43% | 0.00 |

Analyse exploratoire Q2, taux de PASS par origine (tous splits, seuil par défaut) :

| Juge | Origine | n | PASS humain | PASS juge | Score moyen du juge |
|---|---|---|---|---|---|
| mistral | A-baseline | 17 | 59% | 82% | 0.91 |
| mistral | B-bad_chunking | 17 | 59% | 82% | 0.88 |
| mistral | C-weak_embedding | 18 | 78% | 67% | 0.78 |
| mistral | D-low_k | 17 | 76% | 94% | 0.96 |
| mistral | E-noisy_context | 17 | 76% | 82% | 0.88 |
| mistral | F-broken_retrieval | 17 | 100% | 24% | 0.34 |
| mistral | G-hallucination_prone | 17 | 35% | 100% | 1.00 |
| prometheus | A-baseline | 17 | 59% | 94% | 0.84 |
| prometheus | B-bad_chunking | 17 | 59% | 65% | 0.68 |
| prometheus | C-weak_embedding | 18 | 78% | 67% | 0.65 |
| prometheus | D-low_k | 17 | 76% | 76% | 0.69 |
| prometheus | E-noisy_context | 17 | 76% | 76% | 0.68 |
| prometheus | F-broken_retrieval | 17 | 100% | 35% | 0.34 |
| prometheus | G-hallucination_prone | 17 | 35% | 88% | 0.84 |

## relevance

Échecs de lecture exclus :

| Juge | n test | Accord (IC 95 %) | Kappa | Faux acceptés | Faux rejetés | Seuil optimisé (dev) | Kappa test au seuil optimisé |
|---|---|---|---|---|---|---|---|
| mistral | 62 | 61% (49%-72%) | -0.04 (pire que le hasard) | 75% | 36% | 0.25 | -0.10 |
| prometheus | 62 | 58% (46%-70%) | 0.02 (léger) | 50% | 41% | 1.0 | 0.03 |

Échecs de lecture comptés comme FAIL (seuil par défaut) :

| Juge | n test | dont échecs de lecture | Accord (IC 95 %) | Kappa | Faux acceptés | Faux rejetés |
|---|---|---|---|---|---|---|
| mistral | 62 | 0 | 61% (49%-72%) | -0.04 (pire que le hasard) | 75% | 36% |
| prometheus | 62 | 0 | 58% (46%-70%) | 0.02 (léger) | 50% | 41% |

Référence : un juge qui répond toujours PASS obtiendrait 94% d'accord sur ce split test.

Par origine (seuil par défaut) :

| Juge | Origine | n | Accord | Kappa |
|---|---|---|---|---|
| mistral | A-baseline | 8 | 75% | 0.33 |
| mistral | B-bad_chunking | 9 | 56% | -0.20 |
| mistral | C-weak_embedding | 13 | 62% | -0.14 |
| mistral | D-low_k | 9 | 67% | n/a |
| mistral | E-noisy_context | 9 | 56% | n/a |
| mistral | F-broken_retrieval | 7 | 14% | n/a |
| mistral | G-hallucination_prone | 7 | 100% | n/a |
| prometheus | A-baseline | 8 | 75% | 0.33 |
| prometheus | B-bad_chunking | 9 | 44% | -0.22 |
| prometheus | C-weak_embedding | 13 | 69% | 0.24 |
| prometheus | D-low_k | 9 | 44% | n/a |
| prometheus | E-noisy_context | 9 | 56% | n/a |
| prometheus | F-broken_retrieval | 7 | 14% | n/a |
| prometheus | G-hallucination_prone | 7 | 100% | n/a |

Analyse exploratoire Q2, taux de PASS par origine (tous splits, seuil par défaut) :

| Juge | Origine | n | PASS humain | PASS juge | Score moyen du juge |
|---|---|---|---|---|---|
| mistral | A-baseline | 17 | 88% | 65% | 0.81 |
| mistral | B-bad_chunking | 17 | 94% | 47% | 0.66 |
| mistral | C-weak_embedding | 18 | 89% | 72% | 0.82 |
| mistral | D-low_k | 17 | 100% | 71% | 0.78 |
| mistral | E-noisy_context | 17 | 100% | 76% | 0.84 |
| mistral | F-broken_retrieval | 17 | 100% | 12% | 0.21 |
| mistral | G-hallucination_prone | 17 | 100% | 100% | 1.00 |
| prometheus | A-baseline | 17 | 88% | 65% | 0.68 |
| prometheus | B-bad_chunking | 17 | 94% | 41% | 0.49 |
| prometheus | C-weak_embedding | 18 | 89% | 61% | 0.58 |
| prometheus | D-low_k | 17 | 100% | 59% | 0.57 |
| prometheus | E-noisy_context | 17 | 100% | 65% | 0.71 |
| prometheus | F-broken_retrieval | 17 | 100% | 6% | 0.06 |
| prometheus | G-hallucination_prone | 17 | 100% | 100% | 0.94 |

## Fonctionnement

| Juge | Critère | Local | Appels | Erreurs | Échecs de parsing | Contexte possiblement tronqué | Latence médiane | Temps total | Stabilité (n) |
|---|---|---|---|---|---|---|---|---|---|
| mistral | faithfulness | oui | 160 | 0 | 0% | 0 | 17.1 s | 45 min | 100% (20) |
| mistral | relevance | oui | 160 | 0 | 0% | 0 | 8.0 s | 23 min | 100% (20) |
| prometheus | faithfulness | oui | 160 | 0 | 0% | 0 | 35.3 s | 101 min | 100% (20) |
| prometheus | relevance | oui | 160 | 0 | 0% | 0 | 24.1 s | 69 min | 100% (20) |
