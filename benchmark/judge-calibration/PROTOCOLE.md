# Calibration d'un juge LLM local — protocole (annotateur unique)

## 1. Question

Un juge local (Prometheus 2 7B, quantifié Q5_K_M, ~4,8 Go, exécutable sur un MacBook Air M2 16 Go) est-il suffisamment d'accord avec un humain pour servir de garde-fou automatique sur un RAG ?

### Critères de décision, fixés avant de voir les résultats

| Résultat sur le split test (faithfulness, toutes configurations confondues) | Conclusion |
|---|---|
| Kappa ≥ 0,6 **et** faux acceptés ≤ 20 % | Utilisable comme garde-fou bloquant en CI |
| Kappa entre 0,4 et 0,6 | Utilisable en mode advisory (signal, pas blocage) |
| Kappa < 0,4 | Pas fiable seul : relecture humaine obligatoire |

Ces seuils sont écrits **avant** l'expérience et ne doivent pas être modifiés après coup. Un résultat négatif est publiable.

## 2. Périmètre

Volontairement réduit pour être tenable par une seule personne :

| Élément | Choix |
|---|---|
| Corpus | Le jeu du pilote (210 items : 30 questions × 7 configurations A-G) |
| Échantillons | 120 sorties réelles du pilote, stratifiées : ~17 par configuration (une configuration à 18), équilibrées par type de question dans chaque configuration. Les échecs viennent des configurations dégradées (B-G), pas de négatifs synthétiques |
| Critères | faithfulness et relevance |
| Juges | Prometheus 2 7B Q5_K_M et `mistral:7b` (Mistral-7B-Instruct v0.3) ; en option, `llama3.1:8b` comme troisième juge exploratoire, et un juge API **uniquement si le corpus est public** (exploratoire lui aussi). Le générateur du pilote, `qwen2.5:7b-instruct`, n'est pas utilisé comme juge : il jugerait ses propres réponses |
| Annotation | Un annotateur unique, à l'aveugle (ordre mélangé, configuration masquée), puis une seconde passe sur 20 %, au moins 14 jours plus tard |
| Référence | « Toujours PASS », calculée automatiquement |

## 3. Préparation des données

1. **Pilote converti** dans `pilot.jsonl`, un échantillon par ligne :
   ```json
   {"id": "p001", "question": "...", "context": "...", "answer": "...", "origin": "natural"}
   ```
   `context` est **exactement** ce que le générateur a reçu : les passages récupérés, dans l'ordre, joints par `"\n\n---\n\n"`, comme dans `run_pilot.py`. Vérifié pour les 210 items : le prompt reconstruit a la même empreinte SHA-256 que celle enregistrée par le run (`prompt_sha256`), et aucun prompt n'a pu être tronqué (au plus 1 882 tokens, prompt et réponse compris, sous la limite par défaut d'Ollama de 2 048).
   Les identifiants (`p001` à `p210`) sont opaques et attribués dans un ordre mélangé : ils ne révèlent ni la configuration ni la question. La correspondance est dans `pilot_key.json`, **à ne pas ouvrir avant la fin de l'annotation**.
   Commande, depuis `benchmark/judge-calibration/` :
   ```bash
   python convert_pilot.py ../pilot-run/pilot-20260904T145740Z ../rag_configs/configs.yaml pilot.jsonl pilot_key.json
   ```
   `pilot_key.json` n'est pas versionné tant que l'annotation n'est pas terminée.
2. **Annotations de la plateforme externe :** aucune à récupérer, la collecte est fermée (aucun lot terminé). Toute l'annotation est faite avec `label.py`.
3. **Données juge du pilote :** aucune. Le pilote n'a appelé aucun juge (`run_pilot.py` ne fait que la récupération et la génération). Les juges sont lancés pour la première fois avec `run_calibration.py`, après l'annotation.
4. **Sélection :**
   ```bash
   python select_samples.py pilot.jsonl pilot_key.json samples.jsonl --n 120
   ```
   Graine par défaut 7. Tous les échantillons gardent `"origin": "natural"` pendant l'annotation.

## 4. Calendrier

| Quand | Quoi | Durée |
|---|---|---|
| Semaine 1, jours 1 à 3 | Annotation en 4 sessions d'environ 30 échantillons, une session par demi-journée au maximum | 4 × 1 h |
| Fin de semaine 1 | Levée de l'aveugle, puis run des juges, la nuit | Non surveillé |
| Semaine 3, au moins 14 jours après la fin de la première passe | Seconde passe sur 20 % (24 échantillons) | 30 min |
| Semaine 3 | Analyse et lecture des désaccords | 2 h |

Ne pas dépasser une heure par session : la fatigue fait dériver le jugement, ce qui fausse exactement ce qu'on mesure. `label.py` propose une pause toutes les 50 minutes et enregistre le temps passé par échantillon (à citer dans l'article).

## 5. Annotation

- **Avant tout passage d'un juge.** Ne regarder aucun score de juge avant d'avoir terminé.
- **À l'aveugle :** l'ordre est mélangé et l'origine masquée.
- **Consignes figées** (version 2.0 dans `common.py`), alignées sur `benchmark/ANNOTATION.md` de l'étude : faithfulness stricte (toute affirmation non appuyée par le contexte = FAIL, y compris une précision ajoutée), abstention justifiée = PASS en relevance (si le contexte ne contient effectivement pas l'information). Elles sont plus strictes que le barème du juge (PASS = 4/5) : le seuil du juge est optimisé sur le split dev.
- **Ce qui est affiché :** la question, le contexte et la réponse uniquement. Ni réponse de référence, ni type de question, ni configuration.

```bash
python label.py samples.jsonl labels.jsonl --criteria faithfulness relevance
```

**Seconde passe**, au moins 14 jours après la fin de la première passe (§6.2 du protocole d'étude) :

```bash
python label.py samples.jsonl labels_pass2.jsonl --second-pass 0.2 --from-labels labels.jsonl
```

L'accord entre tes deux passes est le **plafond** : un juge ne peut pas être plus d'accord avec toi que toi avec toi-même.

**Fin de l'aveugle :** une fois la première passe terminée, et seulement alors, injecter la configuration comme `origin` :

```bash
python reveal_origin.py samples.jsonl pilot_key.json labels.jsonl samples_revealed.jsonl
```

Le script refuse de s'exécuter tant qu'un échantillon n'est pas annoté. Les juges et l'analyse utilisent ensuite `samples_revealed.jsonl`. La seconde passe se fait toujours sur `samples.jsonl`, sans la configuration.

## 6. Exécution des juges

```bash
pip install "ragnarok-ai[ollama]>=1.12.2"   # 1.12.2 : gabarit natif de Prometheus 2, échec de lecture = erreur explicite
ollama pull hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M
ollama pull mistral:7b
ollama pull llama3.1:8b   # optionnel, juge exploratoire

# Run principal, reprise automatique en cas d'interruption
python run_calibration.py --samples samples_revealed.jsonl --labels labels.jsonl --out results \
  --judge prometheus=hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M \
  --judge mistral=mistral:7b \
  --criteria faithfulness relevance
# Optionnel : ajouter --judge llama=llama3.1:8b (juge exploratoire)

# Stabilité : 3 passages sur 20 échantillons du split test
python run_calibration.py ... --only-split test --limit 20 --repeats 3

# Rapport final
python run_calibration.py ... --analyze-only --second-pass labels_pass2.jsonl
```

Réglages par défaut des juges locaux : température 0, seed 42, `num_ctx` 8192. Ils sont passés à Ollama par le client de `run_calibration.py` : l'adaptateur Ollama utilisé par défaut par `LLMJudge` n'envoie ni température, ni seed, ni `num_ctx`, et ne doit pas être utilisé ici. Le script affiche une estimation du temps restant après les premiers appels. Pour mesurer l'effet de la température, utiliser un `--out` séparé.

## 7. Analyse

**Confirmatoire :** la fiabilité du juge (Q1) sur le split test, jugée selon les critères de décision du §1, pour Prometheus 2 et `mistral:7b`. Les analyses confirmatoires de discrimination entre configurations (Q2), de reproductibilité à deux runs complets (Q3) et de concordance entre frameworks (§7.3 du protocole d'étude) sont reportées à une phase ultérieure.

**Exploratoire :** Q2 sous forme de taux de PASS du juge par configuration, comparés au taux de PASS humain ; la stabilité des juges (3 passages sur 20 échantillons) ; tout juge optionnel (`llama3.1:8b`, juge API). Ces résultats sont présentés comme exploratoires et ne servent pas à la conclusion du §1.

`results/summary.md` fournit les tableaux pour l'article :

- par juge et par critère : accord avec IC 95 %, kappa, faux acceptés, faux rejetés ;
- seuil optimisé sur le split dev, mesuré sur le split test ;
- résultats par configuration (A-G), qui montrent si l'accord dépend du type de défaut. **Le chiffre principal est celui de l'ensemble du split test** : toutes les sorties sont réelles, aucune n'est un négatif synthétique ;
- analyse exploratoire Q2 : taux de PASS humain et taux de PASS du juge par configuration, sur tous les échantillons annotés ;
- fonctionnement : échecs de parsing, contexte possiblement tronqué (doit valoir 0), latence, stabilité ;
- accord entre annotations humaines.

Lire ensuite **tout** `results/disagreements.jsonl` et classer chaque désaccord en 3 à 5 catégories (« juge trop indulgent », « réponse en français mal jugée », « erreur d'annotation de ma part »…).

## 8. Limites à déclarer dans l'article

- **Annotateur unique, auteur de l'outil.** Compensé par l'annotation à l'aveugle, les critères fixés à l'avance, la seconde passe et la publication des données. Ce n'est pas l'équivalent d'un accord inter-annotateurs, et il faut l'écrire. La campagne d'annotation externe a été fermée sans lot terminé.
- **Échecs issus des configurations dégradées :** la proportion de FAIL dépend des défauts des configurations B-G et des questions, pas d'un dosage choisi. Elle est publiée avec les résultats.
- **Juges apparentés :** Prometheus 2 7B est dérivé de Mistral-7B-Instruct v0.2, et le tag Ollama `mistral:7b` correspond à Mistral-7B-Instruct v0.3. Les deux juges sont donc de la même famille : leurs erreurs peuvent être corrélées, et leur accord mutuel ne vaut pas une validation indépendante.
- **Langue :** contenu en français, prompts du juge en anglais.
- **Taille :** environ 60 échantillons dans le split test, soit un IC d'environ ±12 points sur l'accord. Toujours afficher les IC.

## 9. Publication des données

Publier `samples_revealed.jsonl`, les fichiers d'annotations et `results/raw_judgments.jsonl` comme dataset dans l'organisation Hugging Face `ragnarok-ai`. N'importe qui peut alors recalculer les chiffres avec `--analyze-only`, ou réannoter pour contester les tiens.
