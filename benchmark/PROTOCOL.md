# RAGnarok Evaluation Study — Protocol

**Status: DRAFT v1 — frozen before any question is written.**
Changes to this protocol after annotation begins must be logged in the
Amendments section with a rationale.

## 1. Research question

> Can a local-first RAG evaluation framework produce **reliable and
> reproducible** quality measurements — and specifically, how closely do
> local LLM judges reproduce human judgment?

This is *not* a study of whether ragnarok-ai scores are "good". It asks
three falsifiable questions:

- **Q1 — Judge reliability.** How well does each local judge agree with
  human labels, per criterion? (Cohen's kappa, error rates by direction,
  optimal pass thresholds — measured with `ragnarok_ai.calibration`.)
- **Q2 — Discrimination.** Given RAG systems that are *deliberately*
  degraded in known ways, do the evaluation scores rank them correctly?
- **Q3 — Reproducibility.** Do two independent runs of the full protocol
  produce the same conclusions (scores within reported variance)?

## 2. Corpora

Four public technical documentation sets, chosen for verifiable answers,
redistribution-friendly licenses, structure, and conceptual cross-links:

| Corpus | Source | Pinned by | License |
|---|---|---|---|
| `docker` | github.com/docker/docs (selected guides + Dockerfile reference) | git commit hash | Apache-2.0 |
| `python` | docs.python.org — tutorial + selected HOWTOs (not the full reference) | CPython version (e.g. 3.12.x) | PSF |
| `fastapi` | github.com/fastapi/fastapi `docs/en` | git commit hash | MIT |
| `kubernetes` | github.com/kubernetes/website — Concepts section | git commit hash | CC-BY-4.0 |

Rules:

- Each corpus is a **subset** small enough to re-index in minutes on a
  laptop (target: 100–400 markdown files per corpus).
- The exact file list, source commit, and a SHA-256 of the assembled
  corpus are committed in `corpora/<name>/MANIFEST.yaml`.
- Attribution and license texts ship with each corpus copy.
- For the `contradiction` question type, the corpus may include **two
  versions of the same page** (e.g. two release versions of a reference
  page); both files are flagged `contradiction_pair: <id>` in the
  manifest so the ambiguity is deliberate and documented.

## 3. Question set

### 3.1 Taxonomy

| Type | Definition | Expected behavior |
|---|---|---|
| `direct` | Answer stated verbatim in one chunk | answer |
| `contextual` | Answer requires conditions/qualifiers from one section | answer |
| `multi_chunk` | Answer requires combining ≥2 chunks (compare X vs Y, when to use each) | answer |
| `trap` | Question presupposes something false; docs contradict the premise | answer (correcting the premise) |
| `unanswerable` | The docs do not contain the answer | **abstain** |
| `contradiction` | Two corpus versions give different answers | answer (surfacing the version difference) or abstain with explanation |

### 3.2 Phasing

- **Phase 1 — pilot (30 cases)**: 5 per type. Purpose: validate the
  protocol itself (schema, annotation rules, tooling). Pilot cases and
  their annotations may be discarded or revised; nothing from Phase 1 is
  reported as a result.
- **Phase 2 — main benchmark (100 cases)**: 20 direct, 20 contextual,
  20 multi_chunk, 15 trap, 15 unanswerable, 10 contradiction — balanced
  across the four corpora (±2 per corpus per type).
- **Phase 3 — extension (up to 200, optional)**: only if Phase 2 shows
  instability; may add real mined questions (see 3.4) and new corpora.

### 3.3 Authoring rules

- Every case is written against a **pinned corpus version** and must
  cite its `supporting_chunks` (file + anchor) — empty only for
  `unanswerable`.
- The author must be able to verify the reference answer in the cited
  chunks in under 2 minutes; if not, the case is too ambiguous — rewrite
  or drop it.
- Questions are phrased as a real user would ask (no quoting the docs
  verbatim).
- Each case declares `difficulty` (easy / medium / hard) — an authoring
  judgment, used only for stratified analysis, never as ground truth.

### 3.4 Real-world complement (Phase 3)

Questions mined from real usage (`TestsetMiner`, PII-scrubbed capture)
may be added as a separate `real_world` split (~50 cases). They follow
the same schema but `supporting_chunks` may be annotated post-hoc.
Controlled and real-world splits are always reported separately.

## 4. RAG systems under test

Seven configurations over the same corpora, all local (Ollama), each
defined by a committed config file in `rag_configs/`:

| ID | Name | Deliberate defect |
|---|---|---|
| A | `baseline` | none — sensible defaults (chunk ~1000 chars / 150 overlap, top-k 5, grounded prompt, temperature 0.1) |
| B | `bad_chunking` | 128-char chunks, no overlap |
| C | `weak_embedding` | deliberately weaker/mismatched embedding model |
| D | `low_k` | top-k = 1 |
| E | `noisy_context` | top-k 5 but 2 slots filled with random chunks |
| F | `broken_retrieval` | retrieval returns random chunks only |
| G | `hallucination_prone` | temperature 1.0, prompt lacks grounding instruction |

The generator model is **identical across A–F** (pinned Ollama tag) so
that defects isolate one component. Expected ordering for Q2 (weakest
claims we test): A > {B, C, D, E} > F on retrieval-sensitive criteria;
A > G on faithfulness.

## 5. Judges

Each judge runs through `LLMJudge` with temperature 0 and a pinned
Ollama model digest, on the **same** (question, retrieved context,
answer) triples:

- `prometheus-2:7b` (framework default)
- `qwen2.5:7b`
- `llama3.1:8b`
- `mistral:7b`

Criteria per case: faithfulness, answer relevance, completeness
(hallucination is reported as part of faithfulness analysis).

## 6. Human annotation

### 6.1 What is annotated

For each (case × RAG config) output, the annotator labels **binary
verdicts per criterion** (the format `ragnarok_ai.calibration` consumes):

| Criterion | 1 means |
|---|---|
| `retrieval_relevance` | the retrieved chunks are sufficient to answer |
| `faithfulness` | every claim in the answer is supported by the retrieved context |
| `answer_relevance` | the answer addresses the question asked |
| `completeness` | nothing essential (per the reference answer) is missing |

Plus: `confidence` (low / medium / high) and a free-text `note`
(mandatory whenever any criterion is 0 or confidence is low).

Full decision rules, including the `unanswerable` and `contradiction`
special cases, are in [ANNOTATION.md](ANNOTATION.md) and are **frozen
before annotation starts**.

### 6.2 Annotators and agreement

- **Primary annotator**: the maintainer. This is a stated limitation of
  the study, mitigated as follows.
- **Intra-annotator agreement**: a random 20% subsample is re-annotated
  by the primary annotator **at least 14 days later**, blind to the
  original labels; intra-annotator kappa is reported per criterion.
- **Inter-annotator agreement (target)**: a second annotator labels a
  30-case subsample using ANNOTATION.md only (no discussion); kappa is
  reported. If no second annotator can be recruited, the study says so
  explicitly.
- Annotation happens **blind to judge outputs** and blind to which RAG
  config produced the answer.

### 6.3 Ordering and hygiene

- Outputs are annotated in randomized order (seeded shuffle, seed
  committed).
- The annotator may mark a case `invalid` (bad question, corpus error);
  invalid cases are excluded from all analyses and listed in the report.

## 7. Analyses

1. **Judge reliability (Q1)** — per judge × criterion:
   `JudgeCalibrator` over the human labels → kappa (+ Landis & Koch
   band), accuracy with 95% Wilson CI, false-accept / false-reject
   rates, recommended threshold. Stratified by question type and corpus.
2. **Discrimination (Q2)** — per metric (retrieval metrics and each
   judge × criterion): does the metric rank config A above each degraded
   config? Report per-pair ordering correctness and score gaps; a
   metric "discriminates" a defect if the gap exceeds its observed
   run-to-run variance.
3. **Cross-framework concordance (Q1b, Phase 2+)** — the same triples
   scored by RAGAS and DeepEval (local judge configuration where
   supported); correlation of each framework's scores with human labels.
   Framing: *concordance with humans*, never "our score is higher".
4. **Reproducibility (Q3)** — the full pipeline (index → answer →
   judge) is run **twice** from scratch; per-metric deltas are reported,
   and every Q1/Q2 conclusion must hold in both runs.

## 8. Reproducibility requirements

Everything needed to re-run the study is committed:

- corpus manifests (source commit + SHA-256), chunking and retrieval
  configs, RAG configs, judge configs with model digests
  (`ollama show --modelfile` output), prompts;
- seeds for every stochastic step (sampling, shuffling, generation);
- ragnarok-ai version (the study runs on a tagged release);
- all raw outputs and annotations (`results/`, `annotations/`) as
  versioned JSON/YAML;
- hardware note (machine, RAM, quantization) with observed runtimes.

## 9. Repository layout

```
benchmark/
├── PROTOCOL.md            # this file
├── ANNOTATION.md          # frozen annotation rules
├── schemas/
│   ├── case.schema.json   # benchmark case format
│   └── annotation.schema.json
├── corpora/<name>/        # MANIFEST.yaml + licensed corpus copy
├── questions/             # cases (YAML), per phase
├── rag_configs/           # A–G definitions
├── judges/                # judge definitions (model digests, prompts)
├── annotations/           # human labels (append-only)
└── results/               # raw runs + analysis outputs
```

## 10. Publication

Outputs: (1) this repository, self-contained and re-runnable; (2) a
technical write-up — working title *“Can We Trust Local LLM Judges? A
Reproducible Benchmark of Local-First RAG Evaluation”* — reporting
methodology, results, limitations (single primary annotator, corpus
domain skew, model quantization effects), and negative results with the
same prominence as positive ones.

## Amendments

| Date | Change | Rationale |
|---|---|---|
| 2026-09-04 | Case schema: added required `experimental_target` field (analysis-only, never shown to RAG or judges). Pre-annotation. | Each pilot case declares what it is designed to test, separately from the question, to make pilot analysis explicit. |
| 2026-09-04 | fastapi_pair_02 nature corrected (metadata only; corpus files untouched). Pre-annotation. | Verification against the actual files showed 0.100.0 already documented pydantic-settings; the recorded "v1→v2 flip" nature was wrong. The pair is kept but not used for contradiction cases. |
| 2026-09-04 | Added **external distributed annotation** as an additional agreement pass (round `external`), alongside — not replacing — the primary/reannotation/second-annotator design of §6.2. Schema extended with `source`, `batch_id`, `study_version`, `ambiguity_flag`, optional `qualification` (no PII), `started_at`/`completed_at`. External annotators see the reference answer (presented as "Reference information") but **not** `expected_behavior`, `question_type`, or any configuration data. Soft launch on the pilot: ~70 stratified cases × 3 independent annotations in batches of 15; large-scale recruitment reserved for Phase 2. Blindness against a public repository is best-effort and documented as such. | Multiple independent human judgments enable inter-annotator agreement (Fleiss' kappa / Krippendorff's alpha) and turn the single-annotator limitation into a measured quantity. Decisions D1–D8 validated by the maintainer on 2026-09-04. |
