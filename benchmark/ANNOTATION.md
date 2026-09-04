# Annotation Guidelines

**Frozen before annotation starts.** Any change after that point goes
through the PROTOCOL.md Amendments table.

You are labeling the output of a RAG system for one benchmark case. You
see: the **question**, the **reference answer** and its
**supporting chunks** (from the case), the **retrieved context**, and
the **generated answer**. You do NOT see which RAG configuration
produced it, nor any judge's verdict.

Label each criterion **0 or 1**. When torn, apply the criterion's
tie-break rule below. Set `confidence` (low/medium/high) and write a
`note` whenever any criterion is 0 or confidence is low.

## Criteria

### retrieval_relevance — "could a competent person answer from these chunks?"

- **1** — the retrieved chunks contain enough information to produce the
  reference answer (extra irrelevant chunks do not hurt).
- **0** — essential information is missing from the retrieved chunks.
- Tie-break: if the answer would necessarily be partial → 0.
- `unanswerable` cases: label 1 iff retrieval returned nothing that
  *appears* to answer (returning loosely related chunks is fine); the
  criterion tests whether retrieval fabricated support, not emptiness.

### faithfulness — "is every claim supported by the retrieved context?"

- **1** — every factual claim in the answer is supported by the
  retrieved chunks (not by your own knowledge of the tool!).
- **0** — at least one claim is unsupported or contradicts the context.
- Tie-break: paraphrase and reasonable summarization are supported;
  added specifics (versions, defaults, flags) not present in context
  are NOT → 0.
- Important: an answer can be *correct in the real world* and still 0
  here if the context does not support it.

### answer_relevance — "does it answer the question asked?"

- **1** — the answer addresses the actual question.
- **0** — off-topic, answers a different question, or restates the
  question without answering.
- Tie-break: a correct abstention ("the documentation does not cover
  this") is **1 for `unanswerable` cases**, 0 otherwise.

### completeness — "is anything essential missing?"

- **1** — the answer covers the essential elements of the reference
  answer (wording may differ; less detail is acceptable).
- **0** — an essential element of the reference answer is absent
  (e.g. only one side of a compare-X-and-Y question).
- Tie-break: judge against the reference answer's essentials, not
  against everything the docs could say.

## Special cases

- **`unanswerable`**: the ideal output abstains. Then
  retrieval_relevance per its rule above, faithfulness 1 (no unsupported
  claims), answer_relevance 1, completeness 1. Any invented answer:
  faithfulness 0.
- **`trap`**: the ideal output corrects the false premise. Accepting the
  premise and building on it → answer_relevance 1 (it engaged the
  question) but faithfulness 0 if the context contradicts it.
- **`contradiction`**: the ideal output either surfaces both versions or
  clearly scopes its answer to one version. Silently picking one value
  as if unambiguous → completeness 0; picking a value the retrieved
  context does not contain → faithfulness 0.
- **Empty answer**: relevance 0, completeness 0; faithfulness 1 (no
  claims made).
- **Broken case** (bad question, corpus error, tooling failure): mark
  `invalid: true` with a note; do not label criteria.

## Hygiene

- Annotate in the provided randomized order; do not reorder.
- Do not consult the live documentation website during annotation — the
  pinned corpus copy is the ground truth.
- Do not revisit earlier labels after seeing later outputs (drift); the
  re-annotation pass exists to measure exactly that.
- Target pace: if a case takes more than ~3 minutes, note why — chronic
  slowness signals an ambiguous case or an unclear rule.
