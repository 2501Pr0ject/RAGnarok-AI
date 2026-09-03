# Root-Cause Diagnosis

An evaluation score tells you *that* quality is low. Diagnosis tells you *why* — and what to fix first.

When faithfulness sits at 0.6, the question that matters is: is it a retrieval miss (the right chunk never comes back), a ranking problem (it comes back buried), a chunking issue (the right document, but the answer is split across chunks), or a generation problem (good context, unfaithful answer)? Each cause calls for a completely different fix.

## Quick start

```python
from ragnarok_ai import RAGDiagnostician, evaluate

result = await evaluate(rag_pipeline, testset, metrics=["retrieval"])

diagnostician = RAGDiagnostician()
report = await diagnostician.diagnose(result)
print(report.summary())
```

```
Diagnosed 18/50 failing queries (36% failure rate)

Failure causes:
  retrieval_miss              14  (78%)
  pipeline_error               3  (17%)
  generation_incomplete        1  (6%)

Recommendations:
  1. Relevant documents are not being retrieved: revisit chunk size/overlap, ...
  2. Fix pipeline errors first — they mask every other quality signal.
  ...
```

## The failure taxonomy

| Cause | Signal | Typical fix |
|---|---|---|
| `pipeline_error` | The pipeline raised | Fix the bug first |
| `retrieval_miss` | Ground-truth docs never retrieved | Embeddings, chunking, higher k |
| `retrieval_ranking` | Retrieved but ranked low (low MRR) | Add a reranker |
| `context_insufficient` | Right docs, but chunks lack the answer | Bigger chunks / more overlap |
| `generation_hallucination` | Good context, unsupported claims | Tighter prompt, stronger model |
| `generation_incomplete` | Grounded but partial (or empty) answer | More context to the generator |

## Two tiers

**Heuristic tier (default, zero cost).** Everything decidable from metrics the evaluation already computed: errors, empty answers, and — using `ground_truth_docs` against the retrieved document IDs — misses vs. ranking problems. Generated test sets carry ground truth, so this tier covers the retrieval side out of the box.

**LLM tier (opt-in).** Retrieval can pass while generation fails; deciding that needs to read the text. Provide an LLM and the corpus, and queries that pass retrieval get closed YES/NO checks, stopping at the first failure:

1. *Is the question answerable from this context?* → NO: `context_insufficient` (chunking)
2. *Is every claim in the answer supported?* → NO: `generation_hallucination`
3. *Does the answer fully address the question?* → NO: `generation_incomplete`

```python
from ragnarok_ai.adapters import OllamaLLM

diagnostician = RAGDiagnostician(llm=OllamaLLM(model="mistral"))
report = await diagnostician.diagnose(
    result,
    documents={doc.id: doc for doc in knowledge_base},  # id → Document or str
)
```

The corpus mapping is needed because evaluation results keep retrieved document *IDs*, not contents. An unreachable LLM or an unclear verdict is treated as inconclusive, never as a failure.

## Patterns

`Query.metadata` is cross-referenced with failures, so testsets generated with question types surface where the pipeline struggles:

```python
report.patterns
# {"type": {"simple": 0.05, "multi_hop": 0.61, "adversarial": 0.33}}
```

Here multi-hop questions fail twelve times more often than simple ones — a much more actionable signal than a global average.

## Tuning and output

```python
from ragnarok_ai import DiagnosisThresholds

diagnostician = RAGDiagnostician(
    thresholds=DiagnosisThresholds(recall_pass=0.5, mrr_pass=0.5),
)
```

- `report.diagnoses` — per-query cause with evidence (metric values, missing doc IDs, LLM verdicts) for drill-down
- `report.breakdown` / `report.recommendations` — causes counted and advice ordered by dominant cause
- `report.to_dict()` — JSON-serializable, for CI artifacts or dashboards

!!! note
    Queries without `ground_truth_docs` cannot be diagnosed on the retrieval side; with the LLM tier they are still checked on the generation side.
