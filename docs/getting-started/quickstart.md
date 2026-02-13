# Quick Start

Run your first RAG evaluation in minutes.

---

## Demo Evaluation (CLI)

The fastest way to see RAGnarok-AI in action:

```bash
ragnarok evaluate --demo
```

Output:

```
  RAGnarok-AI Demo Evaluation
  ========================================

  Loading NovaTech example dataset...
  Documents: 25
  Queries: 15

  Simulating RAG retrieval (realistic noise)...

    [+] Query  1/15: P=0.67 R=0.50 MRR=1.00
    [+] Query  2/15: P=0.75 R=0.75 MRR=1.00
    ...

  ----------------------------------------
  Results Summary
  ----------------------------------------
    Precision@10:  0.6533
    Recall@10:     0.5667
    MRR:           0.8500
    NDCG@10:       0.6234
  ----------------------------------------
    Average:       0.6734
```

### Options

```bash
# Save results to file
ragnarok evaluate --demo --output results.json

# Limit queries (quick test)
ragnarok evaluate --demo --limit 5

# Set pass/fail threshold
ragnarok evaluate --demo --fail-under 0.7

# JSON output (for CI/CD)
ragnarok evaluate --demo --json
```

---

## Python API

### Basic Evaluation

```python
import asyncio
from ragnarok_ai.data import load_example_dataset
from ragnarok_ai.evaluators import precision_at_k, recall_at_k, mrr, ndcg_at_k

# Load example dataset
dataset = load_example_dataset("novatech")
testset = dataset.to_testset()

# Evaluate retrieval
for query in testset.queries:
    retrieved_ids = ["doc_1", "doc_2", "doc_3"]  # Your RAG results
    relevant_ids = query.ground_truth_docs

    p = precision_at_k(retrieved_ids, relevant_ids, k=10)
    r = recall_at_k(retrieved_ids, relevant_ids, k=10)
    m = mrr(retrieved_ids, relevant_ids)
    n = ndcg_at_k(retrieved_ids, relevant_ids, k=10)

    print(f"P@10={p:.2f} R@10={r:.2f} MRR={m:.2f} NDCG={n:.2f}")
```

### With Your RAG Pipeline

```python
from ragnarok_ai.core.types import RAGResponse, Document

class MyRAG:
    async def query(self, question: str) -> RAGResponse:
        # Your retrieval logic
        docs = self.retrieve(question)
        answer = self.generate(question, docs)

        return RAGResponse(
            answer=answer,
            retrieved_docs=[
                Document(id=d.id, content=d.text)
                for d in docs
            ],
        )

# Evaluate
rag = MyRAG()
for query in testset.queries:
    response = await rag.query(query.text)
    # Compute metrics...
```

---

## LLM-as-Judge

Evaluate answer quality with Prometheus 2:

```python
from ragnarok_ai.evaluators.judge import LLMJudge

judge = LLMJudge()

result = await judge.evaluate_faithfulness(
    context="Python was created by Guido van Rossum in 1991.",
    question="Who created Python?",
    answer="Guido van Rossum created Python.",
)

print(f"Verdict: {result.verdict}")  # PASS
print(f"Score: {result.score:.2f}")  # 0.85
print(f"Explanation: {result.explanation}")
```

See [LLM-as-Judge](../user-guide/judge.md) for details.

---

## Generate Test Set

Create synthetic test questions from your documents:

```bash
# From demo dataset
ragnarok generate --demo --num 10

# From your documents
ragnarok generate --docs ./knowledge/ --num 50 --output testset.json
```

---

## Configuration File

Create `ragnarok.yaml` for reusable settings:

```yaml
# ragnarok.yaml
testset: ./testset.json
output: ./results.json
fail_under: 0.8
metrics:
  - precision
  - recall
  - mrr
  - ndcg
criteria:
  - faithfulness
  - relevance
ollama_url: http://localhost:11434
```

Run with config:

```bash
ragnarok evaluate --config ragnarok.yaml
```

---

## Next Steps

- [Evaluation Guide](../user-guide/evaluation.md) — Deep dive into metrics
- [LLM-as-Judge](../user-guide/judge.md) — Prometheus 2 evaluation
- [GitHub Action](../ci-cd/github-action.md) — CI/CD integration
