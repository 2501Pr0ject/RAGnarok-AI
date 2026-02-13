# Evaluation

RAGnarok-AI provides comprehensive metrics for evaluating RAG pipelines.

---

## Retrieval Metrics

Measure how well your retrieval system finds relevant documents.

### Precision@K

Fraction of retrieved documents that are relevant.

```python
from ragnarok_ai.evaluators import precision_at_k

retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
relevant = ["doc_1", "doc_3", "doc_7"]

p = precision_at_k(retrieved, relevant, k=5)
# p = 2/5 = 0.4 (doc_1 and doc_3 are relevant)
```

### Recall@K

Fraction of relevant documents that were retrieved.

```python
from ragnarok_ai.evaluators import recall_at_k

r = recall_at_k(retrieved, relevant, k=5)
# r = 2/3 = 0.67 (found 2 of 3 relevant docs)
```

### MRR (Mean Reciprocal Rank)

Position of the first relevant document.

```python
from ragnarok_ai.evaluators import mrr

m = mrr(retrieved, relevant)
# m = 1.0 (first relevant doc is at position 1)
```

### NDCG@K (Normalized Discounted Cumulative Gain)

Measures ranking quality with position-based discounting.

```python
from ragnarok_ai.evaluators import ndcg_at_k

n = ndcg_at_k(retrieved, relevant, k=5)
# Higher score = relevant docs ranked higher
```

---

## Full Retrieval Evaluation

Use `evaluate_retrieval` for all metrics at once:

```python
from ragnarok_ai.core.types import Query, Document, RetrievalResult
from ragnarok_ai.evaluators.retrieval import evaluate_retrieval

query = Query(
    text="What is Python?",
    ground_truth_docs=["doc_1", "doc_3"],
)

retrieved_docs = [
    Document(id="doc_1", content="Python is a programming language..."),
    Document(id="doc_2", content="Java is another language..."),
    Document(id="doc_3", content="Python was created by Guido..."),
]

result = RetrievalResult(
    query=query,
    retrieved_docs=retrieved_docs,
    scores=[0.95, 0.85, 0.80],
)

metrics = evaluate_retrieval(result, k=10)

print(f"Precision@10: {metrics.precision:.2f}")
print(f"Recall@10: {metrics.recall:.2f}")
print(f"MRR: {metrics.mrr:.2f}")
print(f"NDCG@10: {metrics.ndcg:.2f}")
```

---

## Generation Metrics

Evaluate answer quality with LLM-as-Judge. See [LLM-as-Judge](judge.md).

| Metric | Description |
|--------|-------------|
| Faithfulness | Is the answer grounded in context? |
| Relevance | Does the answer address the question? |
| Hallucination | Does the answer contain fabricated info? |
| Completeness | Are all aspects of the question covered? |

---

## Batch Evaluation

Evaluate multiple queries efficiently:

```python
from ragnarok_ai.core.types import TestSet, Query
from ragnarok_ai.evaluators.retrieval import evaluate_retrieval

testset = TestSet(
    queries=[
        Query(text="Question 1", ground_truth_docs=["doc_1"]),
        Query(text="Question 2", ground_truth_docs=["doc_2", "doc_3"]),
        # ...
    ]
)

results = []
for query in testset.queries:
    # Get your RAG results
    response = await my_rag.query(query.text)

    # Evaluate
    result = RetrievalResult(
        query=query,
        retrieved_docs=response.retrieved_docs,
        scores=[0.9, 0.8, 0.7],
    )
    metrics = evaluate_retrieval(result, k=10)
    results.append(metrics)

# Aggregate
avg_precision = sum(m.precision for m in results) / len(results)
avg_recall = sum(m.recall for m in results) / len(results)
```

---

## Checkpointing

Resume evaluation on crash:

```python
from ragnarok_ai.checkpoint import CheckpointManager

checkpoint = CheckpointManager("./checkpoints/eval.ckpt")

for i, query in enumerate(testset.queries):
    # Skip already processed
    if checkpoint.is_completed(i):
        continue

    # Evaluate
    result = await evaluate_query(query)

    # Save progress
    checkpoint.save(i, result)
```

---

## Cost Tracking

Track evaluation costs:

```python
results = await evaluate(rag, testset, track_cost=True)

print(results.cost)
# +--------------------+------------+----------+
# | Provider           |     Tokens |     Cost |
# +--------------------+------------+----------+
# | ollama (local)     |     45,230 |    $0.00 |
# | openai             |     12,500 |    $0.38 |
# +--------------------+------------+----------+
```

---

## Interpreting Results

### Retrieval Metrics

| Score | Interpretation |
|-------|----------------|
| > 0.8 | Excellent retrieval |
| 0.6 - 0.8 | Good, room for improvement |
| < 0.6 | Needs attention |

### LLM-as-Judge

| Score | Verdict |
|-------|---------|
| >= 0.7 | PASS |
| 0.4 - 0.7 | PARTIAL |
| < 0.4 | FAIL |

!!! note "Advisory Metrics"
    LLM-as-Judge scores are advisory. Use them for insights, not hard CI gates.

---

## Next Steps

- [LLM-as-Judge](judge.md) — Prometheus 2 evaluation
- [Benchmarking](benchmarking.md) — Track performance over time
- [CLI Reference](../ci-cd/cli-reference.md) — Command-line interface
