# RAGnarok-AI

**Local-first RAG evaluation framework for LLM applications**

Evaluate, benchmark, and monitor your RAG pipelines — 100% locally, no API keys required.

---

## Why RAGnarok-AI?

Building RAG systems is easy. **Knowing if they actually work is hard.**

| Tool | Issue |
|------|-------|
| Giskard | Heavy, slow (45-60 min scans), loses progress on crash |
| RAGAS | Requires OpenAI API keys, no local-first option |
| Manual testing | Doesn't scale, not reproducible |

RAGnarok-AI solves this with:

- **100% Local** — Runs entirely with Ollama, no data leaves your network
- **Fast & Resilient** — Built-in checkpointing, resume on crash
- **Framework Agnostic** — Works with LangChain, LangGraph, LlamaIndex
- **CI/CD Ready** — CLI-first design, JSON output, exit codes

---

## Quick Example

```python
from ragnarok_ai import evaluate, generate_testset

# Generate test questions from your knowledge base
testset = await generate_testset(
    knowledge_base="./docs/",
    num_questions=50,
    llm="ollama/mistral",
    checkpoint=True,
)

# Evaluate your RAG pipeline
results = await evaluate(
    rag_pipeline=my_rag,
    testset=testset,
    metrics=["retrieval", "faithfulness", "relevance"],
)

# Get actionable insights
results.summary()
```

---

## Performance

Benchmarked on Apple M2 16GB, Python 3.10:

**Retrieval Metrics:** ~24,000 queries/sec

| Queries | Time | Peak RAM |
|---------|------|----------|
| 50 | 0.002s | 0.02 MB |
| 500 | 0.021s | 0.03 MB |
| 5000 | 0.217s | 0.17 MB |

**LLM-as-Judge (Prometheus 2):**

| Criterion | Avg Time |
|-----------|----------|
| Faithfulness | ~25s |
| Relevance | ~22s |
| Hallucination | ~28s |

Retrieval is pure computation — instant. LLM-as-Judge is the bottleneck (~25s/eval), but runs 100% local.

---

## Key Features

| Feature | Description |
|---------|-------------|
| 100% Local | Ollama-powered, no API keys required |
| LLM-as-Judge | Prometheus 2 evaluation: faithfulness, relevance, hallucination |
| Cost Tracking | Track token usage. Local models = $0.00 |
| Checkpointing | Resume on crash, no lost progress |
| Framework Agnostic | LangChain, LangGraph, LlamaIndex, custom RAG |
| CI/CD Ready | CLI-first, JSON output, GitHub Action |

---

## Installation

```bash
pip install ragnarok-ai
```

With optional dependencies:

```bash
pip install ragnarok-ai[ollama,qdrant]
```

See [Installation](getting-started/installation.md) for details.

---

## Next Steps

- [Installation](getting-started/installation.md) — Set up RAGnarok-AI
- [Quick Start](getting-started/quickstart.md) — Run your first evaluation
- [CLI Reference](ci-cd/cli-reference.md) — Command-line interface
- [GitHub Action](ci-cd/github-action.md) — CI/CD integration
