# RAGnarok-AI

**Local-first RAG evaluation framework for LLM applications**

Evaluate, benchmark, and monitor your RAG pipelines — 100% locally, no API keys required.

---

## Why RAGnarok-AI?

Building RAG systems is easy. **Knowing if they actually work is hard.**

RAGnarok-AI is built on a few principles:

- **Local-first, always** — every feature works with local models (Ollama). No API key is ever required, and air-gapped deployment is a first-class target.
- **Your data stays yours** — nothing leaves your machine. Production traces hash queries by default; capturing text is opt-in and PII-scrubbed client-side.
- **Trust is measured, not assumed** — judge calibration quantifies agreement with *your* labels (kappa, error rates, threshold tuning) instead of asking you to believe a score.
- **Resumable by design** — long local evaluations crash. Checkpointing means you never lose progress.
- **Engineering tool, not a notebook** — CLI-first, JSON output, exit codes, `--fail-under` CI gates, regression detection, Prometheus export, Helm chart.

---

## Quick Example

```bash
# Generate a test set from your knowledge base
ragnarok generate --docs ./knowledge/ --num 50 --output testset.json
```

```python
from ragnarok_ai import evaluate
from ragnarok_ai.generators import load_testset

testset = load_testset("testset.json")

# my_rag: any object with an async query() method (see Adapters)
results = await evaluate(my_rag, testset)
print(results.summary())
```

Or try it instantly with no setup:

```bash
ragnarok evaluate --demo
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
| LLM-as-Judge | Prometheus 2 evaluation: faithfulness, relevance, hallucination, completeness — with calibration against your own labels |
| Production Intelligence | Drift detection against a recorded baseline, live A/B testing, root-cause diagnosis of failures |
| Production Monitoring | Trace collection, Prometheus metrics export, latency and success-rate tracking |
| Test Generation | Synthetic, adversarial, and multi-hop test sets from your knowledge base — or mined from real production traffic |
| Medical Mode | Clinical abbreviation normalization with optional SLM disambiguation |
| Cost Tracking | Track token usage. Local models = $0.00 |
| Checkpointing | Resume on crash, no lost progress |
| Jupyter Integration | Rich HTML display in notebooks with metrics visualization |
| Framework Agnostic | LangChain, LangGraph, LlamaIndex, DSPy, Haystack, Semantic Kernel, or custom RAG |
| CI/CD Ready | CLI-first, JSON output, exit codes, GitHub Action, streaming evaluation with a live terminal UI |
| Enterprise Ready | Kubernetes Helm charts, air-gapped deployment, data sovereignty |
| Lightweight | Minimal dependencies. No torch/transformers in core |

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
- [User Guide](user-guide/evaluation.md) — Evaluation, diagnosis, judging, calibration, drift, A/B testing
- [CLI Reference](ci-cd/cli-reference.md) — Command-line interface
- [GitHub Action](ci-cd/github-action.md) — CI/CD integration
