<p align="center">
  <h1 align="center">⚡ ragnarok-ai</h1>
  <p align="center">
    <strong>Local-first RAG evaluation framework for LLM applications</strong>
  </p>
  <p align="center">
    Evaluate, benchmark, and monitor your RAG pipelines — 100% locally, no API keys required.
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL--3.0-green.svg" alt="License: AGPL-3.0"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/type%20checked-mypy-blue.svg" alt="Type Checked: mypy"></a>
</p>

<p align="center">
  <a href="#-the-problem">Problem</a> •
  <a href="#-the-solution">Solution</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## 🔥 The Problem

Building RAG systems is easy. **Knowing if they actually work is hard.**

Current evaluation tools are either:

| Tool | Issue |
|------|-------|
| **Giskard** | Heavy, slow (45-60 min scans), loses progress on crash, enterprise-focused |
| **RAGAS** | Requires OpenAI API keys, no local-first option |
| **Manual testing** | Doesn't scale, not reproducible |

**You need a tool that:**
- ✅ Runs 100% locally (Ollama, local models)
- ✅ Evaluates fast with checkpointing (no lost progress)
- ✅ Integrates with your existing stack (LangChain, LangGraph)
- ✅ Fits in CI/CD pipelines
- ✅ Doesn't require a PhD to use

---

## 💡 The Solution

**ragnarok-ai** is a lightweight, local-first framework to evaluate RAG pipelines.

```python
from ragnarok_ai import evaluate, generate_testset

# Generate test questions from your knowledge base
testset = await generate_testset(
    knowledge_base="./docs/",
    num_questions=50,
    types=["simple", "multi_hop", "adversarial"],
    llm="ollama/mistral",
    checkpoint=True,  # Resume if interrupted
)

# Evaluate your RAG pipeline
results = await evaluate(
    rag_pipeline=my_rag,
    testset=testset,
    metrics=["retrieval", "faithfulness", "relevance"],
    llm="ollama/mistral",
)

# Get actionable insights
results.summary()
# ┌─────────────────┬───────┬────────┐
# │ Metric          │ Score │ Status │
# ├─────────────────┼───────┼────────┤
# │ Retrieval P@10  │ 0.82  │ ✅      │
# │ Faithfulness    │ 0.74  │ ⚠️      │
# │ Relevance       │ 0.89  │ ✅      │
# │ Hallucination   │ 0.12  │ ✅      │
# └─────────────────┴───────┴────────┘

results.export("report.html")
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🏠 100% Local** | Runs entirely on your machine with Ollama. No OpenAI, no API keys, no data leaving your network. |
| **⚡ Fast & Resilient** | Built-in checkpointing — crash mid-evaluation? Resume exactly where you left off. |
| **🔌 Framework Agnostic** | Works with LangChain, LangGraph, LlamaIndex, or your custom RAG. |
| **📊 Comprehensive Metrics** | Retrieval quality, faithfulness, relevance, hallucination detection, latency tracking. |
| **🧪 Test Generation** | Auto-generate diverse test sets from your knowledge base. |
| **🚀 CI/CD Ready** | CLI-first design, JSON output, exit codes for pipeline integration. |
| **🪶 Lightweight** | Minimal dependencies. No torch/transformers in core. |

---

## 🆚 Comparison

| Feature | ragnarok-ai | Giskard | RAGAS |
|---------|-------------|---------|-------|
| 100% Local | ✅ | ⚠️ Partial | ❌ |
| Checkpointing | ✅ | ❌ | ❌ |
| Fast evaluation | ✅ | ❌ (45-60 min) | ✅ |
| CLI support | ✅ | ❌ | ❌ |
| LangChain integration | ✅ | ✅ | ✅ |
| Minimal deps | ✅ | ❌ | ⚠️ |
| Free & OSS | ✅ AGPL-3.0 | ⚠️ Open-core | ✅ Apache-2.0 |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install

```bash
# Clone
git clone https://github.com/2501Pr0ject/ragnarok-ai.git
cd ragnarok-ai

# Install with uv
uv venv && source .venv/bin/activate
uv pip install -e ".[ollama,qdrant]"
```

### Run your first evaluation

```bash
# CLI (coming soon)
ragnarok evaluate --rag ./my_rag.py --docs ./knowledge_base/ --output report.html

# Or in Python
python examples/basic_evaluation.py
```

---

## 📦 Installation

### Using uv (recommended)

```bash
uv venv
source .venv/bin/activate  # Linux/macOS
uv pip install -e "."
```

### Optional dependencies

```bash
# LLM providers
uv pip install -e ".[ollama]"      # Ollama support

# Vector stores  
uv pip install -e ".[qdrant]"      # Qdrant support

# RAG frameworks
uv pip install -e ".[langchain]"   # LangChain/LangGraph support

# Everything
uv pip install -e ".[all]"

# Development
uv pip install -e ".[dev]"
pre-commit install
```

---

## 🎯 Use Cases

### Continuous RAG Testing in CI/CD

```yaml
# .github/workflows/rag-tests.yml
- name: Evaluate RAG Quality
  run: |
    ragnarok evaluate \
      --config ragnarok.yaml \
      --fail-under 0.8 \
      --output results.json
```

### Compare Embedding Models

```python
configs = [
    {"embedder": "nomic-embed-text", "chunk_size": 512},
    {"embedder": "mxbai-embed-large", "chunk_size": 256},
]

results = await benchmark(
    rag_factory=create_rag,
    configs=configs,
    testset=testset,
)
results.compare()  # Side-by-side comparison
```

### Monitor Production Quality

```python
# Track quality drift over time
metrics = await evaluate(rag, production_queries)
metrics.log_to("./metrics/")  # Time-series storage
```

---

## 📊 Metrics

### Retrieval Metrics
- **Precision@K** — Relevant docs in top K results
- **Recall@K** — Coverage of relevant docs
- **MRR** — Mean Reciprocal Rank
- **NDCG** — Normalized Discounted Cumulative Gain

### Generation Metrics
- **Faithfulness** — Is the answer grounded in retrieved context?
- **Relevance** — Does the answer address the question?
- **Hallucination** — Does the answer contain fabricated info?
- **Completeness** — Are all aspects of the question covered?

### System Metrics
- **Latency** — End-to-end response time
- **Token usage** — Cost tracking for LLM calls

---

## 🗺️ Roadmap

### v0.1 — Foundation (current)
- [x] Project setup & architecture
- [ ] Core retrieval metrics (precision, recall, MRR)
- [ ] Ollama adapter
- [ ] Qdrant adapter
- [ ] Basic CLI
- [ ] JSON reporter

### v0.2 — Generation Metrics
- [ ] Faithfulness evaluator
- [ ] Relevance evaluator
- [ ] Hallucination detection
- [ ] HTML report with visualizations

### v0.3 — Test Generation
- [ ] Synthetic question generation
- [ ] Multi-hop question support
- [ ] Adversarial question generation
- [ ] Checkpointing system

### v0.4 — Framework Adapters
- [ ] LangChain integration
- [ ] LangGraph integration
- [ ] Custom RAG support

### Future
- [ ] Production monitoring dashboard
- [ ] Rust acceleration for hot paths
- [ ] VS Code extension

---

## 🏗️ Project Structure

```
ragnarok-ai/
├── src/ragnarok_ai/
│   ├── core/           # Types, protocols, exceptions
│   ├── evaluators/     # Metric implementations
│   ├── generators/     # Test set generation
│   ├── adapters/       # LLM, vector store, framework adapters
│   ├── reporters/      # Output formatters (JSON, HTML, console)
│   └── cli/            # Command-line interface
├── tests/              # Test suite (pytest)
├── examples/           # Usage examples
├── benchmarks/         # Performance benchmarks
└── docs/               # Documentation
```

---

## 🛠️ Development

```bash
# Setup
uv pip install -e ".[dev]"
pre-commit install

# Run checks
pytest                    # Tests
pytest --cov=ragnarok_ai  # With coverage
ruff check . --fix        # Lint
ruff format .             # Format  
mypy src/                 # Type check
```

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas where help is appreciated:**
- Additional metrics implementations
- Framework adapters (Haystack, LlamaIndex)
- Documentation & examples
- Performance optimizations

---

## 📄 License

This project is licensed under the [AGPL-3.0 License](LICENSE).

**Why AGPL?** To ensure improvements stay open-source and prevent the "open-core bait-and-switch" model where companies take OSS code and monetize without contributing back.

---

## 🙏 Acknowledgments

Inspired by the frustrations with existing RAG evaluation tools. Built to be what [Giskard](https://github.com/Giskard-AI/giskard) should have been — fast, local, and developer-friendly.

---

<p align="center">
  <sub>Built with ❤️ in Lyon, France</sub>
</p>
