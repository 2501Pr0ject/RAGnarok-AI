<p align="center">
  <img src="https://raw.githubusercontent.com/2501Pr0ject/RAGnarok-AI/main/assets/logo.png" alt="ragnarok-ai logo" width="300">
</p>

<p align="center">
  <strong>Local-first RAG evaluation framework for LLM applications</strong>
</p>

<p align="center">
  Evaluate, benchmark, and monitor your RAG pipelines — 100% locally, no API keys required.
</p>

<p align="center">
  <a href="https://pypi.org/project/ragnarok-ai/"><img src="https://img.shields.io/pypi/v/ragnarok-ai.svg" alt="PyPI"></a>
  <a href="https://github.com/2501Pr0ject/RAGnarok-AI/actions/workflows/ci.yml"><img src="https://github.com/2501Pr0ject/RAGnarok-AI/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL--3.0-green.svg" alt="License: AGPL-3.0"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/type%20checked-mypy-blue.svg" alt="Type Checked: mypy"></a>
  <a href="https://colab.research.google.com/drive/1BC90iuDMwYi4u9I59jfcjNYiBd2MNvTA?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

<p align="center">
  <a href="#-the-problem">Problem</a> •
  <a href="#-the-solution">Solution</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## The Problem

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

## The Solution

**ragnarok-ai** is a lightweight, local-first framework to evaluate RAG pipelines.

<p align="center">
  <img src="https://raw.githubusercontent.com/2501Pr0ject/RAGnarok-AI/main/assets/overview.png" alt="RAGnarok-AI Overview" width="900">
</p>

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

> **v1.3.1 is now available!** Includes Cost Tracking and Jupyter Integration. Install with `pip install ragnarok-ai`

---

## Key Features

| Feature | Description |
|---------|-------------|
| **100% Local** | Runs entirely on your machine with Ollama. No OpenAI, no API keys, no data leaving your network. |
| **LLM-as-Judge** | Multi-criteria evaluation with Prometheus 2: faithfulness, relevance, hallucination, completeness. |
| **Cost Tracking** | Track token usage and costs. Local models = $0.00, see exactly what cloud APIs cost. |
| **Jupyter Integration** | Rich HTML display in notebooks with metrics visualization. |
| **Fast & Resilient** | Built-in checkpointing — crash mid-evaluation? Resume exactly where you left off. |
| **Framework Agnostic** | Works with LangChain, LangGraph, LlamaIndex, or your custom RAG. |
| **Comprehensive Metrics** | Retrieval quality, faithfulness, relevance, hallucination detection, latency tracking. |
| **Test Generation** | Auto-generate diverse test sets from your knowledge base. |
| **CI/CD Ready** | CLI-first design, JSON output, exit codes for pipeline integration. |
| **Lightweight** | Minimal dependencies. No torch/transformers in core. |

---

## Comparison

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

## Performance

Benchmarked on Apple M2 16GB, Python 3.10:

**Retrieval Metrics:** ~24,000 queries/sec
| Queries | Time   | Peak RAM |
|---------|--------|----------|
| 50      | 0.002s | 0.02 MB  |
| 500     | 0.021s | 0.03 MB  |
| 5000    | 0.217s | 0.17 MB  |

**LLM-as-Judge (Prometheus 2):**
| Criterion | Avg Time |
|-----------|----------|
| Faithfulness | ~25s |
| Relevance | ~22s |
| Hallucination | ~28s |

*Retrieval is pure computation — instant. LLM-as-Judge is the bottleneck (~25s/eval), but runs 100% local.*

[Full benchmarks →](benchmarks/README.md)

---

## Quick Start

**Try it now:** [Open in Google Colab](https://colab.research.google.com/drive/1BC90iuDMwYi4u9I59jfcjNYiBd2MNvTA?usp=sharing)

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install

```bash
pip install ragnarok-ai
```

With optional dependencies:

```bash
pip install ragnarok-ai[ollama,qdrant]
```

### Run your first evaluation

```bash
# CLI demo
ragnarok evaluate --demo

# With options
ragnarok evaluate --demo --output results.json --fail-under 0.7

# Or in Python
python examples/basic_evaluation.py
```

---

## Installation

### Using pip

```bash
pip install ragnarok-ai
```

### Optional dependencies

```bash
# LLM providers
pip install ragnarok-ai[ollama]      # Ollama support
pip install ragnarok-ai[openai]      # OpenAI support
pip install ragnarok-ai[anthropic]   # Anthropic support

# Vector stores
pip install ragnarok-ai[qdrant]      # Qdrant support
pip install ragnarok-ai[chroma]      # ChromaDB support
pip install ragnarok-ai[faiss]       # FAISS support

# RAG frameworks
pip install ragnarok-ai[langchain]   # LangChain/LangGraph support
pip install ragnarok-ai[llamaindex]  # LlamaIndex support
pip install ragnarok-ai[dspy]        # DSPy support

# Observability
pip install ragnarok-ai[telemetry]   # OpenTelemetry tracing

# Everything
pip install ragnarok-ai[all]
```

### Development

```bash
git clone https://github.com/2501Pr0ject/RAGnarok-AI.git
cd RAGnarok-AI
pip install -e ".[dev]"
pre-commit install
```

---

## Use Cases

### Continuous RAG Testing in CI/CD

```yaml
# .github/workflows/rag-tests.yml
- uses: 2501Pr0ject/ragnarok-evaluate-action@v1
  with:
    config: ragnarok.yaml
    threshold: 0.8
    # fail-on-threshold: false (default - advisory only)
    # comment-on-pr: true (default - posts PR comment)
```

The action posts a PR comment distinguishing **deterministic** retrieval metrics from **advisory** LLM-as-Judge scores.

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

## Metrics

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

### LLM-as-Judge (v1.2+)

Use Prometheus 2 for comprehensive, local evaluation:

```python
from ragnarok_ai import LLMJudge

# Initialize judge (uses Prometheus 2 by default)
judge = LLMJudge()

# Evaluate a single response
result = await judge.evaluate_all(
    context="Python was created by Guido van Rossum in 1991.",
    question="Who created Python?",
    answer="Guido van Rossum created Python.",
)

print(f"Overall: {result.overall_verdict} ({result.overall_score:.2f})")
# Overall: PASS (0.85)

print(f"Faithfulness: {result.faithfulness.verdict}")
print(f"Hallucination: {result.hallucination.verdict}")
```

**Performance:**
- ~20-30s per evaluation on Apple M2 16GB
- Prometheus 2 Q5_K_M: ~5GB RAM usage
- `keep_alive` enabled by default (prevents model unloading between requests)

**Installation:**
```bash
# Install Prometheus 2 (~5GB, runs on 16GB RAM)
ollama pull hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M
```

### Medical Mode

Reduce false positives in healthcare RAG evaluation with automatic medical abbreviation normalization.

```python
from ragnarok_ai import LLMJudge

# Enable medical mode
judge = LLMJudge(medical_mode=True)

# "CHF" and "congestive heart failure" are now treated as equivalent
result = await judge.evaluate_faithfulness(
    context="Patient diagnosed with CHF.",
    question="What condition does the patient have?",
    answer="Patient has congestive heart failure.",
)
# Without medical_mode: may flag as unfaithful (text mismatch)
# With medical_mode: correctly identifies as faithful
```

**Features:**
- 350+ medical abbreviations (CHF, MI, COPD, DVT...)
- Context-aware disambiguation (MS = multiple sclerosis vs mitral stenosis)
- Multiple formats: dotted (q.d.), slash (s/p), mixed-case (SpO2)
- False positive filtering (OR, US, IT stay unchanged)

Also works with `FaithfulnessEvaluator(llm, medical_mode=True)`.

> Contributed by [@harish1120](https://github.com/harish1120)

### System Metrics
- **Latency** — End-to-end response time
- **Token usage** — Cost tracking for LLM calls

### Cost Tracking (v1.3+)

Track exactly what your evaluations cost:

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

**Local-first advantage:** Ollama evaluations cost $0.00.

### Jupyter Notebook (v1.3.1+)

Rich HTML display for evaluation results:

```python
from ragnarok_ai.notebook import display, display_comparison

# Full dashboard with metrics, cost, latency
display(results)

# Compare multiple pipelines side-by-side
display_comparison([
    ("Baseline", baseline_results),
    ("Improved", improved_results),
])
```

---

## Roadmap

### Completed

<details>
<summary><strong>v0.1 — Foundation</strong></summary>

- [x] Project setup & architecture
- [x] Core retrieval metrics (precision, recall, MRR, NDCG)
- [x] Ollama adapter
- [x] Console reporter
- [x] JSON reporter
- [x] Basic CLI
- [x] CI/CD with GitHub Actions

</details>

<details>
<summary><strong>v0.2 — Generation Metrics & Reporting</strong></summary>

- [x] Qdrant adapter
- [x] Faithfulness evaluator
- [x] Relevance evaluator
- [x] Hallucination detection
- [x] HTML report with drill-down (failed questions, retrieved chunks)
- [x] Intelligent CI gating (stable metrics fail, LLM judgments warn)

</details>

<details>
<summary><strong>v0.3 — Test Generation & Golden Sets</strong></summary>

- [x] Synthetic question generation
- [x] Multi-hop question support
- [x] Adversarial question generation
- [x] Checkpointing system
- [x] Golden set support (human-validated, versioned question sets)
- [x] Baselines library (configs + expected results)
- [x] NovaTech example dataset for quickstart

</details>

<details>
<summary><strong>v0.4 — Framework Adapters & Observability</strong></summary>

- [x] LangChain integration
- [x] LangGraph integration
- [x] Custom RAG protocol support
- [x] OpenTelemetry export for tracing & debugging

</details>

<details>
<summary><strong>v0.5 — Performance & Scale</strong></summary>

- [x] Async parallelization (`max_concurrency` parameter)
- [x] Result caching (`MemoryCache`, `DiskCache`, `CacheProtocol`)
- [x] Batch processing (`BatchEvaluator` for 1000+ queries)
- [x] Progress callbacks (sync and async support)
- [x] Timeout and retry (`timeout`, `max_retries`, `retry_delay`)
- [x] Cache error handling (graceful degradation)

</details>

<details>
<summary><strong>v0.6 — Cloud & Local Adapters</strong></summary>

- [x] vLLM adapter (local high-performance inference)
- [x] OpenAI adapter (optional cloud fallback)
- [x] Anthropic adapter
- [x] ChromaDB adapter
- [x] FAISS adapter (pure local, no server)

</details>

<details>
<summary><strong>v0.7 — Framework Adapters</strong></summary>

- [x] LlamaIndex adapter (Retriever, QueryEngine, Index)
- [x] DSPy adapter (Retrieve, Module, RAG pattern)
- [x] Custom RAG support via `RAGProtocol`
- [x] Adapter contribution guide

</details>

<details>
<summary><strong>v0.8 — Comparison & Benchmarking</strong></summary>

- [x] Comparison mode (`compare()` for side-by-side evaluation)
- [x] Regression detection (alert on quality drop vs baseline)
- [x] Benchmark history tracking (time-series storage)
- [x] Diff reports (what changed between runs)

</details>

<details>
<summary><strong>v0.9 — Agent Evaluation</strong></summary>

- [x] `AgentProtocol` for agent pipelines
- [x] Tool-use correctness metrics (precision, recall, F1)
- [x] Multi-step reasoning evaluators (coherence, goal progress, efficiency)
- [x] ReAct/CoT pattern adapters
- [x] Trajectory analysis (loops, dead ends, failure detection)
- [x] Visualization (ASCII, Mermaid, HTML reports)

</details>

<details>
<summary><strong>v1.0 — Production Ready</strong></summary>

- [x] PyPI publish (`pip install ragnarok-ai`)
- [x] Stable public API
- [x] Complete README with examples
- [x] CHANGELOG.md (v0.1 → v1.0)

</details>

<details>
<summary><strong>v1.1 — CLI Complete</strong></summary>

- [x] `ragnarok generate` command (synthetic testset generation)
- [x] `ragnarok benchmark` command (history tracking, regression detection)
- [x] Standardized JSON envelope for `--json` output
- [x] E2E tests for CLI workflow
- [x] Trusted Publishing (PyPI OIDC)

</details>

<details>
<summary><strong>v1.2 — LLM-as-Judge</strong></summary>

- [x] `LLMJudge` class with Prometheus 2 integration
- [x] Multi-criteria evaluation (faithfulness, relevance, hallucination, completeness)
- [x] 100% local evaluation with Ollama (Q5_K_M quantization, ~5GB)
- [x] Rubric-based prompts with 1-5 scoring normalized to 0-1
- [x] Detailed explanations for each judgment
- [x] Batch evaluation support
- [x] Robust JSON parsing for LLM responses (handles incomplete JSON)
- [x] `keep_alive` support for Ollama (prevents model unloading between requests)

</details>

<details>
<summary><strong>v1.2.5 — Plugin Architecture</strong></summary>

- [x] Plugin system based on Python entry points
- [x] `PluginRegistry` singleton for adapter discovery
- [x] Dynamic discovery of external plugins via `importlib.metadata`
- [x] `ragnarok plugins` CLI command (list, info, filters)
- [x] Support for 4 namespaces: llm, vectorstore, framework, evaluator
- [x] LOCAL/CLOUD classification for all adapters
- [x] Plugin documentation (`docs/PLUGINS.md`)
- [x] E2E plugin test with mock package

</details>

<details>
<summary><strong>v1.3.0 — Cost Tracking</strong></summary>

- [x] Cost tracking module (`ragnarok_ai.cost`)
- [x] Pricing table for OpenAI, Anthropic, Groq, Mistral, Together AI
- [x] Token counting with tiktoken (fallback to estimation)
- [x] `CostTracker` class with context manager support
- [x] `track_cost=True` parameter in `evaluate()`
- [x] Formatted summary table and JSON export
- [x] Local providers (Ollama, vLLM) = $0.00
- [x] Automatic tracking in LLM adapters

</details>

<details>
<summary><strong>v1.3.1 — Jupyter Integration</strong></summary>

- [x] Jupyter notebook module (`ragnarok_ai.notebook`)
- [x] Rich HTML display for evaluation results
- [x] Metrics visualization with progress bars
- [x] Cost breakdown tables
- [x] Pipeline comparison display
- [x] Auto-detection of notebook environment

</details>

<details>
<summary><strong>v1.4.0 — Medical Mode</strong></summary>

- [x] Medical abbreviation normalizer (`medical_mode=True`)
- [x] 350+ abbreviations (CHF, MI, COPD, DVT...)
- [x] Context-aware disambiguation
- [x] Integration with `LLMJudge` and `FaithfulnessEvaluator`
- [x] Contributed by [@harish1120](https://github.com/harish1120)

</details>

### Planned

#### v1.5+ — Post-launch
- [ ] Comprehensive documentation site
- [ ] Performance benchmarks published
- [ ] Production monitoring mode
- [x] `ragnarok judge` CLI command
- [x] `--config ragnarok.yaml` support

### Future

<details>
<summary><strong>Web UI</strong></summary>

- [ ] Basic Web UI (read-only dashboard)
- [ ] Full Web UI dashboard

</details>

<details>
<summary><strong>Developer Experience</strong></summary>

- [x] GitHub Action ([`2501Pr0ject/ragnarok-evaluate-action`](https://github.com/2501Pr0ject/ragnarok-evaluate-action))
- [ ] VS Code extension
- [ ] Interactive CLI (TUI)
- [ ] Rust acceleration for hot paths

</details>

<details>
<summary><strong>More Integrations</strong></summary>

- [ ] Haystack adapter
- [ ] Semantic Kernel adapter
- [x] Groq adapter
- [x] Mistral API adapter
- [x] Together AI adapter
- [ ] pgvector adapter
- [ ] Weaviate adapter
- [ ] Pinecone adapter
- [ ] Milvus adapter

</details>

<details>
<summary><strong>Advanced Features</strong></summary>

- [ ] Streaming evaluation
- [ ] A/B testing support
- [ ] Dataset versioning
- [ ] Fine-tuning recommendations
- [ ] Multi-modal evaluation (images, audio)

</details>

<details>
<summary><strong>Enterprise (On-Premise)</strong></summary>

- [ ] SSO support (SAML, OIDC)
- [ ] Role-based access control
- [ ] Audit logging
- [ ] Air-gapped deployment guide
- [ ] Docker/Kubernetes helm charts

</details>

---

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/2501Pr0ject/RAGnarok-AI/main/assets/architecture.png" alt="RAGnarok-AI Architecture" width="1000">
</p>

<details>
<summary><strong>View project structure</strong></summary>

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

</details>

---

## Development

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

---

## Advanced Usage

### Importing Types

For advanced use cases (custom RAG implementations, type hints), import types directly from submodules:

```python
# Core types
from ragnarok_ai.core.types import Document, Query, RAGResponse, TestSet

# Protocols (for implementing custom adapters)
from ragnarok_ai.core.protocols import RAGProtocol, LLMProtocol, VectorStoreProtocol

# Evaluators
from ragnarok_ai.evaluators import FaithfulnessEvaluator, RelevanceEvaluator

# Adapters
from ragnarok_ai.adapters.llm import OllamaLLM, OpenAILLM
from ragnarok_ai.adapters.vectorstore import ChromaVectorStore, QdrantVectorStore
```

### Implementing a Custom RAG

```python
from ragnarok_ai.core.protocols import RAGProtocol
from ragnarok_ai.core.types import RAGResponse, Document

class MyCustomRAG:
    """Custom RAG implementing the RAGProtocol."""

    async def query(self, question: str, k: int = 5) -> RAGResponse:
        # Your retrieval logic here
        docs = await self.retrieve(question, k)
        answer = await self.generate(question, docs)

        return RAGResponse(
            answer=answer,
            retrieved_docs=[
                Document(id=d.id, content=d.text, metadata=d.meta)
                for d in docs
            ],
        )

# Use with ragnarok-ai
from ragnarok_ai import evaluate

results = await evaluate(
    rag_pipeline=MyCustomRAG(),
    testset=testset,
    metrics=["retrieval", "faithfulness"],
)
```

---

## Feedback

Your feedback helps improve RAGnarok-AI. Pick the right channel:

| Type | Link |
|------|------|
| Bug report | [Report a bug](https://github.com/2501Pr0ject/RAGnarok-AI/issues/new?template=bug_report.yml) |
| Feedback / UX | [Share feedback](https://github.com/2501Pr0ject/RAGnarok-AI/issues/new?template=feedback.yml) |
| Feature request | [Request a feature](https://github.com/2501Pr0ject/RAGnarok-AI/issues/new?template=feature_request.yml) |
| Questions / Ideas | [Discussions](https://github.com/2501Pr0ject/RAGnarok-AI/discussions) |

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Priority areas for contributions:**
- Framework adapters (Haystack, Semantic Kernel)
- Vector store adapters (pgvector, Weaviate, Pinecone)
- LLM adapters (Groq, Mistral API, Together AI)
- Agent evaluation features
- Documentation & examples

---

## License

RAGnarok-AI is dual-licensed:

| License | Use Case |
|---------|----------|
| [AGPL-3.0](LICENSE) | Open source projects, personal use, research |
| [Commercial](LICENSE-COMMERCIAL.md) | Proprietary software, SaaS, organizations with AGPL restrictions |

**Why dual licensing?**
- AGPL ensures improvements stay open-source
- Commercial license enables enterprise adoption without copyleft obligations

For commercial licensing inquiries: abdel.touati@gmail.com

---

## Acknowledgments

Built out of frustration with complex evaluation setups. We wanted something that just works — locally, fast, and without API keys.

---

<p align="center">
  <sub>Built with ❤️ in Lyon, France</sub>
</p>
