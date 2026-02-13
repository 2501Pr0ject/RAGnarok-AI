# Installation

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) (for local LLM evaluation)

---

## Using pip

```bash
pip install ragnarok-ai
```

## Using uv (recommended)

```bash
uv pip install ragnarok-ai
```

---

## Optional Dependencies

RAGnarok-AI has minimal core dependencies. Install extras as needed:

### LLM Providers

```bash
pip install ragnarok-ai[ollama]      # Ollama (local)
pip install ragnarok-ai[openai]      # OpenAI
pip install ragnarok-ai[anthropic]   # Anthropic
pip install ragnarok-ai[vllm]        # vLLM (local high-performance)
pip install ragnarok-ai[groq]        # Groq
pip install ragnarok-ai[mistral]     # Mistral AI
pip install ragnarok-ai[together]    # Together AI
```

### Vector Stores

```bash
pip install ragnarok-ai[qdrant]      # Qdrant
pip install ragnarok-ai[chroma]      # ChromaDB
pip install ragnarok-ai[faiss]       # FAISS (pure local)
pip install ragnarok-ai[pinecone]    # Pinecone (cloud)
pip install ragnarok-ai[weaviate]    # Weaviate
pip install ragnarok-ai[milvus]      # Milvus
pip install ragnarok-ai[pgvector]    # PostgreSQL pgvector
```

### RAG Frameworks

```bash
pip install ragnarok-ai[langchain]   # LangChain/LangGraph
pip install ragnarok-ai[llamaindex]  # LlamaIndex
pip install ragnarok-ai[dspy]        # DSPy
pip install ragnarok-ai[haystack]    # Haystack
pip install ragnarok-ai[semantic-kernel]  # Semantic Kernel
```

### Observability

```bash
pip install ragnarok-ai[telemetry]   # OpenTelemetry tracing
```

### Everything

```bash
pip install ragnarok-ai[all]
```

---

## Ollama Setup

RAGnarok-AI uses Ollama for local LLM evaluation. Install and start Ollama:

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
```

### Install Prometheus 2 (LLM-as-Judge)

For LLM-as-Judge evaluation, install the Prometheus 2 model:

```bash
ollama pull hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M
```

Requirements:

- ~5GB disk space
- 16GB RAM recommended

---

## Development Setup

For contributing or development:

```bash
git clone https://github.com/2501Pr0ject/RAGnarok-AI.git
cd RAGnarok-AI
pip install -e ".[dev]"
pre-commit install
```

### Run Tests

```bash
pytest                    # Unit tests
pytest --cov=ragnarok_ai  # With coverage
ruff check . --fix        # Lint
ruff format .             # Format
mypy src/                 # Type check
```

---

## Verify Installation

```bash
ragnarok --version
# ragnarok-ai v1.4.0

ragnarok evaluate --demo
```

---

## Next Steps

- [Quick Start](quickstart.md) — Run your first evaluation
- [CLI Reference](../ci-cd/cli-reference.md) — Command-line options
