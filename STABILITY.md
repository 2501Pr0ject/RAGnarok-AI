# Stability Policy

RAGnarok-AI follows [Semantic Versioning 2.0.0](https://semver.org/).

## Version Format

```
MAJOR.MINOR.PATCH
```

- **PATCH** (1.0.x): Bug fixes, documentation updates. No breaking changes.
- **MINOR** (1.x.0): New features, backward-compatible. No breaking changes.
- **MAJOR** (x.0.0): Breaking changes. Migration guide provided.

## What We Guarantee (Stable API)

The following are considered **stable** and follow SemVer strictly:

### CLI Interface
- Command names: `evaluate`, `generate`, `benchmark`
- Core options: `--demo`, `--output`, `--fail-under`, `--json`
- Exit codes: `0` = success, `1` = quality threshold failed, `2` = error

### Python API (Public)
```python
from ragnarok_ai import evaluate, compare, __version__
from ragnarok_ai.core.types import Document, Query, TestSet, RAGResponse
from ragnarok_ai.core.protocols import RAGProtocol, LLMProtocol, VectorStoreProtocol
from ragnarok_ai.evaluators import precision_at_k, recall_at_k, mrr, ndcg_at_k
```

### Output Formats
- JSON output structure from `--json` flag
- JSON export from `EvaluationResult`

## What May Change (Experimental)

The following may change in MINOR versions:

- Internal implementation details (`_private` functions)
- Adapter-specific options (e.g., `OllamaLLM` constructor parameters)
- HTML report layout and styling
- CLI output formatting (colors, spacing)
- New optional fields in JSON output

## Deprecation Policy

1. Deprecated features are marked with warnings for at least **one MINOR release**
2. Deprecation warnings include migration instructions
3. Deprecated features are removed in the next **MAJOR release**

## Current Status

| Version | Status | Support |
|---------|--------|---------|
| 1.x | **Current** | Active development |
| 0.x | Legacy | No longer supported |

## Reporting Issues

- **Bugs**: [GitHub Issues](https://github.com/2501Pr0ject/RAGnarok-AI/issues)
- **Security**: abdel.touati@gmail.com (private disclosure)

## Compatibility Matrix

### Python Versions
| Python | Status |
|--------|--------|
| 3.10 | Supported |
| 3.11 | Supported |
| 3.12 | Supported |
| 3.13 | Supported |
| < 3.10 | Not supported |

### LLM Providers
| Provider | Extra | Status |
|----------|-------|--------|
| Ollama | `[ollama]` | Supported |
| OpenAI | `[openai]` | Supported |
| Anthropic | `[anthropic]` | Supported |
| vLLM | `[vllm]` | Supported |

### Vector Stores
| Store | Extra | Status |
|-------|-------|--------|
| Qdrant | `[qdrant]` | Supported |
| ChromaDB | `[chroma]` | Supported |
| FAISS | `[faiss]` | Supported |

### RAG Frameworks
| Framework | Extra | Status |
|-----------|-------|--------|
| LangChain | `[langchain]` | Supported |
| LlamaIndex | `[llamaindex]` | Supported |
| DSPy | `[dspy]` | Supported |
| Custom | - | Supported via `RAGProtocol` |
