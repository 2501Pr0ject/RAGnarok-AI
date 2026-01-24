# ragnarok-ai

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

> A local-first RAG evaluation framework for LLM applications.

## Features

- **Evaluate** — Metrics for RAG systems (faithfulness, retrieval quality, hallucination detection)
- **Generate** — Synthetic test set generation from knowledge bases
- **Monitor** — Production monitoring (latency, quality drift)
- **Benchmark** — Compare RAG configurations
- **Report** — Dashboards, CI/CD integration

## Core Principles

1. **100% Local** — No external API calls required. Everything runs locally.
2. **Minimal Dependencies** — Lightweight core, heavy deps as optional extras.
3. **Performance First** — Optimized for speed.
4. **Simple > Clever** — No over-engineering.
5. **Async First** — Designed for async, with sync wrappers available.

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/2501Pr0ject/ragnarok-ai.git
cd ragnarok-ai

# Create virtual environment and install
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

uv pip install -e "."
```

### With optional dependencies

```bash
# Install with Ollama support
uv pip install -e ".[ollama]"

# Install with Qdrant support
uv pip install -e ".[qdrant]"

# Install with LangChain support
uv pip install -e ".[langchain]"

# Install all optional dependencies
uv pip install -e ".[all]"
```

### Development setup

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Quick Start

```python
from ragnarok_ai import __version__

print(f"ragnarok-ai v{__version__}")
```

## Development

### Commands

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=ragnarok_ai

# Lint code
ruff check .

# Format code
ruff format .

# Type check
mypy src/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Project Structure

```
ragnarok-ai/
├── src/ragnarok_ai/    # Source code
│   ├── core/           # Core types, protocols, exceptions
│   ├── evaluators/     # RAG evaluation metrics
│   ├── generators/     # Test set generation
│   ├── adapters/       # LLM, vector store, framework adapters
│   ├── reporters/      # Output formatters
│   └── cli/            # Command-line interface
├── tests/              # Test suite
├── examples/           # Usage examples
└── benchmarks/         # Performance benchmarks
```

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.10+ |
| Package Manager | uv |
| Linter/Formatter | ruff |
| Type Checker | mypy (strict) |
| Test Framework | pytest |
| LLM Provider | Ollama |
| Vector Store | Qdrant |

## License

This project is licensed under the [AGPL-3.0 License](LICENSE).

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

*Built with focus on local-first, privacy-preserving AI evaluation.*
