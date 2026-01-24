# Contributing to ragnarok-ai

Thank you for your interest in contributing to ragnarok-ai! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ragnarok-ai.git
   cd ragnarok-ai
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/2501Pr0ject/ragnarok-ai.git
   ```

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai/) (for integration tests)

### Installation

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate on Windows

# Install with development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Setup

```bash
# Run all checks
ruff check .                    # Linting
ruff format --check .           # Format check
mypy src/                       # Type checking
pytest --cov=ragnarok_ai        # Tests with coverage
```

---

## Making Changes

### Branching

Create a feature branch from `main`:

```bash
git checkout main
git pull upstream main
git checkout -b feat/your-feature-name
```

**Branch naming conventions:**
- `feat/xxx` — New features
- `fix/xxx` — Bug fixes
- `refactor/xxx` — Code refactoring
- `docs/xxx` — Documentation changes
- `test/xxx` — Test additions/changes

### Keep Your Fork Updated

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

---

## Code Style

We follow strict code style guidelines. Please read [CLAUDE.md](CLAUDE.md) for detailed conventions.

### Key Points

- **Language**: All code, comments, and documentation in English
- **Line length**: 120 characters max
- **Type hints**: Required on all functions and methods
- **Docstrings**: Google style, required for public functions/classes

### Formatting

```bash
# Auto-fix linting issues
ruff check . --fix

# Format code
ruff format .
```

### Type Checking

```bash
# Run mypy in strict mode
mypy src/
```

We use strict mypy settings. Avoid `Any` types unless absolutely necessary.

### Example Function

```python
async def evaluate_retrieval(
    query: str,
    retrieved_docs: list[Document],
    relevant_docs: list[Document],
    *,
    k: int = 10,
) -> RetrievalMetrics:
    """Evaluate retrieval quality for a single query.

    Args:
        query: The search query string.
        retrieved_docs: Documents returned by the retrieval system.
        relevant_docs: Ground truth relevant documents.
        k: Number of top results to consider. Defaults to 10.

    Returns:
        RetrievalMetrics containing precision@k, recall@k, MRR, and NDCG.

    Raises:
        ValueError: If retrieved_docs is empty.
    """
    ...
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ragnarok_ai

# Run specific test file
pytest tests/unit/test_retrieval.py

# Run specific test
pytest -k "test_precision_at_k"

# Skip integration tests
pytest -m "not integration"
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names: `test_precision_returns_zero_for_empty_retrieved`
- Use `pytest.mark.parametrize` for testing multiple inputs
- Use fixtures from `conftest.py` when possible

### Coverage Requirements

- Minimum **80%** overall coverage
- Critical paths (evaluators, core) should have **90%+** coverage

---

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/).

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change that neither fixes bug nor adds feature |
| `perf` | Performance improvement |
| `test` | Adding/updating tests |
| `chore` | Maintenance tasks |
| `ci` | CI/CD changes |

### Scopes

`core`, `evaluator`, `generator`, `metrics`, `cli`, `adapters`, `reporters`, `docs`, `deps`, `ci`

### Examples

```
feat(evaluator): add NDCG metric for retrieval evaluation
fix(ollama): handle connection timeout gracefully
docs(readme): add installation instructions for Windows
test(retrieval): add edge cases for empty query handling
```

---

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest `main`:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   ruff check .
   ruff format --check .
   mypy src/
   pytest --cov=ragnarok_ai --cov-fail-under=80
   ```

3. **Ensure all tests pass** and coverage is maintained

### Submitting

1. Push your branch to your fork:
   ```bash
   git push origin feat/your-feature-name
   ```

2. Open a Pull Request against `main`

3. Fill out the PR template with:
   - Description of changes
   - Related issue (if any)
   - Testing performed
   - Screenshots (if UI changes)

### Review Process

- PRs require at least one approval before merging
- CI must pass (lint, type check, tests)
- Address review comments promptly
- Keep PRs focused and reasonably sized

### After Merge

- Delete your feature branch
- Update your local main:
  ```bash
  git checkout main
  git pull upstream main
  ```

---

## Areas for Contribution

We welcome contributions in these areas:

### High Priority

- **Generation metrics**: Faithfulness, relevance, hallucination detection
- **Test generation**: Synthetic question generation from documents
- **Checkpointing**: Resume interrupted evaluations

### Framework Adapters

- LangChain integration
- LangGraph integration
- LlamaIndex adapter
- Haystack adapter

### Vector Store Adapters

- Qdrant adapter (in progress)
- ChromaDB adapter
- Weaviate adapter

### Reporters

- HTML report with visualizations
- Markdown reporter

### Documentation

- Usage examples
- API documentation
- Tutorials

### Performance

- Batch processing optimizations
- Rust acceleration for hot paths (future)

---

## Questions?

- Open an [issue](https://github.com/2501Pr0ject/ragnarok-ai/issues) for bugs or feature requests
- Start a [discussion](https://github.com/2501Pr0ject/ragnarok-ai/discussions) for questions

---

Thank you for contributing to ragnarok-ai!
