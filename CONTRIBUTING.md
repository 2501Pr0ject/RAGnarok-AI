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
- [Release Checklist](#release-checklist)
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

Create a feature branch from `develop`:

```bash
git checkout develop
git pull upstream develop
git checkout -b feat/your-feature-name
```

> **Note:** We use `develop` for active development. The `main` branch is reserved for stable releases.

**Branch naming conventions:**
- `feat/xxx` — New features
- `fix/xxx` — Bug fixes
- `refactor/xxx` — Code refactoring
- `docs/xxx` — Documentation changes
- `test/xxx` — Test additions/changes

### Keep Your Fork Updated

```bash
git fetch upstream
git checkout develop
git merge upstream/develop
```

---

## Code Style

The conventions are the ones enforced by the tooling configured in `pyproject.toml`:

- **ruff** (`[tool.ruff]`): line length 120, target Python 3.10, rule sets `E`, `W`, `F`, `I`, `B`, `C4`, `UP`, `ARG`, `SIM`, `TCH`, `PTH`, `ERA`, `RUF`; `E501` is left to the formatter. `ruff format` is the only formatter.
- **mypy** (`[tool.mypy]`): `strict = true`, every function fully typed, no untyped decorators, unused `type: ignore` comments are errors.
- **Docstrings**: Google style (`Args:`, `Returns:`, `Raises:`, `Example:` sections), required on public modules, classes and functions; private helpers get a one-line summary.
- **Dataclasses** for value objects are frozen and validated in `__post_init__`; adapters are plain classes implementing the protocols in `ragnarok_ai.core.protocols`.

CI runs `uv run ruff check .`, `uv run ruff format --check .` and `uv run mypy src/`; run the same commands before pushing.

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

1. **Update your branch** with the latest `develop`:
   ```bash
   git fetch upstream
   git rebase upstream/develop
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

2. Open a Pull Request against `develop`

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
- Update your local develop:
  ```bash
  git checkout develop
  git pull upstream develop
  ```

---

## Release Checklist

Releases are cut by the maintainer. Before tagging:

1. `pyproject.toml` and `src/ragnarok_ai/__init__.py` carry the new version; CHANGELOG `[Unreleased]` becomes `[X.Y.Z] - date`.
2. `uv lock --check`, `ruff check`, `ruff format --check`, `mypy src`, `scripts/check_conflict_markers.sh` and `pytest tests/unit` pass locally and CI is green on `main`.
3. The integration tests that need a local model are run by hand, since CI never installs them:
   - `[hhem]`: `uv run --extra hhem pytest tests/unit/judges/test_hhem.py -m integration -v` (about 1 GB in the Hugging Face cache on first run; checks the pinned revisions, the model card examples and the recorded official probabilities within 1e-5);
   - Ollama judges: `uv run pytest -m integration tests/unit/evaluators/test_judge.py` with the default judge model pulled.
4. The wheel is built and imported in a fresh virtual environment (`uv build`, then `pip install dist/*.whl` and `python -c "import ragnarok_ai"`).
5. Tag `vX.Y.Z` on the release commit; the tag push publishes to PyPI. Write the GitHub release notes from the CHANGELOG entry.

## Areas for Contribution

We welcome contributions in these areas:

### Adapters

See **[CONTRIBUTING-ADAPTERS.md](CONTRIBUTING-ADAPTERS.md)** for a detailed guide on creating adapters.

**Framework Adapters** (implement `RAGProtocol`):
- Haystack adapter
- Semantic Kernel adapter

**LLM Adapters** (implement `LLMProtocol`):
- Groq adapter
- Mistral API adapter
- Together AI adapter

**Vector Store Adapters** (implement `VectorStoreProtocol`):
- pgvector adapter
- Weaviate adapter
- Pinecone adapter

### Agent Evaluation (v0.9)

- AgentProtocol design
- Tool-use correctness metrics
- Trajectory analysis

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
