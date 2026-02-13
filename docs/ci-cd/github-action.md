# GitHub Action

Integrate RAGnarok-AI into your CI/CD pipeline.

---

## Overview

The `2501Pr0ject/ragnarok-evaluate-action` provides:

- Automatic RAG evaluation on pull requests
- PR comments with results
- Advisory mode (non-blocking by default)
- Distinction between deterministic and advisory metrics

---

## Quick Start

```yaml
# .github/workflows/rag-tests.yml
name: RAG Evaluation

on:
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: 2501Pr0ject/ragnarok-evaluate-action@v1
        with:
          config: ragnarok.yaml
          threshold: 0.8
```

---

## Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `config` | `ragnarok.yaml` | Path to configuration file |
| `threshold` | `0.8` | Minimum acceptable score (0.0-1.0) |
| `fail-on-threshold` | `false` | Fail CI if threshold not met |
| `comment-on-pr` | `true` | Post results as PR comment |

---

## Configuration File

Create `ragnarok.yaml` in your repository:

```yaml
# ragnarok.yaml
testset: ./tests/testset.json
output: ./results.json
fail_under: 0.8

metrics:
  - precision
  - recall
  - mrr
  - ndcg

criteria:
  - faithfulness
  - relevance
  - hallucination

ollama_url: http://localhost:11434
```

---

## Advisory Mode (Default)

By default, the action runs in advisory mode:

- **Does not fail** the CI pipeline
- Posts a PR comment with suggestions
- Uses humble language ("RAGnarok suggests reviewing...")

```yaml
- uses: 2501Pr0ject/ragnarok-evaluate-action@v1
  with:
    config: ragnarok.yaml
    threshold: 0.8
    fail-on-threshold: false  # Default
```

---

## Blocking Mode

To fail CI on threshold violation:

```yaml
- uses: 2501Pr0ject/ragnarok-evaluate-action@v1
  with:
    config: ragnarok.yaml
    threshold: 0.8
    fail-on-threshold: true
```

---

## PR Comment Format

The action posts a PR comment distinguishing:

**Deterministic Metrics** (hard facts):

```
+  precision@10: 0.82
+  recall@10: 0.75
+  mrr: 0.88
```

**Advisory Metrics** (LLM-as-Judge):

```
~  faithfulness: 0.74 (advisory)
~  relevance: 0.81 (advisory)
~  hallucination: 0.12 (advisory)
```

!!! note "Why the distinction?"
    Retrieval metrics are deterministic and reproducible.
    LLM-as-Judge scores are advisory and may vary between runs.

---

## Example Workflow

Full workflow with Ollama setup:

```yaml
name: RAG Evaluation

on:
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install ragnarok-ai[ollama]

      - name: Install Ollama
        run: |
          curl -fsSL https://ollama.ai/install.sh | sh
          ollama serve &
          sleep 5
          ollama pull mistral

      - name: Run RAGnarok
        uses: 2501Pr0ject/ragnarok-evaluate-action@v1
        with:
          config: ragnarok.yaml
          threshold: 0.8
```

---

## Without LLM-as-Judge

For faster CI without Ollama:

```yaml
# ragnarok.yaml
metrics:
  - precision
  - recall
  - mrr
  - ndcg
criteria: []  # Skip LLM-as-Judge
```

---

## Outputs

The action provides outputs for downstream steps:

```yaml
- uses: 2501Pr0ject/ragnarok-evaluate-action@v1
  id: ragnarok
  with:
    config: ragnarok.yaml

- name: Use results
  run: |
    echo "Average score: ${{ steps.ragnarok.outputs.average }}"
    echo "Status: ${{ steps.ragnarok.outputs.status }}"
```

---

## Marketplace

Find the action on GitHub Marketplace:

[RAGnarok Evaluate Action](https://github.com/marketplace/actions/ragnarok-evaluate-action)

---

## Next Steps

- [CLI Reference](cli-reference.md) — Command-line interface
- [Evaluation Guide](../user-guide/evaluation.md) — Understanding metrics
