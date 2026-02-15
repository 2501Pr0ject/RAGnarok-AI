# CLI Reference

Complete reference for the `ragnarok` command-line interface.

---

## Global Options

```bash
ragnarok [OPTIONS] COMMAND [ARGS]
```

| Option | Description |
|--------|-------------|
| `--version`, `-v` | Show version and exit |
| `--json` | Output in JSON format |
| `--no-color` | Disable colored output |
| `--pii-mode` | PII handling: `hash` (default), `redact`, `full` |
| `--help` | Show help message |

---

## Commands

### version

Show the current version.

```bash
ragnarok version
# ragnarok-ai v1.4.1
```

---

### evaluate

Evaluate a RAG pipeline against a test set.

```bash
ragnarok evaluate [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--demo` | Run demo with NovaTech dataset |
| `--config`, `-c` | Path to ragnarok.yaml config file |
| `--testset`, `-t` | Path to testset JSON file |
| `--output`, `-o` | Output file path for results |
| `--fail-under` | Fail if average score below threshold (0.0-1.0) |
| `--limit`, `-n` | Limit number of queries |
| `--seed` | Random seed for reproducibility |

**Examples:**

```bash
# Demo evaluation
ragnarok evaluate --demo

# With config file
ragnarok evaluate --config ragnarok.yaml

# With threshold
ragnarok evaluate --demo --fail-under 0.7

# Limited queries
ragnarok evaluate --demo --limit 5

# Save results
ragnarok evaluate --demo --output results.json

# JSON output for CI
ragnarok evaluate --demo --json
```

---

### generate

Generate a synthetic test set from documents.

```bash
ragnarok generate [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--demo` | Use NovaTech example dataset |
| `--docs`, `-d` | Path to documents (JSON or directory) |
| `--num`, `-n` | Number of questions to generate (default: 10) |
| `--output`, `-o` | Output file path (default: testset.json) |
| `--model`, `-m` | Ollama model (default: mistral) |
| `--seed`, `-s` | Random seed for reproducibility |
| `--validate` | Validate generated questions |
| `--dry-run` | Show what would be generated |
| `--ollama-url` | Ollama API URL |

**Examples:**

```bash
# From demo dataset
ragnarok generate --demo --num 10

# From documents directory
ragnarok generate --docs ./knowledge/ --num 50

# From JSON file
ragnarok generate --docs documents.json --model llama3

# Dry run
ragnarok generate --demo --dry-run
```

---

### benchmark

Track benchmark history and detect regressions.

```bash
ragnarok benchmark [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--demo` | Run demo with simulated runs |
| `--list`, `-l` | List all recorded configurations |
| `--history`, `-H` | Show history for a config name |
| `--output`, `-o` | Output file for results |
| `--fail-under` | Fail if average below threshold |
| `--dry-run` | Show what would be benchmarked |
| `--storage`, `-s` | Path to storage file |

**Examples:**

```bash
# Run demo
ragnarok benchmark --demo

# List configurations
ragnarok benchmark --list

# View history
ragnarok benchmark --history my-rag-config

# With threshold
ragnarok benchmark --demo --fail-under 0.7
```

---

### judge

Evaluate responses using LLM-as-Judge.

```bash
ragnarok judge [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--context`, `-c` | Context text for evaluation |
| `--question`, `-q` | Question to evaluate |
| `--answer`, `-a` | Answer to evaluate |
| `--file`, `-f` | JSON file with items to evaluate |
| `--criteria` | Comma-separated criteria (default: all) |
| `--model`, `-m` | Ollama model (default: Prometheus 2) |
| `--fail-under` | Fail if average below threshold |
| `--output`, `-o` | Output file for results |
| `--ollama-url` | Ollama API URL |

**Criteria:**

- `faithfulness` — Is the answer grounded in context?
- `relevance` — Does the answer address the question?
- `hallucination` — Does the answer contain fabricated info?
- `completeness` — Are all aspects covered?
- `all` — All criteria (default)

**Examples:**

```bash
# Single evaluation
ragnarok judge \
  --context "Paris is the capital of France." \
  --question "What is the capital of France?" \
  --answer "Paris"

# From file
ragnarok judge --file items.json

# Select criteria
ragnarok judge --file items.json --criteria faithfulness,relevance

# With threshold
ragnarok judge --file items.json --fail-under 0.7

# JSON output
ragnarok judge --file items.json --json
```

---

### dataset

Manage and compare dataset versions.

```bash
ragnarok dataset COMMAND [OPTIONS]
```

#### dataset diff

Compare two dataset versions to detect changes.

```bash
ragnarok dataset diff <v1_path> <v2_path> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--key`, `-k` | Field to use as item key (default: metadata.id or content hash) |
| `--ignore-metadata` | Ignore metadata changes in comparison |
| `--show`, `-n` | Limit number of items shown (default: 10) |
| `--output`, `-o` | Export diff report to JSON file |
| `--fail-on-change` | Exit with error if changes detected (for CI) |

**Examples:**

```bash
# Compare two testset versions
ragnarok dataset diff testset_v1.json testset_v2.json

# Ignore metadata changes
ragnarok dataset diff v1.json v2.json --ignore-metadata

# Export diff report
ragnarok dataset diff v1.json v2.json --output diff_report.json

# CI gating: fail if dataset changed
ragnarok dataset diff baseline.json current.json --fail-on-change
```

**Output:**

```
  RAGnarok-AI Dataset Diff
  ========================================

  v1: testset_v1.json
      hash=a1b2c3d4e5f6g7h8  items=50
  v2: testset_v2.json
      hash=x9y8z7w6v5u4t3s2  items=52

  ----------------------------------------
  Summary
  ----------------------------------------
    Added:     2
    Removed:   0
    Modified:  3
    Unchanged: 47
```

#### dataset info

Show dataset metadata and statistics.

```bash
ragnarok dataset info <path>
```

**Example:**

```bash
ragnarok dataset info testset.json
```

---

### plugins

Manage and list available plugins.

```bash
ragnarok plugins [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--list`, `-l` | List all available plugins |
| `--type`, `-t` | Filter by type: llm, vectorstore, framework, evaluator |
| `--local` | Only show local adapters |
| `--info`, `-i` | Show info for a specific plugin |

**Examples:**

```bash
# List all plugins
ragnarok plugins --list

# Filter by type
ragnarok plugins --list --type llm

# Local only
ragnarok plugins --list --local

# Plugin info
ragnarok plugins --info ollama
```

---

## JSON Output

All commands support `--json` for machine-readable output:

```bash
ragnarok evaluate --demo --json
```

Response envelope:

```json
{
  "command": "evaluate",
  "status": "pass",
  "version": "1.4.1",
  "data": { ... },
  "warnings": [],
  "errors": []
}
```

Status values:

- `pass` — Evaluation passed threshold
- `fail` — Evaluation failed threshold
- `success` — Command completed successfully
- `error` — Command failed
- `dry_run` — Dry run completed

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime failure, threshold not met |
| 2 | Invalid arguments, missing files |

---

## Configuration File

Create `ragnarok.yaml`:

```yaml
testset: ./testset.json
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
  - completeness

ollama_url: http://localhost:11434
```

Use with:

```bash
ragnarok evaluate --config ragnarok.yaml
```

CLI options override config file values.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_HOST` | Ollama API URL |
| `NO_COLOR` | Disable colored output |

---

## Next Steps

- [GitHub Action](github-action.md) — CI/CD integration
- [Quick Start](../getting-started/quickstart.md) — Getting started
