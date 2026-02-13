# Benchmarking

Track performance over time and detect regressions.

---

## Overview

RAGnarok-AI provides benchmark tracking:

- **History** — Store evaluation results over time
- **Baselines** — Set reference points for comparison
- **Regression Detection** — Alert on quality drops
- **Comparison** — Side-by-side analysis

---

## CLI Usage

### Run Demo

```bash
ragnarok benchmark --demo
```

Output:

```
  RAGnarok-AI Benchmark Demo
  ========================================

  Simulating 3 benchmark runs over time...

  Run 1 (Baseline)
    Precision: 0.72  Recall: 0.68
    MRR: 0.75       NDCG: 0.70
    Average: 0.71 -> Set as baseline

  Run 2 (Improved)
    Precision: 0.78  Recall: 0.74
    MRR: 0.80       NDCG: 0.76
    Average: 0.77

  Run 3 (Regression)
    Precision: 0.65  Recall: 0.60
    MRR: 0.68       NDCG: 0.62
    Average: 0.64

  ----------------------------------------
  Regression Detection (Run 3 vs Baseline)
  ----------------------------------------
  - REGRESSION DETECTED:
    - precision: 0.72 -> 0.65 (-9.7%)
    - recall: 0.68 -> 0.60 (-11.8%)
```

### List Configurations

```bash
ragnarok benchmark --list
```

### View History

```bash
ragnarok benchmark --history my-rag-config
```

---

## Python API

### Record Benchmark

```python
from ragnarok_ai.benchmarks import BenchmarkHistory
from ragnarok_ai.benchmarks.storage import JSONFileStore

store = JSONFileStore("./benchmarks.json")
history = BenchmarkHistory(store=store)

# Record evaluation result
record = await history.record(
    eval_result=result,
    config_name="my-rag-v1",
    testset=testset,
)

# Set as baseline
await history.set_baseline(record.id)
```

### Detect Regression

```python
from ragnarok_ai.regression import RegressionDetector, RegressionThresholds

detector = RegressionDetector(
    baseline=baseline_result,
    thresholds=RegressionThresholds(
        precision_drop=0.05,  # Alert if precision drops > 5%
        recall_drop=0.05,
    ),
)

regression = detector.detect(current_result)

if regression.has_regressions:
    for alert in regression.alerts:
        print(f"{alert.metric}: {alert.baseline_value:.2f} -> {alert.current_value:.2f}")
```

---

## Storage

Benchmark history is stored in JSON format:

```json
{
  "records": [
    {
      "id": "abc123",
      "timestamp": "2026-02-13T10:00:00Z",
      "config_name": "my-rag-v1",
      "is_baseline": true,
      "metrics": {
        "precision": 0.72,
        "recall": 0.68,
        "mrr": 0.75,
        "ndcg": 0.70
      }
    }
  ]
}
```

Default location: `.ragnarok/benchmarks.json`

Custom location:

```bash
ragnarok benchmark --demo --storage ./my-benchmarks.json
```

---

## Thresholds

Configure regression thresholds:

```python
thresholds = RegressionThresholds(
    precision_drop=0.05,    # 5% drop
    recall_drop=0.05,
    mrr_drop=0.10,          # 10% drop
    ndcg_drop=0.05,
)
```

---

## CI/CD Integration

Use `--fail-under` for quality gates:

```bash
ragnarok benchmark --demo --fail-under 0.7

# Exit code 0 if average >= 0.7
# Exit code 1 if average < 0.7
```

For GitHub Actions, use the RAGnarok Action with regression detection:

```yaml
- uses: 2501Pr0ject/ragnarok-evaluate-action@v1
  with:
    threshold: 0.8
    fail-on-threshold: false  # Advisory mode
```

---

## Best Practices

1. **Set a baseline** — Mark your initial evaluation as baseline
2. **Track over time** — Run benchmarks on every significant change
3. **Use thresholds** — Define acceptable regression limits
4. **Review trends** — Monitor history for gradual degradation

---

## Next Steps

- [GitHub Action](../ci-cd/github-action.md) — CI/CD integration
- [CLI Reference](../ci-cd/cli-reference.md) — Full command reference
