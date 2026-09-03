# A/B Testing

Compare two RAG configurations on live traffic: split users deterministically between variants, tag monitor traces with the assignment, and let the analyzer tell you which variant wins — with statistical significance, not gut feeling.

Builds on [Production Monitoring](monitoring.md): traces collected by the monitor daemon are the input.

## Quick start

**1. Declare the experiment and route traffic:**

```python
from ragnarok_ai import Experiment, MonitorClient

exp = Experiment(name="reranker-test", variants=["control", "reranker"])
client = MonitorClient()

pipelines = {"control": rag_baseline, "reranker": rag_with_reranker}

async def handle_query(user_id: str, query: str) -> str:
    with client.trace(query) as trace:
        variant = exp.tag(trace, user_id)   # assigns + tags the trace
        return await pipelines[variant].query(query)
```

Assignment is **deterministic and stateless**: `assign(key)` hashes the key with the experiment name, so the same user always sees the same variant — across processes and restarts, with no shared storage. Use whatever key should stay consistent: a user id, session id, or query hash.

**2. Analyze once traffic has accumulated:**

```python
from ragnarok_ai import ABAnalyzer
from ragnarok_ai.monitor.store import MonitorStore

store = MonitorStore()
analyzer = ABAnalyzer(exp)

report = analyzer.analyze(store.get_traces(since=experiment_start))

print(report.winner)  # "reranker", "control", or None
for verdict in report.verdicts:
    print(f"{verdict.metric}: {verdict.a_value:.3f} vs {verdict.b_value:.3f} "
          f"(p={verdict.p_value:.4f}, winner={verdict.winner})")
```

## What is compared

| Metric | Test | Better |
|---|---|---|
| Success rate | Two-proportion z-test | Higher |
| Mean total latency | Welch's t-test | Lower |

Each variant also gets descriptive stats (count, p50/p95 latency) in `report.stats`.

**Overall winner rule**: a variant wins the experiment only if it is significantly better on at least one metric and significantly worse on none. A variant that is faster but fails more often produces `winner=None` — the trade-off is yours to arbitrate, and both verdicts are in the report.

## Options

```python
from ragnarok_ai import ABTestConfig

# Uneven split: send 10% of traffic to the candidate
exp = Experiment(name="test", variants=["control", "candidate"], weights=[0.9, 0.1])

# Stricter significance, larger minimum sample
analyzer = ABAnalyzer(exp, ABTestConfig(alpha=0.01, min_samples=200))
```

- Windows where either variant has fewer than `min_samples` traces are flagged `report.insufficient_data` instead of producing unstable verdicts.
- Changing `salt` reshuffles all assignments (useful to re-run an experiment with fresh buckets).
- More than two variants are supported at assignment time; `analyze(traces, variant_a=..., variant_b=...)` compares any pair.

## Design notes

- **p-values use the normal approximation**, which is accurate at the sample sizes `min_samples` enforces — the module needs no scipy.
- **Peeking caveat**: checking significance repeatedly as data accumulates inflates false positives. Decide the analysis window in advance, or apply a stricter `alpha`.
- **Pair with drift detection**: [drift detection](drift.md) watches the whole traffic; A/B testing compares deliberate configuration changes within it.
