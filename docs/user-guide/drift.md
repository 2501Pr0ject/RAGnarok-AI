# Drift Detection

Production RAG pipelines degrade silently: users start asking different questions, an index rebuild changes retrieval behavior, a model update shifts latency. Drift detection compares current production traffic against a recorded baseline and alerts you when behavior shifts — before quality metrics do.

It builds on [Production Monitoring](monitoring.md): traces collected by the monitor daemon are the input.

## Quick start

**1. Record a baseline from a known-good period:**

```python
from datetime import datetime, timedelta, timezone

from ragnarok_ai import build_baseline
from ragnarok_ai.monitor.store import MonitorStore

store = MonitorStore()  # same DB as the monitor daemon

now = datetime.now(timezone.utc)
traces = store.get_traces(since=now - timedelta(days=7), until=now)
baseline = build_baseline(traces)
baseline.save("drift-baseline.json")
```

**2. Periodically compare recent traffic against it:**

```python
from ragnarok_ai import DriftBaseline, DriftDetector

detector = DriftDetector(DriftBaseline.load("drift-baseline.json"))

report = detector.detect(store.get_traces(since=now - timedelta(hours=1)))
if report.has_drift:
    for finding in report.findings:
        print(f"[{finding.severity.value}] {finding.message}")
```

**3. Or wire it straight into alerting:**

```python
from ragnarok_ai import AlertManager
from ragnarok_ai.alerts.adapters import SlackAlertAdapter

manager = AlertManager()
manager.add_adapter(SlackAlertAdapter(webhook_url="https://hooks.slack.com/..."))

report = await detector.check_and_alert(
    store.get_traces(since=now - timedelta(hours=1)),
    manager,
)
```

Alerts are sent with `source="drift"`, one per finding, with the metric, baseline value, current value, and score in the metadata.

## What is checked

**Distribution drift** — the Population Stability Index (PSI) is computed for each numeric field the baseline tracks:

| Field | Drift signal example |
|---|---|
| `total_latency_ms` | Infrastructure or model change |
| `query_length` | Users asking different kinds of questions |
| `answer_length` | Generation behavior shift |
| `retrieval_count` | Index or retriever configuration change |

PSI thresholds follow the standard convention: below 0.1 stable, 0.1–0.25 moderate shift (**warning**), above 0.25 major shift (**critical**).

**Metric drift** — scalar health checks:

- **Success rate**: absolute drop in percentage points (default: 5 points warns)
- **p95 latency**: relative increase (default: +25% warns)

Changes beyond `threshold x critical_multiplier` (default 2.0) are critical.

## Tuning

```python
from ragnarok_ai import DriftThresholds

detector = DriftDetector(
    baseline,
    DriftThresholds(
        psi_warning=0.1,
        psi_critical=0.25,
        success_rate_drop=0.05,   # 5 percentage points
        latency_increase=0.25,    # +25% on p95
        critical_multiplier=2.0,
        min_samples=100,
    ),
)
```

Windows smaller than `min_samples` are skipped and flagged `report.insufficient_data` rather than producing noisy findings.

## Design notes

- **The baseline is self-contained and PII-free.** It stores bucket edges and proportions, never raw traces or query text, so `drift-baseline.json` can be committed to a repo or shipped with a deployment.
- **Refresh the baseline deliberately.** After an intentional change (new model, new index), record a new baseline from the first known-good window — otherwise every check will keep flagging the improvement as drift.
- **Pair with regression testing.** Drift detection tells you *production behavior changed*; [evaluation](evaluation.md) tells you *whether quality is still acceptable*. A drift alert is a good trigger for an evaluation run.
