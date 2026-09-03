# Production Testsets

Synthetic test sets evaluate the questions you *imagined*. Production mining evaluates the questions users *actually ask* — the frequent ones, the failing ones, the slow ones.

Builds on [Production Monitoring](monitoring.md), and closes the loop with [Drift Detection](drift.md): drift detected → mine the drifted window → evaluate → know whether quality really moved.

## Step 1 — Opt into query capture

By default, the monitor stores only a **hash** of each query (PII safety). Mining needs the text, so capture is explicit:

```python
from ragnarok_ai import MonitorClient

client = MonitorClient(capture_queries=True)
```

Before a query leaves the client, it is **scrubbed of inline PII**: email addresses, IP addresses, SSN-like and card-like numbers, and home paths are replaced with `[REDACTED]` (or a short hash with `pii_mode=PiiMode.HASH`), keeping the rest of the sentence intact:

```
"email me at bob@corp.com about CHF"  →  "email me at [REDACTED] about CHF"
```

Existing monitor databases are migrated automatically (a nullable `query_text` column); traces recorded before capture was enabled simply have no text and are skipped by the miner.

## Step 2 — Mine a test set

```python
from datetime import datetime, timedelta, timezone

from ragnarok_ai import TestsetMiner
from ragnarok_ai.monitor.store import MonitorStore

miner = TestsetMiner(MonitorStore())

testset = miner.mine(
    strategy="failures",                                     # or "frequent", "slow"
    since=datetime.now(timezone.utc) - timedelta(days=7),
    limit=50,
    min_frequency=2,
)
```

| Strategy | Selects | Use it to |
|---|---|---|
| `frequent` | Most-asked queries first | Evaluate what matters most to users |
| `failures` | Highest failure rate first (only queries that failed) | Reproduce and fix real errors |
| `slow` | Highest average latency first | Chase performance regressions |

Duplicate queries are collapsed by hash (the most recent wording wins), and each mined `Query` carries its production statistics in metadata:

```python
testset.queries[0].metadata
# {"source": "production", "frequency": 37, "failure_rate": 0.19,
#  "avg_latency_ms": 842.1, "first_seen": "...", "last_seen": "..."}
```

## Step 3 — Evaluate

```python
from ragnarok_ai import evaluate
from ragnarok_ai.dataset.io import save_testset

save_testset(testset, "production-failures.json")   # version it like any test set

results = await evaluate(rag_pipeline, testset, metrics=["faithfulness", "relevance"])
```

!!! note "No ground truth"
    Mined queries represent real traffic, not annotated data: they have no `ground_truth_docs` or `expected_answer`. LLM-judged metrics (faithfulness, relevance, hallucination) work directly; retrieval metrics and retrieval-side [diagnosis](diagnosis.md) need ground truth added separately — mined test sets are a starting point for curation, not a replacement for it.

## Privacy notes

- Capture is **opt-in per client**; the default behavior (hash only) is unchanged.
- Scrubbing happens **client-side** — raw text never reaches the daemon or the store.
- The inline patterns are conservative (emails, IPs, SSN/card-like numbers, home paths). If your queries may contain domain-specific identifiers (patient names, account codes), review mined test sets before committing them, or keep capture disabled and build test sets by hand.
- Trace retention applies: mined text follows the store's existing purge policy (7 days by default).
