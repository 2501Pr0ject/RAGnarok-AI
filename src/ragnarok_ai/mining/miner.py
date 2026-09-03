"""Mine production traffic into evaluation test sets.

Synthetic test sets ask the questions you *imagined*; production mining
builds test sets from the questions users *actually ask* — the frequent
ones, the ones that fail, the slow ones. Combined with drift detection,
it closes the loop: drift detected → mine the window → evaluate → know
whether quality really moved.

Requires query capture to be enabled on the client
(``MonitorClient(capture_queries=True)``): queries are scrubbed of inline
PII before leaving the client, and only those traces carry text to mine.

Example:
    >>> from ragnarok_ai.mining import TestsetMiner
    >>> from ragnarok_ai.monitor.store import MonitorStore
    >>>
    >>> from ragnarok_ai.dataset.io import save_testset
    >>>
    >>> miner = TestsetMiner(MonitorStore())
    >>> testset = miner.mine(strategy="failures", since=last_week, limit=50)
    >>> save_testset(testset, "failing-queries.json")
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import TYPE_CHECKING, Literal

from ragnarok_ai.core.types import Query, TestSet

if TYPE_CHECKING:
    from ragnarok_ai.monitor.models import TraceEvent
    from ragnarok_ai.monitor.store import MonitorStore

Strategy = Literal["frequent", "failures", "slow"]


@dataclass
class _QueryGroup:
    """Traces sharing one query hash."""

    text: str
    traces: list[TraceEvent]

    @property
    def frequency(self) -> int:
        return len(self.traces)

    @property
    def failure_rate(self) -> float:
        return sum(1 for t in self.traces if not t.success) / len(self.traces)

    @property
    def avg_latency_ms(self) -> float:
        return mean(t.total_latency_ms for t in self.traces)

    @property
    def first_seen(self) -> datetime:
        return min(t.timestamp for t in self.traces)

    @property
    def last_seen(self) -> datetime:
        return max(t.timestamp for t in self.traces)


class TestsetMiner:
    """Build evaluation test sets from production monitor traces.

    Traces are deduplicated by query hash; each unique query becomes one
    ``Query`` carrying production statistics in its metadata (frequency,
    failure rate, average latency, first/last seen).

    Mined queries have no ``ground_truth_docs`` or ``expected_answer`` —
    they represent real traffic, not annotated data. LLM-judged metrics
    (faithfulness, relevance, hallucination) work directly; retrieval
    metrics need ground truth added separately.

    Attributes:
        store: The monitor store holding production traces.

    Example:
        >>> miner = TestsetMiner(store)
        >>> testset = miner.mine(strategy="frequent", limit=100)
    """

    __test__ = False  # Prevent pytest collection warning

    def __init__(self, store: MonitorStore) -> None:
        """Initialize the miner.

        Args:
            store: Monitor store to read traces from.
        """
        self.store = store

    def mine(
        self,
        *,
        strategy: Strategy = "frequent",
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        min_frequency: int = 1,
        name: str | None = None,
    ) -> TestSet:
        """Mine a test set from production traces.

        Args:
            strategy: Which queries to prioritize —
                ``"frequent"``: most-asked queries first;
                ``"failures"``: queries with the highest failure rate
                (only those that failed at least once);
                ``"slow"``: highest average latency first.
            since: Only traces at or after this time.
            until: Only traces before this time.
            limit: Maximum number of queries in the test set.
            min_frequency: Only queries asked at least this many times.
            name: Test set name (default derived from the strategy).

        Returns:
            A ``TestSet`` with ``source="production"`` and per-query
            production statistics in metadata.

        Raises:
            ValueError: If no trace in the window carries query text —
                enable ``MonitorClient(capture_queries=True)`` to record
                (PII-scrubbed) queries.
        """
        traces = self.store.get_traces(since=since, until=until)
        groups = self._group(traces)

        if not groups:
            if traces:
                msg = (
                    "No trace in this window carries query text. Enable "
                    "MonitorClient(capture_queries=True) to record PII-scrubbed "
                    "queries for mining."
                )
            else:
                msg = "No traces found in this window."
            raise ValueError(msg)

        selected = self._select(groups, strategy, limit, min_frequency)

        queries = [
            Query(
                text=group.text,
                metadata={
                    "source": "production",
                    "query_hash": query_hash,
                    "frequency": group.frequency,
                    "failure_rate": round(group.failure_rate, 4),
                    "avg_latency_ms": round(group.avg_latency_ms, 1),
                    "first_seen": group.first_seen.isoformat(),
                    "last_seen": group.last_seen.isoformat(),
                },
            )
            for query_hash, group in selected
        ]

        return TestSet(
            name=name or f"production-{strategy}",
            queries=queries,
            source="production",
            description=f"Mined from production traces (strategy={strategy})",
            metadata={
                "strategy": strategy,
                "mined_at": datetime.now(timezone.utc).isoformat(),
                "window_start": since.isoformat() if since else None,
                "window_end": until.isoformat() if until else None,
                "unique_queries_in_window": len(groups),
            },
        )

    # ── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _group(traces: list[TraceEvent]) -> dict[str, _QueryGroup]:
        """Group text-carrying traces by query hash (latest text wins)."""
        groups: dict[str, _QueryGroup] = {}
        for trace in traces:  # get_traces returns oldest first
            if not trace.query_text:
                continue
            group = groups.get(trace.query_hash)
            if group is None:
                groups[trace.query_hash] = _QueryGroup(text=trace.query_text, traces=[trace])
            else:
                group.traces.append(trace)
                group.text = trace.query_text  # keep the most recent wording
        return groups

    @staticmethod
    def _select(
        groups: dict[str, _QueryGroup],
        strategy: Strategy,
        limit: int,
        min_frequency: int,
    ) -> list[tuple[str, _QueryGroup]]:
        """Order groups per strategy and apply filters."""
        items = [(h, g) for h, g in groups.items() if g.frequency >= min_frequency]

        if strategy == "frequent":
            items.sort(key=lambda kv: (-kv[1].frequency, kv[1].text))
        elif strategy == "failures":
            items = [kv for kv in items if kv[1].failure_rate > 0]
            items.sort(key=lambda kv: (-kv[1].failure_rate, -kv[1].frequency, kv[1].text))
        elif strategy == "slow":
            items.sort(key=lambda kv: (-kv[1].avg_latency_ms, kv[1].text))
        else:  # pragma: no cover - Literal prevents this at type-check time
            msg = f"Unknown mining strategy: {strategy!r}"
            raise ValueError(msg)

        return items[:limit]
