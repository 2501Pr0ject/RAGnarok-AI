"""Tests for production testset mining."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ragnarok_ai.mining.miner import TestsetMiner
from ragnarok_ai.monitor.models import TraceEvent
from ragnarok_ai.monitor.store import MonitorStore

START = datetime(2026, 9, 1, tzinfo=timezone.utc)


def make_trace(
    text: str | None,
    *,
    query_hash: str | None = None,
    success: bool = True,
    latency: float = 200.0,
    at: datetime = START,
) -> TraceEvent:
    """Build a trace; hash defaults to the text so equal texts group."""
    return TraceEvent(
        query_hash=query_hash or f"h-{text}",
        query_length=len(text) if text else 0,
        query_text=text,
        total_latency_ms=latency,
        success=success,
        timestamp=at,
    )


@pytest.fixture
def store() -> MonitorStore:
    return MonitorStore(db_path=":memory:")


class TestMining:
    """Test suite for TestsetMiner.mine."""

    def test_frequent_strategy_orders_by_frequency(self, store: MonitorStore) -> None:
        for _ in range(3):
            store.insert(make_trace("What is CHF?"))
        store.insert(make_trace("What is MI?"))

        testset = TestsetMiner(store).mine(strategy="frequent")

        assert testset.source == "production"
        assert [q.text for q in testset.queries] == ["What is CHF?", "What is MI?"]
        assert testset.queries[0].metadata["frequency"] == 3
        assert testset.queries[0].metadata["source"] == "production"

    def test_duplicates_collapse_to_latest_wording(self, store: MonitorStore) -> None:
        store.insert(make_trace("what is chf", query_hash="same", at=START))
        store.insert(make_trace("What is CHF?", query_hash="same", at=START + timedelta(hours=1)))

        testset = TestsetMiner(store).mine()

        assert len(testset.queries) == 1
        assert testset.queries[0].text == "What is CHF?"
        assert testset.queries[0].metadata["frequency"] == 2

    def test_failures_strategy_keeps_only_failing_queries(self, store: MonitorStore) -> None:
        store.insert(make_trace("always works"))
        store.insert(make_trace("sometimes fails", query_hash="sf", success=True))
        store.insert(make_trace("sometimes fails", query_hash="sf", success=False))
        store.insert(make_trace("always fails", success=False))

        testset = TestsetMiner(store).mine(strategy="failures")

        texts = [q.text for q in testset.queries]
        assert texts == ["always fails", "sometimes fails"]  # by failure rate
        assert testset.queries[0].metadata["failure_rate"] == 1.0
        assert testset.queries[1].metadata["failure_rate"] == 0.5

    def test_slow_strategy_orders_by_latency(self, store: MonitorStore) -> None:
        store.insert(make_trace("fast", latency=100))
        store.insert(make_trace("slow", latency=2000))

        testset = TestsetMiner(store).mine(strategy="slow")

        assert [q.text for q in testset.queries] == ["slow", "fast"]
        assert testset.queries[0].metadata["avg_latency_ms"] == 2000.0

    def test_limit_and_min_frequency(self, store: MonitorStore) -> None:
        for i in range(5):
            store.insert(make_trace(f"once-{i}"))
        for _ in range(3):
            store.insert(make_trace("popular"))

        testset = TestsetMiner(store).mine(min_frequency=2, limit=10)
        assert [q.text for q in testset.queries] == ["popular"]

        capped = TestsetMiner(store).mine(limit=2)
        assert len(capped.queries) == 2

    def test_window_filter(self, store: MonitorStore) -> None:
        store.insert(make_trace("old", at=START - timedelta(days=10)))
        store.insert(make_trace("recent", at=START))

        testset = TestsetMiner(store).mine(since=START - timedelta(days=1))

        assert [q.text for q in testset.queries] == ["recent"]
        assert testset.metadata["window_start"] is not None

    def test_no_captured_text_raises_helpful_error(self, store: MonitorStore) -> None:
        store.insert(make_trace(None, query_hash="h1"))

        with pytest.raises(ValueError, match="capture_queries=True"):
            TestsetMiner(store).mine()

    def test_empty_window_raises(self, store: MonitorStore) -> None:
        with pytest.raises(ValueError, match="No traces found"):
            TestsetMiner(store).mine()

    def test_traces_without_text_are_skipped(self, store: MonitorStore) -> None:
        store.insert(make_trace("captured"))
        store.insert(make_trace(None, query_hash="uncaptured"))

        testset = TestsetMiner(store).mine()

        assert [q.text for q in testset.queries] == ["captured"]

    def test_testset_metadata_and_name(self, store: MonitorStore) -> None:
        store.insert(make_trace("q"))

        testset = TestsetMiner(store).mine(strategy="slow", name="custom")

        assert testset.name == "custom"
        assert testset.metadata["strategy"] == "slow"
        assert testset.metadata["unique_queries_in_window"] == 1

        default = TestsetMiner(store).mine()
        assert default.name == "production-frequent"


class TestCaptureFlow:
    """Query capture from client to store to miner."""

    def test_client_captures_scrubbed_query(self) -> None:
        from ragnarok_ai.monitor.client import MonitorClient
        from ragnarok_ai.privacy import PiiMode

        client = MonitorClient(capture_queries=True, pii_mode=PiiMode.REDACT)

        with client.trace("email bob@corp.com about CHF", force=True) as trace:
            pass

        assert trace.query_text == "email [REDACTED] about CHF"
        assert client._buffer[-1]["query_text"] == "email [REDACTED] about CHF"

    def test_capture_is_off_by_default(self) -> None:
        from ragnarok_ai.monitor.client import MonitorClient

        client = MonitorClient()

        with client.trace("What is CHF?", force=True) as trace:
            pass

        assert trace.query_text is None
        assert client._buffer[-1]["query_text"] is None

    def test_query_text_roundtrips_through_store(self, store: MonitorStore) -> None:
        store.insert(make_trace("What is CHF?"))

        loaded = store.get_traces()[0]

        assert loaded.query_text == "What is CHF?"
