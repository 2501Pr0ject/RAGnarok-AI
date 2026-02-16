"""Tests for monitor SQLite store."""

from datetime import datetime, timedelta, timezone

import pytest

from ragnarok_ai.monitor.models import TraceEvent
from ragnarok_ai.monitor.store import MonitorStore


@pytest.fixture
def store() -> MonitorStore:
    """Create an in-memory store for testing."""
    return MonitorStore(db_path=":memory:")


@pytest.fixture
def sample_trace() -> TraceEvent:
    """Create a sample trace event."""
    return TraceEvent(
        query_hash="abc123",
        query_length=42,
        retrieval_latency_ms=50.0,
        retrieval_count=5,
        generation_latency_ms=150.0,
        answer_length=100,
        total_latency_ms=200.0,
        model_version="mistral:7b",
        success=True,
    )


class TestMonitorStoreBasics:
    """Basic store operations tests."""

    def test_insert_and_count(self, store: MonitorStore, sample_trace: TraceEvent) -> None:
        """Test inserting a trace and counting."""
        assert store.count() == 0
        store.insert(sample_trace)
        assert store.count() == 1

    def test_insert_batch(self, store: MonitorStore) -> None:
        """Test batch insertion."""
        traces = [
            TraceEvent(query_hash=f"hash{i}", query_length=i * 10, total_latency_ms=float(i * 100))
            for i in range(10)
        ]
        inserted = store.insert_batch(traces)
        assert inserted == 10
        assert store.count() == 10

    def test_insert_batch_empty(self, store: MonitorStore) -> None:
        """Test batch insertion with empty list."""
        inserted = store.insert_batch([])
        assert inserted == 0
        assert store.count() == 0

    def test_insert_with_metadata(self, store: MonitorStore) -> None:
        """Test inserting trace with metadata."""
        trace = TraceEvent(
            query_hash="meta123",
            query_length=50,
            total_latency_ms=100.0,
            metadata={"tenant": "acme", "route": "/api/query"},
        )
        store.insert(trace)
        assert store.count() == 1


class TestMonitorStoreMetrics:
    """Metrics calculation tests."""

    def test_success_rate_all_success(self, store: MonitorStore) -> None:
        """Test success rate with all successful traces."""
        traces = [
            TraceEvent(query_hash=f"h{i}", query_length=10, total_latency_ms=100.0, success=True)
            for i in range(10)
        ]
        store.insert_batch(traces)

        assert store.get_success_rate() == 1.0

    def test_success_rate_mixed(self, store: MonitorStore) -> None:
        """Test success rate with mixed results."""
        # 8 success, 2 failures
        traces = [
            TraceEvent(
                query_hash=f"h{i}",
                query_length=10,
                total_latency_ms=100.0,
                success=(i < 8),
            )
            for i in range(10)
        ]
        store.insert_batch(traces)

        assert store.get_success_rate() == 0.8

    def test_success_rate_empty(self, store: MonitorStore) -> None:
        """Test success rate with no traces."""
        assert store.get_success_rate() == 1.0

    def test_success_rate_since(self, store: MonitorStore) -> None:
        """Test success rate with time filter."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(hours=2)
        recent_time = now - timedelta(minutes=30)

        # Old traces (all fail)
        old_traces = [
            TraceEvent(
                query_hash=f"old{i}",
                query_length=10,
                total_latency_ms=100.0,
                success=False,
                timestamp=old_time,
            )
            for i in range(5)
        ]
        # Recent traces (all success)
        recent_traces = [
            TraceEvent(
                query_hash=f"new{i}",
                query_length=10,
                total_latency_ms=100.0,
                success=True,
                timestamp=recent_time,
            )
            for i in range(5)
        ]
        store.insert_batch(old_traces + recent_traces)

        # Total success rate should be 50%
        assert store.get_success_rate() == 0.5

        # Last hour: 100% success
        assert store.get_success_rate(since=now - timedelta(hours=1)) == 1.0

    def test_latency_percentiles(self, store: MonitorStore) -> None:
        """Test latency percentile calculation."""
        # Insert 100 traces with latencies 1-100ms
        traces = [
            TraceEvent(query_hash=f"h{i}", query_length=10, total_latency_ms=float(i + 1))
            for i in range(100)
        ]
        store.insert_batch(traces)

        p50, p95, p99 = store.get_latency_percentiles()

        # With 1-100ms latencies:
        # p50 should be around 50ms
        # p95 should be around 95ms
        # p99 should be around 99ms
        assert 45 <= p50 <= 55
        assert 90 <= p95 <= 100
        assert 95 <= p99 <= 100

    def test_latency_percentiles_few_samples(self, store: MonitorStore) -> None:
        """Test percentiles with few samples."""
        traces = [
            TraceEvent(query_hash=f"h{i}", query_length=10, total_latency_ms=float(i * 100))
            for i in range(3)
        ]
        store.insert_batch(traces)

        _p50, _p95, p99 = store.get_latency_percentiles()
        # With few samples, p95/p99 should be max value
        assert p99 == 200.0

    def test_latency_percentiles_empty(self, store: MonitorStore) -> None:
        """Test percentiles with no traces."""
        p50, p95, p99 = store.get_latency_percentiles()
        assert (p50, p95, p99) == (0.0, 0.0, 0.0)

    def test_count_since(self, store: MonitorStore) -> None:
        """Test counting traces since a timestamp."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(hours=2)
        recent_time = now - timedelta(minutes=30)

        old_traces = [
            TraceEvent(query_hash=f"old{i}", query_length=10, total_latency_ms=100.0, timestamp=old_time)
            for i in range(5)
        ]
        recent_traces = [
            TraceEvent(query_hash=f"new{i}", query_length=10, total_latency_ms=100.0, timestamp=recent_time)
            for i in range(3)
        ]
        store.insert_batch(old_traces + recent_traces)

        assert store.count() == 8
        assert store.count_since(now - timedelta(hours=1)) == 3


class TestMonitorStoreAggregation:
    """Aggregation tests."""

    def test_aggregate_hour(self, store: MonitorStore) -> None:
        """Test hourly aggregation."""
        hour = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Insert traces within the hour
        traces = [
            TraceEvent(
                query_hash=f"h{i}",
                query_length=10,
                total_latency_ms=float(100 + i * 10),
                retrieval_latency_ms=float(30 + i),
                generation_latency_ms=float(70 + i * 5),
                success=(i < 9),  # 9 success, 1 failure
                timestamp=hour + timedelta(minutes=i * 5),
            )
            for i in range(10)
        ]
        store.insert_batch(traces)

        agg = store.aggregate_hour(hour)

        assert agg is not None
        assert agg.hour == hour
        assert agg.total_requests == 10
        assert agg.success_count == 9
        assert agg.error_count == 1
        assert agg.retrieval_latency_avg is not None
        assert agg.generation_latency_avg is not None

    def test_aggregate_hour_empty(self, store: MonitorStore) -> None:
        """Test aggregation for hour with no traces."""
        hour = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        agg = store.aggregate_hour(hour)
        assert agg is None

    def test_store_and_retrieve_aggregate(self, store: MonitorStore) -> None:
        """Test storing and retrieving aggregates."""
        hour = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Insert traces and compute aggregate
        traces = [
            TraceEvent(
                query_hash=f"h{i}",
                query_length=10,
                total_latency_ms=float(100 + i * 10),
                timestamp=hour + timedelta(minutes=i * 5),
            )
            for i in range(10)
        ]
        store.insert_batch(traces)

        agg = store.aggregate_hour(hour)
        assert agg is not None

        store.store_aggregate(agg)

        # Retrieve
        retrieved = store.get_aggregate(hour)
        assert retrieved is not None
        assert retrieved.hour == agg.hour
        assert retrieved.total_requests == agg.total_requests
        assert retrieved.latency_p50 == agg.latency_p50

    def test_get_aggregate_not_found(self, store: MonitorStore) -> None:
        """Test retrieving non-existent aggregate."""
        hour = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        agg = store.get_aggregate(hour)
        assert agg is None


class TestMonitorStorePurge:
    """Retention and purge tests."""

    def test_purge_old_traces(self) -> None:
        """Test purging old traces."""
        store = MonitorStore(db_path=":memory:", retention_days=7)
        now = datetime.now(timezone.utc)

        # Old traces (10 days ago)
        old_traces = [
            TraceEvent(
                query_hash=f"old{i}",
                query_length=10,
                total_latency_ms=100.0,
                timestamp=now - timedelta(days=10),
            )
            for i in range(5)
        ]
        # Recent traces (1 day ago)
        recent_traces = [
            TraceEvent(
                query_hash=f"new{i}",
                query_length=10,
                total_latency_ms=100.0,
                timestamp=now - timedelta(days=1),
            )
            for i in range(3)
        ]
        store.insert_batch(old_traces + recent_traces)

        assert store.count() == 8

        deleted = store.purge_old_traces()

        assert deleted == 5
        assert store.count() == 3

    def test_get_last_trace_time(self, store: MonitorStore) -> None:
        """Test getting last trace timestamp."""
        now = datetime.now(timezone.utc)

        # Initially no traces
        assert store.get_last_trace_time() is None

        # Insert traces
        traces = [
            TraceEvent(
                query_hash=f"h{i}",
                query_length=10,
                total_latency_ms=100.0,
                timestamp=now - timedelta(minutes=i * 10),
            )
            for i in range(3)
        ]
        store.insert_batch(traces)

        last = store.get_last_trace_time()
        assert last is not None
        # Most recent should be 'now' (i=0)
        assert abs((last - now).total_seconds()) < 1
