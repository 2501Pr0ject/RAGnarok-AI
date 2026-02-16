"""Tests for monitor models."""

from datetime import datetime, timezone

from ragnarok_ai.monitor.models import (
    AggregateMetrics,
    IngestRequest,
    IngestResponse,
    TraceEvent,
)


class TestTraceEvent:
    """Tests for TraceEvent model."""

    def test_create_minimal(self) -> None:
        """Test creating a trace with minimal fields."""
        trace = TraceEvent(
            query_hash="abc123",
            query_length=42,
            total_latency_ms=100.5,
        )

        assert trace.query_hash == "abc123"
        assert trace.query_length == 42
        assert trace.total_latency_ms == 100.5
        assert trace.success is True
        assert trace.id is not None
        assert len(trace.id) == 16
        assert trace.timestamp is not None

    def test_create_full(self) -> None:
        """Test creating a trace with all fields."""
        ts = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        trace = TraceEvent(
            id="custom_id_12345",
            timestamp=ts,
            query_hash="xyz789",
            query_length=100,
            retrieval_latency_ms=50.0,
            retrieval_count=5,
            generation_latency_ms=200.0,
            answer_length=150,
            total_latency_ms=250.0,
            model_version="mistral:7b",
            success=True,
            metadata={"tenant": "acme"},
        )

        assert trace.id == "custom_id_12345"
        assert trace.timestamp == ts
        assert trace.retrieval_latency_ms == 50.0
        assert trace.retrieval_count == 5
        assert trace.generation_latency_ms == 200.0
        assert trace.answer_length == 150
        assert trace.model_version == "mistral:7b"
        assert trace.metadata == {"tenant": "acme"}

    def test_create_error_trace(self) -> None:
        """Test creating a trace for failed request."""
        trace = TraceEvent(
            query_hash="err123",
            query_length=10,
            total_latency_ms=50.0,
            success=False,
            error_type="TimeoutError",
        )

        assert trace.success is False
        assert trace.error_type == "TimeoutError"

    def test_auto_generated_id(self) -> None:
        """Test that IDs are auto-generated and unique."""
        trace1 = TraceEvent(query_hash="a", query_length=1, total_latency_ms=1.0)
        trace2 = TraceEvent(query_hash="a", query_length=1, total_latency_ms=1.0)

        assert trace1.id != trace2.id

    def test_auto_generated_timestamp(self) -> None:
        """Test that timestamps are auto-generated."""
        before = datetime.now(timezone.utc)
        trace = TraceEvent(query_hash="a", query_length=1, total_latency_ms=1.0)
        after = datetime.now(timezone.utc)

        assert before <= trace.timestamp <= after


class TestAggregateMetrics:
    """Tests for AggregateMetrics model."""

    def test_create_aggregate(self) -> None:
        """Test creating aggregate metrics."""
        hour = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        agg = AggregateMetrics(
            hour=hour,
            total_requests=100,
            success_count=98,
            error_count=2,
            latency_p50=120.0,
            latency_p95=350.0,
            latency_p99=500.0,
            latency_avg=150.0,
            retrieval_latency_avg=50.0,
            generation_latency_avg=100.0,
        )

        assert agg.hour == hour
        assert agg.total_requests == 100
        assert agg.success_count == 98
        assert agg.error_count == 2
        assert agg.latency_p50 == 120.0
        assert agg.latency_p95 == 350.0
        assert agg.latency_p99 == 500.0

    def test_aggregate_without_retrieval_generation(self) -> None:
        """Test aggregate without retrieval/generation breakdown."""
        hour = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        agg = AggregateMetrics(
            hour=hour,
            total_requests=50,
            success_count=50,
            error_count=0,
            latency_p50=100.0,
            latency_p95=200.0,
            latency_p99=300.0,
            latency_avg=110.0,
        )

        assert agg.retrieval_latency_avg is None
        assert agg.generation_latency_avg is None


class TestIngestModels:
    """Tests for ingestion request/response models."""

    def test_ingest_request(self) -> None:
        """Test IngestRequest model."""
        traces = [
            TraceEvent(query_hash="a", query_length=10, total_latency_ms=100.0),
            TraceEvent(query_hash="b", query_length=20, total_latency_ms=200.0),
        ]
        request = IngestRequest(traces=traces)

        assert len(request.traces) == 2
        assert request.traces[0].query_hash == "a"

    def test_ingest_response(self) -> None:
        """Test IngestResponse model."""
        response = IngestResponse(accepted=10, dropped=2)

        assert response.accepted == 10
        assert response.dropped == 2
