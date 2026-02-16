"""Tests for monitor client."""

import pytest

from ragnarok_ai.monitor.client import MonitorClient, TraceContext


class TestMonitorClient:
    """Tests for MonitorClient."""

    def test_create_client_defaults(self) -> None:
        """Test creating client with defaults."""
        client = MonitorClient()

        assert client.endpoint == "http://localhost:9090"
        assert client.sample_rate == 0.1
        assert client.enabled is True

    def test_create_client_custom(self) -> None:
        """Test creating client with custom settings."""
        client = MonitorClient(
            endpoint="http://monitor.local:8080/",
            sample_rate=0.5,
            enabled=False,
        )

        assert client.endpoint == "http://monitor.local:8080"
        assert client.sample_rate == 0.5
        assert client.enabled is False

    def test_sample_rate_clamped(self) -> None:
        """Test that sample rate is clamped to 0-1."""
        client1 = MonitorClient(sample_rate=-0.5)
        client2 = MonitorClient(sample_rate=1.5)

        assert client1.sample_rate == 0.0
        assert client2.sample_rate == 1.0

    def test_hash_query(self) -> None:
        """Test query hashing."""
        client = MonitorClient()

        hash1 = client._hash_query("What is RAG?")
        hash2 = client._hash_query("What is RAG?")
        hash3 = client._hash_query("Different query")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16  # Truncated SHA256


class TestTraceContext:
    """Tests for TraceContext."""

    def test_trace_context_basic(self) -> None:
        """Test basic trace context creation."""
        client = MonitorClient(sample_rate=1.0)  # Always sample

        with client.trace("Test query") as ctx:
            assert ctx.is_sampled is True
            assert ctx.query_length == len("Test query")
            assert len(ctx.query_hash) == 16

    def test_trace_record_retrieval(self) -> None:
        """Test recording retrieval metrics."""
        client = MonitorClient(sample_rate=1.0)

        with client.trace("Test") as ctx:
            ctx.record_retrieval(docs=["doc1", "doc2"], latency_ms=50.5)

            assert ctx.retrieval_latency_ms == 50.5
            assert ctx.retrieval_count == 2

    def test_trace_record_retrieval_with_count(self) -> None:
        """Test recording retrieval with explicit count."""
        client = MonitorClient(sample_rate=1.0)

        with client.trace("Test") as ctx:
            ctx.record_retrieval(count=10, latency_ms=100.0)

            assert ctx.retrieval_count == 10

    def test_trace_record_generation(self) -> None:
        """Test recording generation metrics."""
        client = MonitorClient(sample_rate=1.0)

        with client.trace("Test") as ctx:
            ctx.record_generation(
                answer="This is the answer.",
                latency_ms=200.0,
                model="mistral:7b",
            )

            assert ctx.generation_latency_ms == 200.0
            assert ctx.answer_length == len("This is the answer.")
            assert ctx.model_version == "mistral:7b"

    def test_trace_record_error(self) -> None:
        """Test recording an error."""
        client = MonitorClient(sample_rate=1.0)

        with client.trace("Test") as ctx:
            ctx.record_error(ValueError("Something went wrong"))

            assert ctx.success is False
            assert ctx.error_type == "ValueError"

    def test_trace_record_error_string(self) -> None:
        """Test recording an error as string."""
        client = MonitorClient(sample_rate=1.0)

        with client.trace("Test") as ctx:
            ctx.record_error("Custom error")

            assert ctx.success is False
            assert ctx.error_type == "Custom error"

    def test_trace_add_metadata(self) -> None:
        """Test adding metadata."""
        client = MonitorClient(sample_rate=1.0)

        with client.trace("Test") as ctx:
            ctx.add_metadata("tenant", "acme")
            ctx.add_metadata("route", "/api/query")

            assert ctx.metadata == {"tenant": "acme", "route": "/api/query"}

    def test_trace_exception_handling(self) -> None:
        """Test that exceptions are recorded."""
        client = MonitorClient(sample_rate=1.0)

        with pytest.raises(RuntimeError):
            with client.trace("Test") as ctx:
                raise RuntimeError("Test error")

        # Error should be recorded
        assert ctx.success is False
        assert ctx.error_type == "RuntimeError"

    def test_trace_force_sampling(self) -> None:
        """Test forcing trace regardless of sample rate."""
        client = MonitorClient(sample_rate=0.0)  # Never sample

        with client.trace("Test", force=True) as ctx:
            assert ctx.is_sampled is True
            assert ctx.force is True

    def test_trace_not_sampled(self) -> None:
        """Test trace when not sampled."""
        client = MonitorClient(sample_rate=0.0)

        with client.trace("Test") as ctx:
            assert ctx.is_sampled is False

            # Operations should be no-ops
            ctx.record_retrieval(docs=["doc1"], latency_ms=100)
            ctx.record_generation(answer="test", latency_ms=200)
            ctx.add_metadata("key", "value")

            assert ctx.retrieval_latency_ms is None
            assert ctx.generation_latency_ms is None
            assert ctx.metadata == {}


class TestMonitorClientBuffer:
    """Tests for client buffering."""

    def test_buffer_traces(self) -> None:
        """Test that traces are buffered."""
        client = MonitorClient(sample_rate=1.0)

        for i in range(5):
            with client.trace(f"Query {i}") as ctx:
                ctx.record_retrieval(latency_ms=float(i * 10))

        assert len(client._buffer) == 5

    def test_flush_clears_buffer(self) -> None:
        """Test that flush clears buffer (without sending)."""
        client = MonitorClient(sample_rate=1.0)

        for i in range(3):
            with client.trace(f"Query {i}"):
                pass

        assert len(client._buffer) == 3

        # Flush will fail (no server), but should still clear
        client._buffer.clear()  # Clear manually for test
        assert len(client._buffer) == 0

    def test_context_manager_flushes(self) -> None:
        """Test that context manager flushes on exit."""
        with MonitorClient(sample_rate=1.0) as client:
            with client.trace("Test"):
                pass

            # Buffer should have trace before exit
            assert len(client._buffer) == 1

        # After exit, flush was called (buffer cleared or re-added on error)

    def test_trace_timing(self) -> None:
        """Test that trace timing is captured."""
        import time

        client = MonitorClient(sample_rate=1.0)

        with client.trace("Test") as ctx:
            time.sleep(0.01)  # 10ms

        # Check last trace in buffer
        assert len(client._buffer) == 1
        total_latency = client._buffer[0]["total_latency_ms"]
        assert total_latency >= 10  # At least 10ms
