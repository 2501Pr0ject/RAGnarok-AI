"""Tests for Prometheus metrics export."""

import pytest

from ragnarok_ai.monitor.metrics import PrometheusExporter, format_prometheus_metrics
from ragnarok_ai.monitor.models import TraceEvent
from ragnarok_ai.monitor.store import MonitorStore


@pytest.fixture
def store() -> MonitorStore:
    """Create an in-memory store for testing."""
    return MonitorStore(db_path=":memory:")


class TestPrometheusExporter:
    """Tests for PrometheusExporter."""

    def test_export_empty_store(self, store: MonitorStore) -> None:
        """Test export with no traces."""
        exporter = PrometheusExporter(store)
        metrics = exporter.export()

        assert "ragnarok_requests_total" in metrics
        assert "ragnarok_success_rate" in metrics
        assert "ragnarok_latency_seconds" in metrics

    def test_export_with_traces(self, store: MonitorStore) -> None:
        """Test export with traces."""
        # Insert some traces
        traces = [
            TraceEvent(
                query_hash=f"h{i}",
                query_length=10,
                total_latency_ms=float(100 + i * 10),
                success=(i < 9),  # 9 success, 1 error
            )
            for i in range(10)
        ]
        store.insert_batch(traces)

        exporter = PrometheusExporter(store)
        metrics = exporter.export()

        # Check counters
        assert 'ragnarok_requests_total{status="success"} 9' in metrics
        assert 'ragnarok_requests_total{status="error"} 1' in metrics

        # Check success rate
        assert "ragnarok_success_rate 0.9" in metrics

        # Check latency summary
        assert 'ragnarok_latency_seconds{quantile="0.5"}' in metrics
        assert 'ragnarok_latency_seconds{quantile="0.95"}' in metrics
        assert 'ragnarok_latency_seconds{quantile="0.99"}' in metrics
        assert "ragnarok_latency_seconds_count 10" in metrics

    def test_export_format_valid_prometheus(self, store: MonitorStore) -> None:
        """Test that export produces valid Prometheus format."""
        traces = [TraceEvent(query_hash=f"h{i}", query_length=10, total_latency_ms=100.0) for i in range(5)]
        store.insert_batch(traces)

        exporter = PrometheusExporter(store)
        metrics = exporter.export()

        # Check format: lines are either comments, empty, or metric
        for line in metrics.split("\n"):
            if line.strip():
                assert (
                    line.startswith("#")  # Comment
                    or line.startswith("ragnarok_")  # Metric
                ), f"Invalid line: {line}"

    def test_format_prometheus_metrics_function(self, store: MonitorStore) -> None:
        """Test convenience function."""
        traces = [TraceEvent(query_hash="h1", query_length=10, total_latency_ms=100.0)]
        store.insert_batch(traces)

        metrics = format_prometheus_metrics(store)

        assert "ragnarok_requests_total" in metrics
        assert "ragnarok_latency_seconds" in metrics

    def test_export_includes_last_trace_info(self, store: MonitorStore) -> None:
        """Test that last trace time is included."""
        traces = [TraceEvent(query_hash="h1", query_length=10, total_latency_ms=100.0)]
        store.insert_batch(traces)

        exporter = PrometheusExporter(store)
        metrics = exporter.export()

        assert "ragnarok_last_trace_seconds" in metrics

    def test_export_latency_in_seconds(self, store: MonitorStore) -> None:
        """Test that latency is exported in seconds, not milliseconds."""
        traces = [TraceEvent(query_hash=f"h{i}", query_length=10, total_latency_ms=500.0) for i in range(10)]
        store.insert_batch(traces)

        exporter = PrometheusExporter(store)
        metrics = exporter.export()

        # 500ms = 0.5 seconds
        # The p50 should be around 0.5
        for line in metrics.split("\n"):
            if 'ragnarok_latency_seconds{quantile="0.5"}' in line:
                value = float(line.split()[-1])
                assert 0.4 <= value <= 0.6, f"Expected ~0.5s, got {value}"
                break
