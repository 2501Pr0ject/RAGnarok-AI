"""Prometheus metrics formatting for production monitoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragnarok_ai.monitor.store import MonitorStore


# Histogram buckets for latency (in seconds)
LATENCY_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]


class PrometheusExporter:
    """Export metrics in Prometheus text format."""

    def __init__(self, store: MonitorStore) -> None:
        """Initialize the exporter.

        Args:
            store: MonitorStore instance to read metrics from.
        """
        self.store = store

    def export(self) -> str:
        """Generate Prometheus-format metrics.

        Returns:
            Prometheus text format metrics string.
        """
        lines: list[str] = []

        # Get current stats
        now = datetime.now(timezone.utc)
        last_24h = now - timedelta(hours=24)
        last_1h = now - timedelta(hours=1)

        total_count = self.store.count()
        count_last_hour = self.store.count_since(last_1h)
        success_rate = self.store.get_success_rate(since=last_24h)
        p50, p95, p99 = self.store.get_latency_percentiles(since=last_24h)

        # Convert to seconds for Prometheus
        p50_sec = p50 / 1000.0
        p95_sec = p95 / 1000.0
        p99_sec = p99 / 1000.0

        # Total requests counter
        lines.append("# HELP ragnarok_requests_total Total number of RAG requests")
        lines.append("# TYPE ragnarok_requests_total counter")

        # Calculate success/error counts
        success_count = int(total_count * success_rate)
        error_count = total_count - success_count

        lines.append(f'ragnarok_requests_total{{status="success"}} {success_count}')
        lines.append(f'ragnarok_requests_total{{status="error"}} {error_count}')
        lines.append("")

        # Requests in last hour (gauge)
        lines.append("# HELP ragnarok_requests_last_hour Requests in the last hour")
        lines.append("# TYPE ragnarok_requests_last_hour gauge")
        lines.append(f"ragnarok_requests_last_hour {count_last_hour}")
        lines.append("")

        # Success rate gauge
        lines.append("# HELP ragnarok_success_rate Success rate (0.0-1.0)")
        lines.append("# TYPE ragnarok_success_rate gauge")
        lines.append(f"ragnarok_success_rate {success_rate:.4f}")
        lines.append("")

        # Latency summary (percentiles)
        lines.append("# HELP ragnarok_latency_seconds RAG request latency in seconds")
        lines.append("# TYPE ragnarok_latency_seconds summary")
        lines.append(f'ragnarok_latency_seconds{{quantile="0.5"}} {p50_sec:.6f}')
        lines.append(f'ragnarok_latency_seconds{{quantile="0.95"}} {p95_sec:.6f}')
        lines.append(f'ragnarok_latency_seconds{{quantile="0.99"}} {p99_sec:.6f}')
        lines.append(f"ragnarok_latency_seconds_count {total_count}")
        lines.append("")

        # Monitor uptime info
        last_trace = self.store.get_last_trace_time()
        if last_trace:
            seconds_since_last = (now - last_trace).total_seconds()
            lines.append("# HELP ragnarok_last_trace_seconds Seconds since last trace")
            lines.append("# TYPE ragnarok_last_trace_seconds gauge")
            lines.append(f"ragnarok_last_trace_seconds {seconds_since_last:.1f}")
            lines.append("")

        return "\n".join(lines)


def format_prometheus_metrics(store: MonitorStore) -> str:
    """Convenience function to export Prometheus metrics.

    Args:
        store: MonitorStore instance.

    Returns:
        Prometheus text format metrics string.
    """
    exporter = PrometheusExporter(store)
    return exporter.export()
