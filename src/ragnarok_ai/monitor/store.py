"""SQLite storage for production monitoring traces."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, quantiles
from typing import TYPE_CHECKING

from ragnarok_ai.monitor.models import AggregateMetrics, TraceEvent

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# Default paths
DEFAULT_DB_PATH = Path.home() / ".ragnarok" / "monitor.db"
DEFAULT_RETENTION_DAYS = 7
DEFAULT_AGGREGATE_RETENTION_DAYS = 90


class MonitorStore:
    """SQLite-based storage for production traces.

    Handles:
    - Raw trace storage with automatic retention
    - Hourly aggregation for efficient querying
    - Prometheus-compatible metrics export
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        aggregate_retention_days: int = DEFAULT_AGGREGATE_RETENTION_DAYS,
    ) -> None:
        """Initialize the monitor store.

        Args:
            db_path: Path to SQLite database. Use ":memory:" for testing.
            retention_days: Days to keep raw traces.
            aggregate_retention_days: Days to keep hourly aggregates.
        """
        self._is_memory = db_path == ":memory:"
        self.retention_days = retention_days
        self.aggregate_retention_days = aggregate_retention_days
        self._persistent_conn: sqlite3.Connection | None = None

        self.db_path: str | Path
        if self._is_memory:
            self.db_path = ":memory:"
            # Create a persistent connection for in-memory database
            self._persistent_conn = sqlite3.connect(
                ":memory:",
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                check_same_thread=False,
            )
            self._persistent_conn.row_factory = sqlite3.Row
        else:  # pragma: no cover
            self.db_path = Path(db_path)
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        if self._is_memory and self._persistent_conn is not None:
            # Use persistent connection for in-memory database
            try:
                yield self._persistent_conn
                self._persistent_conn.commit()
            except Exception:  # pragma: no cover
                self._persistent_conn.rollback()
                raise
        else:  # pragma: no cover
            conn = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def close(self) -> None:
        """Close any persistent connections."""
        if self._persistent_conn is not None:
            self._persistent_conn.close()
            self._persistent_conn = None

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript(
                """
                -- Raw traces (kept for retention_days)
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    query_length INTEGER,
                    retrieval_latency_ms REAL,
                    retrieval_count INTEGER,
                    generation_latency_ms REAL,
                    answer_length INTEGER,
                    total_latency_ms REAL NOT NULL,
                    model_version TEXT,
                    success INTEGER NOT NULL DEFAULT 1,
                    error_type TEXT,
                    metadata TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_traces_timestamp
                    ON traces(timestamp);
                CREATE INDEX IF NOT EXISTS idx_traces_success
                    ON traces(success);

                -- Hourly aggregates (kept for aggregate_retention_days)
                CREATE TABLE IF NOT EXISTS aggregates (
                    hour TEXT PRIMARY KEY,
                    total_requests INTEGER NOT NULL,
                    success_count INTEGER NOT NULL,
                    error_count INTEGER NOT NULL,
                    latency_p50 REAL NOT NULL,
                    latency_p95 REAL NOT NULL,
                    latency_p99 REAL NOT NULL,
                    latency_avg REAL NOT NULL,
                    retrieval_latency_avg REAL,
                    generation_latency_avg REAL
                );
            """
            )

    def insert(self, trace: TraceEvent) -> None:
        """Insert a single trace event.

        Args:
            trace: The trace event to store.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO traces (
                    id, timestamp, query_hash, query_length,
                    retrieval_latency_ms, retrieval_count,
                    generation_latency_ms, answer_length,
                    total_latency_ms, model_version, success, error_type, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.id,
                    trace.timestamp.isoformat(),
                    trace.query_hash,
                    trace.query_length,
                    trace.retrieval_latency_ms,
                    trace.retrieval_count,
                    trace.generation_latency_ms,
                    trace.answer_length,
                    trace.total_latency_ms,
                    trace.model_version,
                    1 if trace.success else 0,
                    trace.error_type,
                    json.dumps(trace.metadata) if trace.metadata else None,
                ),
            )

    def insert_batch(self, traces: Sequence[TraceEvent]) -> int:
        """Insert multiple trace events.

        Args:
            traces: Sequence of trace events to store.

        Returns:
            Number of traces inserted.
        """
        if not traces:
            return 0

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO traces (
                    id, timestamp, query_hash, query_length,
                    retrieval_latency_ms, retrieval_count,
                    generation_latency_ms, answer_length,
                    total_latency_ms, model_version, success, error_type, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        t.id,
                        t.timestamp.isoformat(),
                        t.query_hash,
                        t.query_length,
                        t.retrieval_latency_ms,
                        t.retrieval_count,
                        t.generation_latency_ms,
                        t.answer_length,
                        t.total_latency_ms,
                        t.model_version,
                        1 if t.success else 0,
                        t.error_type,
                        json.dumps(t.metadata) if t.metadata else None,
                    )
                    for t in traces
                ],
            )
        return len(traces)

    def count(self) -> int:
        """Count total traces in storage."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM traces").fetchone()
            return row[0] if row else 0

    def count_since(self, since: datetime) -> int:
        """Count traces since a given timestamp."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM traces WHERE timestamp >= ?",
                (since.isoformat(),),
            ).fetchone()
            return row[0] if row else 0

    def get_success_rate(self, since: datetime | None = None) -> float:
        """Calculate success rate.

        Args:
            since: Optional start time. If None, uses all traces.

        Returns:
            Success rate as a float between 0.0 and 1.0.
        """
        with self._connect() as conn:
            if since:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
                    FROM traces WHERE timestamp >= ?
                    """,
                    (since.isoformat(),),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
                    FROM traces
                    """
                ).fetchone()

            if not row or row["total"] == 0:
                return 1.0  # No traces = 100% success (no failures)

            return float(row["successes"]) / float(row["total"])

    def get_latency_percentiles(self, since: datetime | None = None) -> tuple[float, float, float]:
        """Calculate latency percentiles (p50, p95, p99).

        Args:
            since: Optional start time. If None, uses all traces.

        Returns:
            Tuple of (p50, p95, p99) in milliseconds.
        """
        with self._connect() as conn:
            if since:
                rows = conn.execute(
                    "SELECT total_latency_ms FROM traces WHERE timestamp >= ?",
                    (since.isoformat(),),
                ).fetchall()
            else:
                rows = conn.execute("SELECT total_latency_ms FROM traces").fetchall()

            if not rows:
                return (0.0, 0.0, 0.0)

            latencies = sorted(row["total_latency_ms"] for row in rows)

            if len(latencies) < 4:
                # Not enough data for proper percentiles
                return (latencies[0], latencies[-1], latencies[-1])

            qs = quantiles(latencies, n=100)
            p50 = qs[49]
            p95 = qs[94] if len(latencies) >= 20 else latencies[-1]
            p99 = qs[98] if len(latencies) >= 100 else latencies[-1]

            return (p50, p95, p99)

    def aggregate_hour(self, hour: datetime) -> AggregateMetrics | None:
        """Compute aggregate metrics for a specific hour.

        Args:
            hour: The hour to aggregate (will be truncated to hour).

        Returns:
            AggregateMetrics for the hour, or None if no traces.
        """
        # Truncate to hour
        hour_start = hour.replace(minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    total_latency_ms,
                    retrieval_latency_ms,
                    generation_latency_ms,
                    success
                FROM traces
                WHERE timestamp >= ? AND timestamp < ?
                """,
                (hour_start.isoformat(), hour_end.isoformat()),
            ).fetchall()

            if not rows:
                return None

            latencies = [row["total_latency_ms"] for row in rows]
            retrieval_latencies = [
                row["retrieval_latency_ms"] for row in rows if row["retrieval_latency_ms"] is not None
            ]
            generation_latencies = [
                row["generation_latency_ms"] for row in rows if row["generation_latency_ms"] is not None
            ]

            total = len(rows)
            success_count = sum(1 for row in rows if row["success"])
            error_count = total - success_count

            # Calculate percentiles
            sorted_latencies = sorted(latencies)
            if len(sorted_latencies) >= 4:
                qs = quantiles(sorted_latencies, n=100)
                p50 = qs[49]
                p95 = qs[94] if len(sorted_latencies) >= 20 else sorted_latencies[-1]
                p99 = qs[98] if len(sorted_latencies) >= 100 else sorted_latencies[-1]
            else:  # pragma: no cover
                p50 = sorted_latencies[len(sorted_latencies) // 2]
                p95 = sorted_latencies[-1]
                p99 = sorted_latencies[-1]

            return AggregateMetrics(
                hour=hour_start,
                total_requests=total,
                success_count=success_count,
                error_count=error_count,
                latency_p50=p50,
                latency_p95=p95,
                latency_p99=p99,
                latency_avg=mean(latencies),
                retrieval_latency_avg=mean(retrieval_latencies) if retrieval_latencies else None,
                generation_latency_avg=mean(generation_latencies) if generation_latencies else None,
            )

    def store_aggregate(self, agg: AggregateMetrics) -> None:
        """Store or update an hourly aggregate.

        Args:
            agg: The aggregate metrics to store.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO aggregates (
                    hour, total_requests, success_count, error_count,
                    latency_p50, latency_p95, latency_p99, latency_avg,
                    retrieval_latency_avg, generation_latency_avg
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agg.hour.isoformat(),
                    agg.total_requests,
                    agg.success_count,
                    agg.error_count,
                    agg.latency_p50,
                    agg.latency_p95,
                    agg.latency_p99,
                    agg.latency_avg,
                    agg.retrieval_latency_avg,
                    agg.generation_latency_avg,
                ),
            )

    def get_aggregate(self, hour: datetime) -> AggregateMetrics | None:
        """Get stored aggregate for a specific hour.

        Args:
            hour: The hour to retrieve (will be truncated).

        Returns:
            Stored AggregateMetrics or None.
        """
        hour_start = hour.replace(minute=0, second=0, microsecond=0)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM aggregates WHERE hour = ?",
                (hour_start.isoformat(),),
            ).fetchone()

            if not row:
                return None

            return AggregateMetrics(
                hour=datetime.fromisoformat(row["hour"]),
                total_requests=row["total_requests"],
                success_count=row["success_count"],
                error_count=row["error_count"],
                latency_p50=row["latency_p50"],
                latency_p95=row["latency_p95"],
                latency_p99=row["latency_p99"],
                latency_avg=row["latency_avg"],
                retrieval_latency_avg=row["retrieval_latency_avg"],
                generation_latency_avg=row["generation_latency_avg"],
            )

    def purge_old_traces(self) -> int:
        """Remove traces older than retention period.

        Returns:
            Number of traces deleted.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM traces WHERE timestamp < ?",
                (cutoff.isoformat(),),
            )
            return cursor.rowcount

    def purge_old_aggregates(self) -> int:  # pragma: no cover
        """Remove aggregates older than aggregate retention period.

        Returns:
            Number of aggregates deleted.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.aggregate_retention_days)
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM aggregates WHERE hour < ?",
                (cutoff.isoformat(),),
            )
            return cursor.rowcount

    def get_last_trace_time(self) -> datetime | None:
        """Get timestamp of the most recent trace.

        Returns:
            Timestamp of last trace or None if no traces.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(timestamp) as last_ts FROM traces").fetchone()

            if not row or not row["last_ts"]:
                return None

            return datetime.fromisoformat(row["last_ts"])
