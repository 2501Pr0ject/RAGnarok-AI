"""Monitor client for instrumenting RAG pipelines.

Usage:
    from ragnarok_ai.monitor import MonitorClient

    client = MonitorClient(endpoint="http://localhost:9090", sample_rate=0.1)

    async def handle_query(query: str) -> str:
        with client.trace(query) as trace:
            docs = await retriever.search(query)
            trace.record_retrieval(docs, latency_ms=120.5)

            answer = await llm.generate(query, docs)
            trace.record_generation(answer, latency_ms=450.2)

            return answer
"""

from __future__ import annotations

import hashlib
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# Default configuration
DEFAULT_ENDPOINT = "http://localhost:9090"
DEFAULT_SAMPLE_RATE = 0.1  # 10% sampling


@dataclass
class TraceContext:
    """Context for a single trace.

    Use within a context manager to automatically record timing and send traces.
    """

    client: MonitorClient
    query_hash: str
    query_length: int
    is_sampled: bool
    force: bool = False

    # Timing
    start_time: float = field(default_factory=time.perf_counter)

    # Recorded metrics
    retrieval_latency_ms: float | None = None
    retrieval_count: int | None = None
    generation_latency_ms: float | None = None
    answer_length: int | None = None
    model_version: str | None = None
    success: bool = True
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_retrieval(
        self,
        docs: Sequence[Any] | None = None,
        latency_ms: float | None = None,
        count: int | None = None,
    ) -> None:
        """Record retrieval metrics.

        Args:
            docs: Retrieved documents (used for count if count not provided).
            latency_ms: Retrieval latency in milliseconds.
            count: Number of documents retrieved.
        """
        if not self.is_sampled and not self.force:
            return

        self.retrieval_latency_ms = latency_ms
        if count is not None:
            self.retrieval_count = count
        elif docs is not None:
            self.retrieval_count = len(docs)

    def record_generation(
        self,
        answer: str | None = None,
        latency_ms: float | None = None,
        model: str | None = None,
    ) -> None:
        """Record generation metrics.

        Args:
            answer: Generated answer (used for length).
            latency_ms: Generation latency in milliseconds.
            model: Model version used.
        """
        if not self.is_sampled and not self.force:
            return

        self.generation_latency_ms = latency_ms
        if answer is not None:
            self.answer_length = len(answer)
        if model is not None:
            self.model_version = model

    def record_error(self, error: Exception | str) -> None:
        """Record an error that occurred during the trace.

        Args:
            error: The exception or error message.
        """
        self.success = False
        if isinstance(error, Exception):
            self.error_type = type(error).__name__
        else:
            self.error_type = str(error)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the trace.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        if self.is_sampled or self.force:
            self.metadata[key] = value

    def _get_total_latency_ms(self) -> float:
        """Calculate total latency from start time."""
        return (time.perf_counter() - self.start_time) * 1000


class MonitorClient:
    """Client for sending traces to the monitor daemon.

    Provides sampling to reduce overhead and a context manager interface
    for easy instrumentation.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        enabled: bool = True,
    ) -> None:
        """Initialize the monitor client.

        Args:
            endpoint: Monitor daemon URL (e.g., "http://localhost:9090").
            sample_rate: Fraction of traces to sample (0.0-1.0).
            enabled: Whether monitoring is enabled.
        """
        self.endpoint = endpoint.rstrip("/")
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.enabled = enabled
        self._buffer: list[dict[str, Any]] = []
        self._buffer_size = 100  # Batch size before flush

    def _should_sample(self) -> bool:
        """Determine if this request should be sampled."""
        return random.random() < self.sample_rate

    def _hash_query(self, query: str) -> str:
        """Hash a query for PII safety.

        Args:
            query: The query string.

        Returns:
            Truncated SHA256 hash of the query.
        """
        return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]

    @contextmanager
    def trace(self, query: str, force: bool = False) -> Iterator[TraceContext]:
        """Context manager for tracing a RAG request.

        Args:
            query: The user query.
            force: Force tracing regardless of sampling.

        Yields:
            TraceContext for recording metrics.

        Example:
            with client.trace("What is RAG?") as trace:
                docs = retriever.search(query)
                trace.record_retrieval(docs, latency_ms=50)
                answer = llm.generate(query, docs)
                trace.record_generation(answer, latency_ms=200)
        """
        is_sampled = force or (self.enabled and self._should_sample())

        ctx = TraceContext(
            client=self,
            query_hash=self._hash_query(query),
            query_length=len(query),
            is_sampled=is_sampled,
            force=force,
        )

        try:
            yield ctx
        except Exception as e:
            ctx.record_error(e)
            raise
        finally:
            if is_sampled:
                self._record_trace(ctx)

    def _record_trace(self, ctx: TraceContext) -> None:
        """Record a completed trace to buffer.

        Args:
            ctx: The completed trace context.
        """
        trace_data = {
            "id": uuid4().hex[:16],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_hash": ctx.query_hash,
            "query_length": ctx.query_length,
            "retrieval_latency_ms": ctx.retrieval_latency_ms,
            "retrieval_count": ctx.retrieval_count,
            "generation_latency_ms": ctx.generation_latency_ms,
            "answer_length": ctx.answer_length,
            "total_latency_ms": ctx._get_total_latency_ms(),
            "model_version": ctx.model_version,
            "success": ctx.success,
            "error_type": ctx.error_type,
            "metadata": ctx.metadata if ctx.metadata else None,
        }

        self._buffer.append(trace_data)

        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self) -> int:
        """Send buffered traces to daemon.

        Returns:
            Number of traces sent.
        """
        if not self._buffer:
            return 0

        traces = self._buffer.copy()
        self._buffer.clear()

        try:
            self._send_traces(traces)
            return len(traces)
        except Exception:
            # Re-buffer on failure (with limit to prevent memory issues)
            if len(self._buffer) < self._buffer_size * 10:
                self._buffer.extend(traces)
            return 0

    def _send_traces(self, traces: list[dict[str, Any]]) -> None:
        """Send traces to daemon via HTTP.

        Args:
            traces: List of trace dictionaries.
        """
        import json
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(self.endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or 9090

        # Simple HTTP POST
        body = json.dumps({"traces": traces})
        request = (
            f"POST /ingest HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{body}"
        )

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.sendall(request.encode("utf-8"))
            # Read response (basic validation)
            response = sock.recv(1024)
            if b"200" not in response:
                raise RuntimeError(f"Ingest failed: {response[:100].decode('utf-8', errors='replace')}")

    def __enter__(self) -> MonitorClient:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - flush on exit."""
        self.flush()
