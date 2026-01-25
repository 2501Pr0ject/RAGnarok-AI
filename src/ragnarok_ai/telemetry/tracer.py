"""Core tracer implementation for ragnarok-ai.

This module provides a tracer abstraction that wraps OpenTelemetry
for tracing RAG evaluation steps.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

# Global tracer instance
_tracer: RAGTracer | None = None


class SpanContext:
    """Context manager for a tracing span."""

    def __init__(
        self,
        name: str,
        tracer: RAGTracer,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._tracer = tracer
        self._attributes = attributes or {}
        self._start_time: float = 0
        self._span: Any = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span."""
        self._attributes[key] = value
        if self._span is not None and hasattr(self._span, "set_attribute"):
            self._span.set_attribute(key, _normalize_attribute(value))

    def set_status_ok(self) -> None:
        """Mark the span as successful."""
        if self._span is not None and hasattr(self._span, "set_status"):
            try:
                from opentelemetry.trace import StatusCode

                self._span.set_status(StatusCode.OK)
            except ImportError:
                pass

    def set_status_error(self, message: str) -> None:
        """Mark the span as errored."""
        if self._span is not None and hasattr(self._span, "set_status"):
            try:
                from opentelemetry.trace import StatusCode

                self._span.set_status(StatusCode.ERROR, message)
            except ImportError:
                pass

    def record_exception(self, exception: BaseException) -> None:
        """Record an exception on the span."""
        if self._span is not None and hasattr(self._span, "record_exception"):
            self._span.record_exception(exception)


class RAGTracer:
    """Tracer for RAG evaluation operations.

    Provides a high-level interface for tracing RAG operations
    with OpenTelemetry. Falls back to no-op if OpenTelemetry
    is not installed.

    Example:
        >>> tracer = RAGTracer(service_name="my-rag-eval")
        >>> with tracer.start_span("evaluate") as span:
        ...     span.set_attribute("num_queries", 10)
        ...     # evaluation logic
    """

    def __init__(
        self,
        service_name: str = "ragnarok-ai",
        *,
        exporter: Any | None = None,
    ) -> None:
        """Initialize the tracer.

        Args:
            service_name: Name of the service for traces.
            exporter: Optional OpenTelemetry span exporter.
        """
        self._service_name = service_name
        self._exporter = exporter
        self._otel_tracer: Any = None
        self._provider: Any = None
        self._enabled = False

        self._setup_tracer()

    def _setup_tracer(self) -> None:
        """Set up OpenTelemetry tracer if available."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import SERVICE_NAME, Resource
            from opentelemetry.sdk.trace import TracerProvider

            resource = Resource(attributes={SERVICE_NAME: self._service_name})
            self._provider = TracerProvider(resource=resource)

            if self._exporter is not None:
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                self._provider.add_span_processor(BatchSpanProcessor(self._exporter))

            trace.set_tracer_provider(self._provider)
            self._otel_tracer = trace.get_tracer(self._service_name)
            self._enabled = True
        except ImportError:
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[SpanContext, None, None]:
        """Start a new span.

        Args:
            name: Name of the span.
            attributes: Initial attributes for the span.

        Yields:
            SpanContext for the span.
        """
        ctx = SpanContext(name, self, attributes)
        ctx._start_time = time.perf_counter()

        if self._enabled and self._otel_tracer is not None:
            with self._otel_tracer.start_as_current_span(name) as span:
                ctx._span = span
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, _normalize_attribute(value))
                try:
                    yield ctx
                except Exception as e:
                    ctx.record_exception(e)
                    ctx.set_status_error(str(e))
                    raise
                else:
                    ctx.set_status_ok()
        else:
            yield ctx

    def shutdown(self) -> None:
        """Shutdown the tracer and flush pending spans."""
        if self._provider is not None and hasattr(self._provider, "shutdown"):
            self._provider.shutdown()


class NoOpTracer(RAGTracer):
    """No-op tracer that does nothing.

    Used when telemetry is disabled or OpenTelemetry is not installed.
    """

    def __init__(self) -> None:
        """Initialize without setting up OpenTelemetry."""
        self._service_name = "noop"
        self._exporter = None
        self._otel_tracer = None
        self._provider = None
        self._enabled = False

    def _setup_tracer(self) -> None:
        """No-op setup."""
        pass

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[SpanContext, None, None]:
        """No-op span context."""
        yield SpanContext(name, self, attributes)

    def shutdown(self) -> None:
        """No-op shutdown."""
        pass


def _normalize_attribute(value: Any) -> Any:
    """Normalize a value for OpenTelemetry attributes.

    OpenTelemetry only supports primitive types and lists of primitives.

    Args:
        value: Value to normalize.

    Returns:
        Normalized value suitable for OpenTelemetry.
    """
    if value is None:
        return ""
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        return [_normalize_attribute(v) for v in value]
    return str(value)


def get_tracer() -> RAGTracer:
    """Get the global tracer instance.

    Returns:
        The global RAGTracer, or a NoOpTracer if not set.
    """
    global _tracer
    if _tracer is None:
        return NoOpTracer()
    return _tracer


def set_tracer(tracer: RAGTracer | None) -> None:
    """Set the global tracer instance.

    Args:
        tracer: The tracer to use globally, or None to disable.
    """
    global _tracer
    _tracer = tracer
