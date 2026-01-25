"""OpenTelemetry exporters for ragnarok-ai.

This module provides OTLP exporters for sending traces to
OpenTelemetry-compatible backends like Jaeger, Phoenix, or Grafana Tempo.
"""

from __future__ import annotations

from typing import Any, Literal


class OTLPExporter:
    """OTLP exporter configuration.

    Wraps OpenTelemetry OTLP exporters for both gRPC and HTTP protocols.

    Example:
        >>> exporter = OTLPExporter(endpoint="http://localhost:4317")
        >>> tracer = RAGTracer(exporter=exporter.get_span_exporter())

        >>> # HTTP exporter
        >>> exporter = OTLPExporter(
        ...     endpoint="http://localhost:4318/v1/traces",
        ...     protocol="http",
        ... )
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        *,
        protocol: Literal["grpc", "http"] = "grpc",
        headers: dict[str, str] | None = None,
        timeout: int = 30,
        insecure: bool = True,
    ) -> None:
        """Initialize the OTLP exporter.

        Args:
            endpoint: OTLP collector endpoint.
                     Default gRPC: http://localhost:4317
                     Default HTTP: http://localhost:4318/v1/traces
            protocol: Protocol to use ("grpc" or "http").
            headers: Optional headers to send with requests.
            timeout: Request timeout in seconds.
            insecure: Whether to use insecure connection (no TLS).
        """
        self._endpoint = endpoint
        self._protocol = protocol
        self._headers = headers or {}
        self._timeout = timeout
        self._insecure = insecure
        self._exporter: Any = None

    @property
    def endpoint(self) -> str:
        """Get the configured endpoint."""
        return self._endpoint

    @property
    def protocol(self) -> str:
        """Get the configured protocol."""
        return self._protocol

    def get_span_exporter(self) -> Any:
        """Get the OpenTelemetry span exporter.

        Returns:
            Configured OTLP span exporter.

        Raises:
            ImportError: If required OpenTelemetry packages are not installed.
        """
        if self._exporter is not None:
            return self._exporter

        if self._protocol == "grpc":
            self._exporter = self._create_grpc_exporter()
        else:
            self._exporter = self._create_http_exporter()

        return self._exporter

    def _create_grpc_exporter(self) -> Any:
        """Create a gRPC OTLP exporter."""
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError as e:
            raise ImportError(
                "opentelemetry-exporter-otlp-proto-grpc is required for gRPC export. "
                "Install with: pip install ragnarok-ai[telemetry]"
            ) from e

        return OTLPSpanExporter(
            endpoint=self._endpoint,
            headers=self._headers or None,
            timeout=self._timeout,
            insecure=self._insecure,
        )

    def _create_http_exporter(self) -> Any:
        """Create an HTTP OTLP exporter."""
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError as e:
            raise ImportError(
                "opentelemetry-exporter-otlp-proto-http is required for HTTP export. "
                "Install with: pip install ragnarok-ai[telemetry]"
            ) from e

        return OTLPSpanExporter(
            endpoint=self._endpoint,
            headers=self._headers or None,
            timeout=self._timeout,
        )


def create_otlp_exporter(
    endpoint: str = "http://localhost:4317",
    *,
    protocol: Literal["grpc", "http"] = "grpc",
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    insecure: bool = True,
) -> OTLPExporter:
    """Create an OTLP exporter.

    Convenience function for creating an OTLP exporter.

    Args:
        endpoint: OTLP collector endpoint.
        protocol: Protocol to use ("grpc" or "http").
        headers: Optional headers to send with requests.
        timeout: Request timeout in seconds.
        insecure: Whether to use insecure connection (no TLS).

    Returns:
        Configured OTLPExporter.

    Example:
        >>> exporter = create_otlp_exporter("http://localhost:4317")
        >>> # Use with RAGTracer
        >>> from ragnarok_ai.telemetry import RAGTracer
        >>> tracer = RAGTracer(exporter=exporter.get_span_exporter())
    """
    return OTLPExporter(
        endpoint=endpoint,
        protocol=protocol,
        headers=headers,
        timeout=timeout,
        insecure=insecure,
    )
