"""Telemetry module for ragnarok-ai.

This module provides OpenTelemetry integration for tracing RAG evaluations.
"""

from __future__ import annotations

from ragnarok_ai.telemetry.exporters import (
    OTLPExporter,
    create_otlp_exporter,
)
from ragnarok_ai.telemetry.tracer import (
    RAGTracer,
    get_tracer,
    set_tracer,
)

__all__ = [
    "OTLPExporter",
    "RAGTracer",
    "create_otlp_exporter",
    "get_tracer",
    "set_tracer",
]
