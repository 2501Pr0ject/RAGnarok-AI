"""Pydantic models for production monitoring."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """Single production trace event.

    Captures metrics from a single RAG pipeline execution.
    Query text is hashed for PII safety.
    """

    id: str = Field(default_factory=lambda: uuid4().hex[:16])
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Query (hashed for PII safety)
    query_hash: str
    query_length: int

    # Retrieval metrics
    retrieval_latency_ms: float | None = None
    retrieval_count: int | None = None

    # Generation metrics
    generation_latency_ms: float | None = None
    answer_length: int | None = None

    # Total
    total_latency_ms: float

    # Context
    model_version: str | None = None
    success: bool = True
    error_type: str | None = None

    # Extra metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class AggregateMetrics(BaseModel):
    """Hourly aggregated metrics.

    Pre-computed aggregates for efficient querying and Prometheus export.
    """

    hour: datetime  # Truncated to hour

    # Counts
    total_requests: int
    success_count: int
    error_count: int

    # Latency distribution (milliseconds)
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_avg: float

    # Retrieval
    retrieval_latency_avg: float | None = None

    # Generation
    generation_latency_avg: float | None = None

    model_config = {"frozen": False}


class IngestRequest(BaseModel):
    """Request payload for trace ingestion."""

    traces: list[TraceEvent]


class IngestResponse(BaseModel):
    """Response for trace ingestion."""

    accepted: int
    dropped: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    uptime_seconds: float
    traces_collected: int


class StatsResponse(BaseModel):
    """Statistics response for CLI."""

    uptime_seconds: float
    traces_total: int
    traces_last_hour: int
    success_rate: float
    latency: LatencyStats


class LatencyStats(BaseModel):
    """Latency statistics."""

    p50: float
    p95: float
    p99: float
