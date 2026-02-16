"""Production monitoring for RAGnarok-AI.

This module provides production-grade monitoring capabilities:
- Trace collection with sampling
- SQLite storage for persistence
- Prometheus metrics export
- CLI commands for management

Usage:
    from ragnarok_ai.monitor import MonitorClient

    client = MonitorClient(endpoint="http://localhost:9090", sample_rate=0.1)

    with client.trace("What is RAG?") as trace:
        docs = retriever.search(query)
        trace.record_retrieval(docs, latency_ms=50)

        answer = llm.generate(query, docs)
        trace.record_generation(answer, latency_ms=200)
"""

from __future__ import annotations

from ragnarok_ai.monitor.client import MonitorClient, TraceContext
from ragnarok_ai.monitor.models import AggregateMetrics, TraceEvent

__all__ = [
    "AggregateMetrics",
    "MonitorClient",
    "TraceContext",
    "TraceEvent",
]
