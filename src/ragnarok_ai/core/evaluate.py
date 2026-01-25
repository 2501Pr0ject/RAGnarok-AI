"""Evaluation module for RAG pipelines.

This module provides functions to evaluate RAG pipelines against test sets.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ragnarok_ai.core.protocols import RAGProtocol
    from ragnarok_ai.core.types import Query, TestSet
    from ragnarok_ai.evaluators.retrieval import RetrievalMetrics
    from ragnarok_ai.telemetry.tracer import RAGTracer


class EvaluationResult:
    """Result of evaluating a RAG pipeline against a test set."""

    def __init__(
        self,
        testset: TestSet,
        metrics: list[RetrievalMetrics],
        responses: list[str],
    ) -> None:
        self.testset = testset
        self.metrics = metrics
        self.responses = responses

    def summary(self) -> dict[str, float]:
        if not self.metrics:
            return {}
        n = len(self.metrics)
        return {
            "num_queries": n,
            "precision": sum(m.precision for m in self.metrics) / n,
            "recall": sum(m.recall for m in self.metrics) / n,
            "mrr": sum(m.mrr for m in self.metrics) / n,
            "ndcg": sum(m.ndcg for m in self.metrics) / n,
        }


async def evaluate(
    rag_pipeline: RAGProtocol,
    testset: TestSet,
    *,
    k: int = 10,
    tracer: RAGTracer | None = None,
) -> EvaluationResult:
    """Evaluate a RAG pipeline against a test set.

    Args:
        rag_pipeline: RAG pipeline implementing RAGProtocol.
        testset: Test set of queries with ground truth.
        k: Number of top results to consider for @K metrics.
        tracer: Optional RAGTracer for OpenTelemetry tracing.

    Returns:
        EvaluationResult containing metrics and responses.

    Example:
        >>> from ragnarok_ai import evaluate
        >>> from ragnarok_ai.telemetry import RAGTracer, OTLPExporter
        >>>
        >>> # Without tracing
        >>> result = await evaluate(rag_pipeline, testset)
        >>>
        >>> # With tracing
        >>> exporter = OTLPExporter(endpoint="http://localhost:4317")
        >>> tracer = RAGTracer(exporter=exporter.get_span_exporter())
        >>> result = await evaluate(rag_pipeline, testset, tracer=tracer)
    """
    metrics: list[RetrievalMetrics] = []
    responses: list[str] = []

    async for _, metric, answer in evaluate_stream(rag_pipeline, testset, k=k, tracer=tracer):
        metrics.append(metric)
        responses.append(answer)

    return EvaluationResult(testset, metrics, responses)


async def evaluate_stream(
    rag_pipeline: RAGProtocol,
    testset: TestSet,
    *,
    k: int = 10,
    tracer: RAGTracer | None = None,
) -> AsyncIterator[tuple[Query, RetrievalMetrics, str]]:
    """Evaluate a RAG pipeline against a test set, yielding results as they complete.

    Args:
        rag_pipeline: RAG pipeline implementing RAGProtocol.
        testset: Test set of queries with ground truth.
        k: Number of top results to consider for @K metrics.
        tracer: Optional RAGTracer for OpenTelemetry tracing.

    Yields:
        Tuples of (query, metrics, response) for each query.

    Raises:
        EvaluationError: If evaluation fails for any query.
    """
    from ragnarok_ai.core.exceptions import EvaluationError
    from ragnarok_ai.core.types import RetrievalResult
    from ragnarok_ai.evaluators.retrieval import evaluate_retrieval
    from ragnarok_ai.telemetry.tracer import NoOpTracer

    # Use provided tracer or no-op
    active_tracer = tracer if tracer is not None else NoOpTracer()

    with active_tracer.start_span(
        "evaluate",
        attributes={
            "ragnarok.num_queries": len(testset.queries),
            "ragnarok.k": k,
        },
    ) as eval_span:
        for i, query in enumerate(testset):
            query_attrs: dict[str, Any] = {
                "ragnarok.query_index": i,
                "ragnarok.query_text": query.text[:200],  # Truncate for safety
                "ragnarok.num_ground_truth_docs": len(query.ground_truth_docs),
            }

            with active_tracer.start_span(f"query_{i}", attributes=query_attrs) as query_span:
                # RAG query
                with active_tracer.start_span("rag_query") as rag_span:
                    start_time = time.perf_counter()
                    try:
                        response = await rag_pipeline.query(query.text)
                    except Exception as e:
                        raise EvaluationError(f"Failed to query pipeline for '{query.text}': {e}") from e
                    query_latency = time.perf_counter() - start_time

                    rag_span.set_attribute("ragnarok.latency_ms", query_latency * 1000)
                    rag_span.set_attribute("ragnarok.num_retrieved_docs", len(response.retrieved_docs))
                    rag_span.set_attribute(
                        "ragnarok.retrieved_doc_ids",
                        [doc.id for doc in response.retrieved_docs[:10]],  # Limit to first 10
                    )
                    rag_span.set_attribute("ragnarok.answer_length", len(response.answer))

                # Metrics calculation
                with active_tracer.start_span("evaluate_metrics") as metrics_span:
                    start_time = time.perf_counter()
                    try:
                        retrieval_result = RetrievalResult(
                            query=query,
                            retrieved_docs=response.retrieved_docs,
                        )
                        metric = evaluate_retrieval(retrieval_result, k=k)
                    except Exception as e:
                        raise EvaluationError(f"Failed to calculate metrics for '{query.text}': {e}") from e
                    metrics_latency = time.perf_counter() - start_time

                    metrics_span.set_attribute("ragnarok.latency_ms", metrics_latency * 1000)
                    metrics_span.set_attribute("ragnarok.precision", metric.precision)
                    metrics_span.set_attribute("ragnarok.recall", metric.recall)
                    metrics_span.set_attribute("ragnarok.mrr", metric.mrr)
                    metrics_span.set_attribute("ragnarok.ndcg", metric.ndcg)

                # Update query span with results
                query_span.set_attribute("ragnarok.precision", metric.precision)
                query_span.set_attribute("ragnarok.recall", metric.recall)

            yield query, metric, response.answer

        # Set evaluation-level attributes
        eval_span.set_attribute("ragnarok.completed", True)
