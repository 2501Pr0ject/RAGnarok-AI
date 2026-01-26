"""Evaluation module for RAG pipelines.

This module provides functions to evaluate RAG pipelines against test sets.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ragnarok_ai.cache.base import CacheProtocol
    from ragnarok_ai.core.protocols import RAGProtocol
    from ragnarok_ai.core.types import Query, TestSet
    from ragnarok_ai.evaluators.retrieval import RetrievalMetrics
    from ragnarok_ai.telemetry.tracer import RAGTracer


@dataclass
class QueryResult:
    """Result of evaluating a single query."""

    query: Query
    metric: RetrievalMetrics
    answer: str
    latency_ms: float
    error: Exception | None = None


@dataclass
class ProgressInfo:
    """Progress information for callbacks."""

    current: int
    total: int
    query: Query | None = None
    result: QueryResult | None = None
    error: Exception | None = None


# Type alias for progress callback
ProgressCallback = Callable[[ProgressInfo], Any]


@dataclass
class EvaluationResult:
    """Result of evaluating a RAG pipeline against a test set."""

    testset: TestSet
    metrics: list[RetrievalMetrics]
    responses: list[str]
    query_results: list[QueryResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    errors: list[tuple[Query, Exception]] = field(default_factory=list)

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


class QueryTimeoutError(Exception):
    """Raised when a query exceeds the timeout."""

    pass


async def evaluate(
    rag_pipeline: RAGProtocol,
    testset: TestSet,
    *,
    k: int = 10,
    max_concurrency: int = 1,
    tracer: RAGTracer | None = None,
    on_progress: ProgressCallback | None = None,
    fail_fast: bool = True,
    cache: CacheProtocol | None = None,
    pipeline_id: str | None = None,
    timeout: float | None = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
) -> EvaluationResult:
    """Evaluate a RAG pipeline against a test set.

    Args:
        rag_pipeline: RAG pipeline implementing RAGProtocol.
        testset: Test set of queries with ground truth.
        k: Number of top results to consider for @K metrics.
        max_concurrency: Maximum number of concurrent query evaluations.
                        Set to 1 for sequential execution (default).
        tracer: Optional RAGTracer for OpenTelemetry tracing.
        on_progress: Optional callback for progress updates.
        fail_fast: If True, stop on first error. If False, continue and collect errors.
        cache: Optional cache for storing/retrieving evaluation results.
        pipeline_id: Optional identifier for cache key isolation.
        timeout: Optional timeout in seconds for each query. None means no timeout.
        max_retries: Maximum number of retries on failure (default: 0, no retries).
        retry_delay: Delay in seconds between retries (default: 1.0).

    Returns:
        EvaluationResult containing metrics and responses.

    Example:
        >>> from ragnarok_ai import evaluate
        >>> from ragnarok_ai.cache import MemoryCache
        >>>
        >>> # Sequential evaluation (default)
        >>> result = await evaluate(rag_pipeline, testset)
        >>>
        >>> # Parallel evaluation with 10 concurrent queries
        >>> result = await evaluate(rag_pipeline, testset, max_concurrency=10)
        >>>
        >>> # With timeout and retries
        >>> result = await evaluate(
        ...     rag_pipeline, testset,
        ...     timeout=30.0,      # 30 second timeout per query
        ...     max_retries=3,     # Retry up to 3 times
        ...     retry_delay=2.0,   # Wait 2 seconds between retries
        ... )
        >>>
        >>> # With caching
        >>> cache = MemoryCache()
        >>> result = await evaluate(rag_pipeline, testset, cache=cache)
        >>> print(cache.stats())  # Shows hit/miss statistics
    """
    if max_concurrency < 1:
        max_concurrency = 1

    # Sequential execution (original behavior)
    if max_concurrency == 1:
        return await _evaluate_sequential(
            rag_pipeline,
            testset,
            k=k,
            tracer=tracer,
            on_progress=on_progress,
            fail_fast=fail_fast,
            cache=cache,
            pipeline_id=pipeline_id,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    # Parallel execution
    return await _evaluate_parallel(
        rag_pipeline,
        testset,
        k=k,
        max_concurrency=max_concurrency,
        tracer=tracer,
        on_progress=on_progress,
        fail_fast=fail_fast,
        cache=cache,
        pipeline_id=pipeline_id,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


async def _evaluate_sequential(
    rag_pipeline: RAGProtocol,
    testset: TestSet,
    *,
    k: int = 10,
    tracer: RAGTracer | None = None,
    on_progress: ProgressCallback | None = None,
    fail_fast: bool = True,
    cache: CacheProtocol | None = None,
    pipeline_id: str | None = None,
    timeout: float | None = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
) -> EvaluationResult:
    """Evaluate queries sequentially (original behavior)."""
    metrics: list[RetrievalMetrics] = []
    responses: list[str] = []
    query_results: list[QueryResult] = []
    errors: list[tuple[Query, Exception]] = []
    total = len(testset.queries)
    start_time = time.perf_counter()

    async for query, metric, answer in evaluate_stream(
        rag_pipeline,
        testset,
        k=k,
        tracer=tracer,
        fail_fast=fail_fast,
        cache=cache,
        pipeline_id=pipeline_id,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    ):
        metrics.append(metric)
        responses.append(answer)

        if on_progress:
            info = ProgressInfo(current=len(metrics), total=total, query=query)
            callback_result = on_progress(info)
            if asyncio.iscoroutine(callback_result):
                await callback_result

    total_latency = (time.perf_counter() - start_time) * 1000
    return EvaluationResult(
        testset=testset,
        metrics=metrics,
        responses=responses,
        query_results=query_results,
        total_latency_ms=total_latency,
        errors=errors,
    )


async def _evaluate_parallel(
    rag_pipeline: RAGProtocol,
    testset: TestSet,
    *,
    k: int = 10,
    max_concurrency: int = 10,
    tracer: RAGTracer | None = None,
    on_progress: ProgressCallback | None = None,
    fail_fast: bool = True,
    cache: CacheProtocol | None = None,
    pipeline_id: str | None = None,
    timeout: float | None = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
) -> EvaluationResult:
    """Evaluate queries in parallel with concurrency control."""
    import logging

    from ragnarok_ai.cache.base import compute_cache_key
    from ragnarok_ai.telemetry.tracer import NoOpTracer

    logger = logging.getLogger(__name__)
    active_tracer = tracer if tracer is not None else NoOpTracer()
    semaphore = asyncio.Semaphore(max_concurrency)
    total = len(testset.queries)
    completed = 0
    completed_lock = asyncio.Lock()
    results: dict[int, QueryResult] = {}
    errors: list[tuple[Query, Exception]] = []
    cancel_event = asyncio.Event()
    start_time = time.perf_counter()

    async def evaluate_single(index: int, query: Query) -> None:
        nonlocal completed

        if cancel_event.is_set():
            return

        async with semaphore:
            if cancel_event.is_set():
                return

            # Check cache first (with error handling)
            if cache is not None:
                try:
                    cache_key = compute_cache_key(query, pipeline_id=pipeline_id, k=k)
                    cached_result = await cache.get(cache_key)
                    if cached_result is not None:
                        results[index] = cached_result
                        async with completed_lock:
                            completed += 1
                            current = completed
                        if on_progress:
                            info = ProgressInfo(current=current, total=total, query=query, result=cached_result)
                            callback_result = on_progress(info)
                            if asyncio.iscoroutine(callback_result):
                                await callback_result
                        return
                except Exception as e:
                    logger.warning(f"Cache read error for query {index}: {e}")
                    # Continue without cache

            query_result = await _evaluate_single_query_with_retry(
                rag_pipeline,
                query,
                index,
                k=k,
                tracer=active_tracer,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            results[index] = query_result

            # Store in cache (with error handling)
            if cache is not None and query_result.error is None:
                try:
                    cache_key = compute_cache_key(query, pipeline_id=pipeline_id, k=k)
                    await cache.set(cache_key, query_result)
                except Exception as e:
                    logger.warning(f"Cache write error for query {index}: {e}")

            async with completed_lock:
                completed += 1
                current = completed

            if query_result.error:
                errors.append((query, query_result.error))
                if fail_fast:
                    cancel_event.set()
                    return

            if on_progress:
                info = ProgressInfo(
                    current=current,
                    total=total,
                    query=query,
                    result=query_result,
                )
                callback_result = on_progress(info)
                if asyncio.iscoroutine(callback_result):
                    await callback_result

    with active_tracer.start_span(
        "evaluate",
        attributes={
            "ragnarok.num_queries": total,
            "ragnarok.k": k,
            "ragnarok.max_concurrency": max_concurrency,
        },
    ) as eval_span:
        # Create tasks for all queries
        tasks = [asyncio.create_task(evaluate_single(i, query)) for i, query in enumerate(testset.queries)]

        # Wait for all tasks (or until cancelled)
        await asyncio.gather(*tasks, return_exceptions=True)

        eval_span.set_attribute("ragnarok.completed", not cancel_event.is_set())
        eval_span.set_attribute("ragnarok.num_errors", len(errors))

    # Build ordered results
    metrics: list[RetrievalMetrics] = []
    responses: list[str] = []
    query_results: list[QueryResult] = []

    for i in range(total):
        if i in results:
            qr = results[i]
            query_results.append(qr)
            if qr.error is None:
                metrics.append(qr.metric)
                responses.append(qr.answer)

    total_latency = (time.perf_counter() - start_time) * 1000

    # Re-raise first error if fail_fast
    if fail_fast and errors:
        from ragnarok_ai.core.exceptions import EvaluationError

        query, error = errors[0]
        raise EvaluationError(f"Failed to evaluate query '{query.text}': {error}") from error

    return EvaluationResult(
        testset=testset,
        metrics=metrics,
        responses=responses,
        query_results=query_results,
        total_latency_ms=total_latency,
        errors=errors,
    )


async def _evaluate_single_query_with_retry(
    rag_pipeline: RAGProtocol,
    query: Query,
    index: int,
    *,
    k: int = 10,
    tracer: RAGTracer | None = None,
    timeout: float | None = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
) -> QueryResult:
    """Evaluate a single query with timeout and retry support."""
    from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

    last_error: Exception | None = None
    attempts = max_retries + 1  # Total attempts = retries + 1

    for attempt in range(attempts):
        try:
            if timeout is not None:
                result = await asyncio.wait_for(
                    _evaluate_single_query(rag_pipeline, query, index, k=k, tracer=tracer),
                    timeout=timeout,
                )
            else:
                result = await _evaluate_single_query(rag_pipeline, query, index, k=k, tracer=tracer)

            # If successful or error is not retryable, return
            if result.error is None:
                return result

            last_error = result.error

        except asyncio.TimeoutError:
            last_error = QueryTimeoutError(f"Query timed out after {timeout}s")

        except Exception as e:
            last_error = e

        # If we have more attempts, wait before retrying
        if attempt < attempts - 1:
            await asyncio.sleep(retry_delay)

    # All retries exhausted, return error result
    return QueryResult(
        query=query,
        metric=RetrievalMetrics(precision=0, recall=0, mrr=0, ndcg=0, k=k),
        answer="",
        latency_ms=0,
        error=last_error,
    )


async def _evaluate_single_query(
    rag_pipeline: RAGProtocol,
    query: Query,
    index: int,
    *,
    k: int = 10,
    tracer: RAGTracer | None = None,
) -> QueryResult:
    """Evaluate a single query and return the result."""
    from ragnarok_ai.core.types import RetrievalResult
    from ragnarok_ai.evaluators.retrieval import RetrievalMetrics, evaluate_retrieval
    from ragnarok_ai.telemetry.tracer import NoOpTracer

    active_tracer = tracer if tracer is not None else NoOpTracer()

    query_attrs: dict[str, Any] = {
        "ragnarok.query_index": index,
        "ragnarok.query_text": query.text[:200],
        "ragnarok.num_ground_truth_docs": len(query.ground_truth_docs),
    }

    with active_tracer.start_span(f"query_{index}", attributes=query_attrs) as query_span:
        start_time = time.perf_counter()

        try:
            # RAG query
            with active_tracer.start_span("rag_query") as rag_span:
                rag_start = time.perf_counter()
                response = await rag_pipeline.query(query.text)
                rag_latency = (time.perf_counter() - rag_start) * 1000

                rag_span.set_attribute("ragnarok.latency_ms", rag_latency)
                rag_span.set_attribute("ragnarok.num_retrieved_docs", len(response.retrieved_docs))
                rag_span.set_attribute(
                    "ragnarok.retrieved_doc_ids",
                    [doc.id for doc in response.retrieved_docs[:10]],
                )
                rag_span.set_attribute("ragnarok.answer_length", len(response.answer))

            # Metrics calculation
            with active_tracer.start_span("evaluate_metrics") as metrics_span:
                metrics_start = time.perf_counter()
                retrieval_result = RetrievalResult(
                    query=query,
                    retrieved_docs=response.retrieved_docs,
                )
                metric = evaluate_retrieval(retrieval_result, k=k)
                metrics_latency = (time.perf_counter() - metrics_start) * 1000

                metrics_span.set_attribute("ragnarok.latency_ms", metrics_latency)
                metrics_span.set_attribute("ragnarok.precision", metric.precision)
                metrics_span.set_attribute("ragnarok.recall", metric.recall)
                metrics_span.set_attribute("ragnarok.mrr", metric.mrr)
                metrics_span.set_attribute("ragnarok.ndcg", metric.ndcg)

            query_span.set_attribute("ragnarok.precision", metric.precision)
            query_span.set_attribute("ragnarok.recall", metric.recall)

            total_latency = (time.perf_counter() - start_time) * 1000
            return QueryResult(
                query=query,
                metric=metric,
                answer=response.answer,
                latency_ms=total_latency,
            )

        except Exception as e:
            total_latency = (time.perf_counter() - start_time) * 1000
            return QueryResult(
                query=query,
                metric=RetrievalMetrics(precision=0, recall=0, mrr=0, ndcg=0, k=k),
                answer="",
                latency_ms=total_latency,
                error=e,
            )


async def evaluate_stream(
    rag_pipeline: RAGProtocol,
    testset: TestSet,
    *,
    k: int = 10,
    tracer: RAGTracer | None = None,
    fail_fast: bool = True,
    cache: CacheProtocol | None = None,
    pipeline_id: str | None = None,
    timeout: float | None = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
) -> AsyncIterator[tuple[Query, RetrievalMetrics, str]]:
    """Evaluate a RAG pipeline against a test set, yielding results as they complete.

    Args:
        rag_pipeline: RAG pipeline implementing RAGProtocol.
        testset: Test set of queries with ground truth.
        k: Number of top results to consider for @K metrics.
        tracer: Optional RAGTracer for OpenTelemetry tracing.
        fail_fast: If True, stop on first error. If False, skip failed queries.
        cache: Optional cache for storing/retrieving evaluation results.
        pipeline_id: Optional identifier for cache key isolation.
        timeout: Optional timeout in seconds for each query.
        max_retries: Maximum number of retries on failure.
        retry_delay: Delay in seconds between retries.

    Yields:
        Tuples of (query, metrics, response) for each query.

    Raises:
        EvaluationError: If evaluation fails for any query and fail_fast is True.
    """
    import logging

    from ragnarok_ai.cache.base import compute_cache_key
    from ragnarok_ai.core.exceptions import EvaluationError
    from ragnarok_ai.telemetry.tracer import NoOpTracer

    logger = logging.getLogger(__name__)
    active_tracer = tracer if tracer is not None else NoOpTracer()

    with active_tracer.start_span(
        "evaluate",
        attributes={
            "ragnarok.num_queries": len(testset.queries),
            "ragnarok.k": k,
        },
    ) as eval_span:
        for i, query in enumerate(testset):
            # Check cache first (with error handling)
            if cache is not None:
                try:
                    cache_key = compute_cache_key(query, pipeline_id=pipeline_id, k=k)
                    cached_result = await cache.get(cache_key)
                    if cached_result is not None:
                        yield query, cached_result.metric, cached_result.answer
                        continue
                except Exception as e:
                    logger.warning(f"Cache read error for query {i}: {e}")
                    # Continue without cache

            result = await _evaluate_single_query_with_retry(
                rag_pipeline,
                query,
                i,
                k=k,
                tracer=active_tracer,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )

            # Store in cache (with error handling)
            if cache is not None and result.error is None:
                try:
                    cache_key = compute_cache_key(query, pipeline_id=pipeline_id, k=k)
                    await cache.set(cache_key, result)
                except Exception as e:
                    logger.warning(f"Cache write error for query {i}: {e}")

            if result.error:
                if fail_fast:
                    raise EvaluationError(f"Failed to evaluate query '{query.text}': {result.error}") from result.error
                continue

            yield query, result.metric, result.answer

        eval_span.set_attribute("ragnarok.completed", True)
