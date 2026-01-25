from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import RAGProtocol
    from ragnarok_ai.core.types import Query, TestSet
    from ragnarok_ai.evaluators.retrieval import RetrievalMetrics


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
) -> EvaluationResult:
    """Evaluate a RAG pipeline against a test set.

    Args:
        rag_pipeline: RAG pipeline implementing RAGProtocol.
        testset: Test set of queries with ground truth.
        k: Number of top results to consider for @K metrics.

    Returns:
        EvaluationResult containing metrics and responses.
    """
    metrics: list[RetrievalMetrics] = []
    responses: list[str] = []

    async for _, metric, answer in evaluate_stream(rag_pipeline, testset, k=k):
        metrics.append(metric)
        responses.append(answer)

    return EvaluationResult(testset, metrics, responses)


async def evaluate_stream(
    rag_pipeline: RAGProtocol,
    testset: TestSet,
    *,
    k: int = 10,
) -> AsyncIterator[tuple[Query, RetrievalMetrics, str]]:
    """Evaluate a RAG pipeline against a test set, yielding results as they complete.

    Args:
        rag_pipeline: RAG pipeline implementing RAGProtocol.
        testset: Test set of queries with ground truth.
        k: Number of top results to consider for @K metrics.

    Yields:
        Tuples of (query, metrics, response) for each query.

    Raises:
        EvaluationError: If evaluation fails for any query.
    """
    from ragnarok_ai.core.exceptions import EvaluationError
    from ragnarok_ai.core.types import RetrievalResult
    from ragnarok_ai.evaluators.retrieval import evaluate_retrieval

    for query in testset:
        try:
            response = await rag_pipeline.query(query.text)
        except Exception as e:
            raise EvaluationError(f"Failed to query pipeline for '{query.text}': {e}") from e

        try:
            retrieval_result = RetrievalResult(
                query=query,
                retrieved_docs=response.retrieved_docs,
            )
            metric = evaluate_retrieval(retrieval_result, k=k)
        except Exception as e:
            raise EvaluationError(f"Failed to calculate metrics for '{query.text}': {e}") from e

        yield query, metric, response.answer
