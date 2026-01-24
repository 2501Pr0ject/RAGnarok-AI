"""Retrieval evaluation metrics for ragnarok-ai.

This module provides pure functions for calculating retrieval quality metrics:
- Precision@K: Proportion of retrieved documents that are relevant
- Recall@K: Proportion of relevant documents that were retrieved
- MRR: Mean Reciprocal Rank - position of first relevant result
- NDCG@K: Normalized Discounted Cumulative Gain - ranking quality

All functions are pure (no side effects) and don't require LLM calls.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ragnarok_ai.core.types import RetrievalResult


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate precision at K.

    Precision@K measures the proportion of retrieved documents (in top K)
    that are relevant.

    Args:
        retrieved_ids: List of retrieved document IDs, ordered by relevance.
        relevant_ids: List of ground truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Precision score between 0.0 and 1.0.

    Example:
        >>> precision_at_k(["a", "b", "c", "d"], ["a", "c", "e"], k=4)
        0.5
        >>> precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3)
        1.0
    """
    if k <= 0:
        return 0.0

    retrieved_at_k = retrieved_ids[:k]
    if not retrieved_at_k:
        return 0.0

    relevant_set = set(relevant_ids)
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)

    return relevant_retrieved / len(retrieved_at_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate recall at K.

    Recall@K measures the proportion of relevant documents that were
    retrieved in the top K results.

    Args:
        retrieved_ids: List of retrieved document IDs, ordered by relevance.
        relevant_ids: List of ground truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Recall score between 0.0 and 1.0.

    Example:
        >>> recall_at_k(["a", "b", "c", "d"], ["a", "c", "e"], k=4)
        0.6666666666666666
        >>> recall_at_k(["a", "b", "c"], ["a", "b"], k=3)
        1.0
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    relevant_retrieved = len(retrieved_at_k & relevant_set)

    return relevant_retrieved / len(relevant_set)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Calculate Mean Reciprocal Rank.

    MRR measures the position of the first relevant document in the
    retrieved results. Returns 1/rank of the first relevant result.

    Args:
        retrieved_ids: List of retrieved document IDs, ordered by relevance.
        relevant_ids: List of ground truth relevant document IDs.

    Returns:
        MRR score between 0.0 and 1.0. Returns 0.0 if no relevant doc found.

    Example:
        >>> mrr(["a", "b", "c"], ["b", "d"])
        0.5
        >>> mrr(["a", "b", "c"], ["a"])
        1.0
        >>> mrr(["a", "b", "c"], ["d", "e"])
        0.0
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.

    NDCG@K measures the quality of the ranking, giving higher scores
    when relevant documents appear earlier in the results.

    Uses binary relevance (1 if relevant, 0 if not).

    Args:
        retrieved_ids: List of retrieved document IDs, ordered by relevance.
        relevant_ids: List of ground truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        NDCG score between 0.0 and 1.0.

    Example:
        >>> ndcg_at_k(["a", "b", "c"], ["a", "c"], k=3)
        0.9197207891481876
        >>> ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], k=3)
        1.0
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    retrieved_at_k = retrieved_ids[:k]

    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_at_k, start=1):
        if doc_id in relevant_set:
            # Binary relevance: rel_i = 1 if relevant, 0 otherwise
            # DCG formula: sum of rel_i / log2(rank + 1)
            dcg += 1.0 / math.log2(rank + 1)

    # Calculate IDCG (Ideal DCG) - best possible ranking
    # All relevant docs at the top, up to k
    num_relevant_at_k = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, num_relevant_at_k + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


class RetrievalMetrics(BaseModel):
    """Aggregated retrieval evaluation metrics.

    Attributes:
        precision: Precision@K score.
        recall: Recall@K score.
        mrr: Mean Reciprocal Rank score.
        ndcg: NDCG@K score.
        k: The K value used for @K metrics.

    Example:
        >>> metrics = RetrievalMetrics(
        ...     precision=0.8,
        ...     recall=0.6,
        ...     mrr=1.0,
        ...     ndcg=0.92,
        ...     k=10,
        ... )
    """

    model_config = {"frozen": True}

    precision: float = Field(..., ge=0.0, le=1.0, description="Precision@K score")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall@K score")
    mrr: float = Field(..., ge=0.0, le=1.0, description="Mean Reciprocal Rank score")
    ndcg: float = Field(..., ge=0.0, le=1.0, description="NDCG@K score")
    k: int = Field(..., ge=1, description="K value used for @K metrics")


def evaluate_retrieval(result: RetrievalResult, k: int = 10) -> RetrievalMetrics:
    """Evaluate retrieval quality for a single query result.

    Computes all retrieval metrics (precision, recall, MRR, NDCG) for
    the given retrieval result against ground truth.

    Args:
        result: The retrieval result containing retrieved docs and ground truth.
        k: Number of top results to consider for @K metrics. Defaults to 10.

    Returns:
        RetrievalMetrics containing all computed metrics.

    Raises:
        ValueError: If k is less than 1.

    Example:
        >>> from ragnarok_ai.core.types import Query, Document, RetrievalResult
        >>> query = Query(text="What is RAG?", ground_truth_docs=["d1", "d3"])
        >>> docs = [
        ...     Document(id="d1", content="..."),
        ...     Document(id="d2", content="..."),
        ...     Document(id="d3", content="..."),
        ... ]
        >>> result = RetrievalResult(query=query, retrieved_docs=docs)
        >>> metrics = evaluate_retrieval(result, k=3)
        >>> print(f"Precision@3: {metrics.precision:.2f}")
        Precision@3: 0.67
    """
    if k < 1:
        msg = f"k must be at least 1, got {k}"
        raise ValueError(msg)

    retrieved_ids = [doc.id for doc in result.retrieved_docs]
    relevant_ids = result.query.ground_truth_docs

    return RetrievalMetrics(
        precision=precision_at_k(retrieved_ids, relevant_ids, k),
        recall=recall_at_k(retrieved_ids, relevant_ids, k),
        mrr=mrr(retrieved_ids, relevant_ids),
        ndcg=ndcg_at_k(retrieved_ids, relevant_ids, k),
        k=k,
    )
