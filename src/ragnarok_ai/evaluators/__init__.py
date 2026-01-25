"""Evaluators module for ragnarok-ai.

This module provides evaluation metrics for RAG systems,
including retrieval quality and generation quality metrics.
"""

from __future__ import annotations

from ragnarok_ai.evaluators.faithfulness import (
    ClaimVerification,
    FaithfulnessEvaluator,
    FaithfulnessResult,
)
from ragnarok_ai.evaluators.hallucination import (
    Hallucination,
    HallucinationDetector,
    HallucinationResult,
)
from ragnarok_ai.evaluators.relevance import (
    RelevanceEvaluator,
    RelevanceResult,
)
from ragnarok_ai.evaluators.retrieval import (
    RetrievalMetrics,
    evaluate_retrieval,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "ClaimVerification",
    "FaithfulnessEvaluator",
    "FaithfulnessResult",
    "Hallucination",
    "HallucinationDetector",
    "HallucinationResult",
    "RelevanceEvaluator",
    "RelevanceResult",
    "RetrievalMetrics",
    "evaluate_retrieval",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
