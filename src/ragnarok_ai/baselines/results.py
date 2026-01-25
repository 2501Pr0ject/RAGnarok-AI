"""Baseline expected results for ragnarok-ai.

This module provides pre-computed expected results for baseline configurations.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BaselineResult(BaseModel):
    """Expected results for a baseline configuration.

    Attributes:
        baseline_name: Name of the baseline.
        dataset: Dataset used for computing these results.
        precision_at_k: Expected precision@k scores.
        recall_at_k: Expected recall@k scores.
        mrr: Expected Mean Reciprocal Rank.
        ndcg: Expected Normalized Discounted Cumulative Gain.
        faithfulness: Expected faithfulness score (if applicable).
        relevance: Expected relevance score (if applicable).
        latency_ms: Expected latency in milliseconds.
        metadata: Additional result metadata.
    """

    baseline_name: str = Field(..., description="Baseline name")
    dataset: str = Field(default="ragnarok-reference", description="Reference dataset")
    precision_at_k: dict[int, float] = Field(default_factory=dict, description="Precision@k")
    recall_at_k: dict[int, float] = Field(default_factory=dict, description="Recall@k")
    mrr: float = Field(default=0.0, description="Mean Reciprocal Rank")
    ndcg: float = Field(default=0.0, description="NDCG")
    faithfulness: float | None = Field(default=None, description="Faithfulness score")
    relevance: float | None = Field(default=None, description="Relevance score")
    latency_ms: float | None = Field(default=None, description="Latency in ms")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def get_metric(self, metric_name: str, k: int | None = None) -> float | None:
        """Get a specific metric value.

        Args:
            metric_name: Name of the metric (precision, recall, mrr, ndcg, etc.).
            k: For @k metrics, the k value.

        Returns:
            The metric value, or None if not available.
        """
        metric_lower = metric_name.lower()

        if metric_lower in ("precision", "precision_at_k", "p@k"):
            if k is not None:
                return self.precision_at_k.get(k)
            # Return P@10 as default
            return self.precision_at_k.get(10)

        if metric_lower in ("recall", "recall_at_k", "r@k"):
            if k is not None:
                return self.recall_at_k.get(k)
            return self.recall_at_k.get(10)

        if metric_lower == "mrr":
            return self.mrr

        if metric_lower == "ndcg":
            return self.ndcg

        if metric_lower == "faithfulness":
            return self.faithfulness

        if metric_lower == "relevance":
            return self.relevance

        if metric_lower in ("latency", "latency_ms"):
            return self.latency_ms

        return None


# Pre-computed baseline results
# These are reference values based on standard benchmarks
BASELINE_RESULTS: dict[str, BaselineResult] = {
    "balanced": BaselineResult(
        baseline_name="balanced",
        dataset="ragnarok-reference-v1",
        precision_at_k={5: 0.78, 10: 0.75, 20: 0.70},
        recall_at_k={5: 0.45, 10: 0.62, 20: 0.78},
        mrr=0.72,
        ndcg=0.68,
        faithfulness=0.82,
        relevance=0.79,
        latency_ms=150,
        metadata={"benchmark_date": "2024-01-15"},
    ),
    "precision": BaselineResult(
        baseline_name="precision",
        dataset="ragnarok-reference-v1",
        precision_at_k={5: 0.85, 10: 0.82, 20: 0.76},
        recall_at_k={5: 0.52, 10: 0.70, 20: 0.85},
        mrr=0.80,
        ndcg=0.75,
        faithfulness=0.88,
        relevance=0.85,
        latency_ms=280,
        metadata={"benchmark_date": "2024-01-15"},
    ),
    "speed": BaselineResult(
        baseline_name="speed",
        dataset="ragnarok-reference-v1",
        precision_at_k={5: 0.70, 10: 0.68, 20: 0.62},
        recall_at_k={5: 0.38, 10: 0.52, 20: 0.68},
        mrr=0.65,
        ndcg=0.60,
        faithfulness=0.75,
        relevance=0.72,
        latency_ms=80,
        metadata={"benchmark_date": "2024-01-15"},
    ),
    "memory_efficient": BaselineResult(
        baseline_name="memory_efficient",
        dataset="ragnarok-reference-v1",
        precision_at_k={5: 0.72, 10: 0.70, 20: 0.65},
        recall_at_k={5: 0.40, 10: 0.55, 20: 0.70},
        mrr=0.67,
        ndcg=0.62,
        faithfulness=0.78,
        relevance=0.74,
        latency_ms=100,
        metadata={"benchmark_date": "2024-01-15"},
    ),
    "semantic": BaselineResult(
        baseline_name="semantic",
        dataset="ragnarok-reference-v1",
        precision_at_k={5: 0.80, 10: 0.78, 20: 0.72},
        recall_at_k={5: 0.48, 10: 0.65, 20: 0.80},
        mrr=0.75,
        ndcg=0.70,
        faithfulness=0.85,
        relevance=0.82,
        latency_ms=200,
        metadata={"benchmark_date": "2024-01-15"},
    ),
}


def get_baseline_result(name: str) -> BaselineResult:
    """Get expected results for a baseline.

    Args:
        name: Name of the baseline.

    Returns:
        The baseline expected results.

    Raises:
        KeyError: If baseline name is not found.
    """
    if name not in BASELINE_RESULTS:
        available = ", ".join(BASELINE_RESULTS.keys())
        msg = f"Unknown baseline '{name}'. Available: {available}"
        raise KeyError(msg)
    return BASELINE_RESULTS[name]
