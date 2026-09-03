"""Data models for live A/B testing of RAG configurations.

An ``Experiment`` deterministically splits production traffic between
named variants (hash-based bucketing: the same key always lands on the
same variant, with no coordination or storage). Variants are tagged into
monitor trace metadata, and ``ABAnalyzer`` later compares the two
populations with statistical significance tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from ragnarok_ai.monitor.client import TraceContext

# Metadata keys used to tag traces with experiment assignments
EXPERIMENT_KEY = "experiment"
VARIANT_KEY = "variant"


class Experiment(BaseModel):
    """A live A/B experiment splitting traffic between variants.

    Assignment is deterministic: ``assign(key)`` hashes the key with the
    experiment name and salt, so a given user/session/query key always
    gets the same variant, across processes and restarts, without any
    shared state.

    Attributes:
        name: Experiment identifier (recorded in trace metadata).
        variants: Variant names, e.g. ``["control", "reranker"]``.
        weights: Traffic split, same length as variants, sums to 1.0.
            Defaults to an even split.
        salt: Extra hash salt. Change it to reshuffle assignments.

    Example:
        >>> exp = Experiment(name="reranker-test", variants=["control", "reranker"])
        >>> exp.assign("user-42")
        'control'
        >>> exp.assign("user-42")  # always the same
        'control'
    """

    name: str
    variants: list[str] = Field(default_factory=lambda: ["A", "B"])
    weights: list[float] | None = None
    salt: str = ""

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate(self) -> Experiment:
        if len(self.variants) < 2:
            msg = "An experiment needs at least two variants"
            raise ValueError(msg)
        if len(set(self.variants)) != len(self.variants):
            msg = "Variant names must be unique"
            raise ValueError(msg)
        if self.weights is not None:
            if len(self.weights) != len(self.variants):
                msg = "weights must have the same length as variants"
                raise ValueError(msg)
            if any(w <= 0 for w in self.weights):
                msg = "weights must be positive"
                raise ValueError(msg)
            if abs(sum(self.weights) - 1.0) > 1e-6:
                msg = "weights must sum to 1.0"
                raise ValueError(msg)
        return self

    def assign(self, key: str) -> str:
        """Deterministically assign *key* to a variant.

        Args:
            key: Stable identifier for the traffic unit — a user id,
                session id, or query hash. Whatever entity should
                consistently see the same configuration.

        Returns:
            The assigned variant name.
        """
        digest = hashlib.sha256(f"{self.name}:{self.salt}:{key}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64  # uniform [0, 1)

        weights = self.weights or [1.0 / len(self.variants)] * len(self.variants)
        cumulative = 0.0
        for variant, weight in zip(self.variants, weights, strict=True):
            cumulative += weight
            if bucket < cumulative:
                return variant
        return self.variants[-1]  # float rounding edge

    def tag(self, trace: TraceContext, key: str) -> str:
        """Assign *key* to a variant and tag the trace with it.

        Convenience for instrumented pipelines:

            >>> with client.trace(query) as trace:
            ...     variant = exp.tag(trace, user_id)
            ...     rag = pipelines[variant]

        Args:
            trace: The active monitor trace context.
            key: Stable identifier for the traffic unit.

        Returns:
            The assigned variant name.
        """
        variant = self.assign(key)
        trace.add_metadata(EXPERIMENT_KEY, self.name)
        trace.add_metadata(VARIANT_KEY, variant)
        return variant


@dataclass
class ABTestConfig:
    """Analysis configuration.

    Attributes:
        alpha: Significance level for the tests (default 0.05).
        min_samples: Minimum traces per variant for analysis; below it
            the report is flagged ``insufficient_data``. The p-values use
            a normal approximation, which also needs reasonably sized
            samples to be trustworthy.
    """

    alpha: float = 0.05
    min_samples: int = 50


@dataclass(frozen=True)
class VariantStats:
    """Descriptive statistics for one variant's traffic.

    Attributes:
        variant: Variant name.
        count: Number of traces.
        success_rate: Fraction of successful requests.
        latency_mean: Mean total latency (ms).
        latency_p50: Median total latency (ms).
        latency_p95: 95th percentile total latency (ms).
    """

    variant: str
    count: int
    success_rate: float
    latency_mean: float
    latency_p50: float
    latency_p95: float


@dataclass(frozen=True)
class MetricVerdict:
    """Statistical comparison of one metric between two variants.

    Attributes:
        metric: "success_rate" or "latency_mean".
        a_value: Value for the first variant.
        b_value: Value for the second variant.
        delta: b_value - a_value.
        p_value: Two-sided p-value (normal approximation).
        significant: True when p_value < alpha.
        winner: Name of the better variant when significant, else None.
            Higher is better for success_rate, lower for latency.
    """

    metric: str
    a_value: float
    b_value: float
    delta: float
    p_value: float
    significant: bool
    winner: str | None


@dataclass
class ABTestReport:
    """Result of analyzing an A/B experiment.

    Attributes:
        experiment: Experiment name.
        variant_a: First compared variant name.
        variant_b: Second compared variant name.
        stats: Per-variant descriptive statistics.
        verdicts: Per-metric statistical comparisons.
        winner: Overall winner — the variant that is significantly
            better on at least one metric and significantly worse on
            none. None when there is no clear winner.
        insufficient_data: True when either variant had fewer than
            ``min_samples`` traces (verdicts are then empty).
        timestamp: When the analysis ran.
    """

    experiment: str
    variant_a: str
    variant_b: str
    stats: dict[str, VariantStats]
    verdicts: list[MetricVerdict]
    winner: str | None
    insufficient_data: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "experiment": self.experiment,
            "variant_a": self.variant_a,
            "variant_b": self.variant_b,
            "winner": self.winner,
            "insufficient_data": self.insufficient_data,
            "timestamp": self.timestamp.isoformat(),
            "stats": {
                name: {
                    "count": s.count,
                    "success_rate": s.success_rate,
                    "latency_mean": s.latency_mean,
                    "latency_p50": s.latency_p50,
                    "latency_p95": s.latency_p95,
                }
                for name, s in self.stats.items()
            },
            "verdicts": [
                {
                    "metric": v.metric,
                    "a_value": v.a_value,
                    "b_value": v.b_value,
                    "delta": v.delta,
                    "p_value": v.p_value,
                    "significant": v.significant,
                    "winner": v.winner,
                }
                for v in self.verdicts
            ],
        }
