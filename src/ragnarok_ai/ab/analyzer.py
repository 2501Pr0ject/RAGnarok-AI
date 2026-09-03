"""Statistical analysis of live A/B experiments.

Compares two variants of a running experiment from monitor traces:

- **Success rate** with a two-proportion z-test.
- **Mean latency** with Welch's t-test.

Both p-values use the normal approximation, which is accurate for the
sample sizes the ``min_samples`` guard enforces — this keeps the module
dependency-free (no scipy).

    >>> from ragnarok_ai.ab import ABAnalyzer, Experiment
    >>>
    >>> exp = Experiment(name="reranker-test", variants=["control", "reranker"])
    >>> analyzer = ABAnalyzer(exp)
    >>> report = analyzer.analyze(store.get_traces(since=experiment_start))
    >>> report.winner
    'reranker'
"""

from __future__ import annotations

import math
from statistics import mean, quantiles, variance
from typing import TYPE_CHECKING

from ragnarok_ai.ab.models import (
    EXPERIMENT_KEY,
    VARIANT_KEY,
    ABTestConfig,
    ABTestReport,
    Experiment,
    MetricVerdict,
    VariantStats,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ragnarok_ai.monitor.models import TraceEvent


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _two_sided_p(statistic: float) -> float:
    """Two-sided p-value for a standard-normal test statistic."""
    return 2.0 * (1.0 - _norm_cdf(abs(statistic)))


def _two_proportion_p(successes_a: int, n_a: int, successes_b: int, n_b: int) -> float:
    """Two-proportion z-test p-value (pooled standard error)."""
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    pooled = (successes_a + successes_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0  # both rates identical and degenerate (all 0s or all 1s)
    return _two_sided_p((p_a - p_b) / se)


def _welch_p(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """Welch's t-test p-value (normal approximation for large samples)."""
    var_a = variance(values_a)
    var_b = variance(values_b)
    se = math.sqrt(var_a / len(values_a) + var_b / len(values_b))
    if se == 0:
        return 1.0
    return _two_sided_p((mean(values_a) - mean(values_b)) / se)


def _percentile(values: Sequence[float], pct: int) -> float:
    """Percentile with small-sample fallback."""
    ordered = sorted(values)
    if len(ordered) >= 20:
        return quantiles(ordered, n=100)[pct - 1]
    return ordered[min(len(ordered) - 1, int(len(ordered) * pct / 100))]


def _variant_stats(variant: str, traces: Sequence[TraceEvent]) -> VariantStats:
    """Descriptive statistics for one variant's traces."""
    latencies = [t.total_latency_ms for t in traces]
    return VariantStats(
        variant=variant,
        count=len(traces),
        success_rate=sum(1 for t in traces if t.success) / len(traces),
        latency_mean=mean(latencies),
        latency_p50=_percentile(latencies, 50),
        latency_p95=_percentile(latencies, 95),
    )


class ABAnalyzer:
    """Compare two variants of a live experiment from monitor traces.

    Attributes:
        experiment: The experiment to analyze.
        config: Significance level and sample-size guard.

    Example:
        >>> analyzer = ABAnalyzer(exp)
        >>> report = analyzer.analyze(traces)
        >>> for verdict in report.verdicts:
        ...     print(verdict.metric, verdict.p_value, verdict.winner)
    """

    def __init__(self, experiment: Experiment, config: ABTestConfig | None = None) -> None:
        """Initialize the analyzer.

        Args:
            experiment: The experiment whose traces to analyze.
            config: Optional analysis configuration.
        """
        self.experiment = experiment
        self.config = config or ABTestConfig()

    def split(self, traces: Sequence[TraceEvent]) -> dict[str, list[TraceEvent]]:
        """Group this experiment's traces by variant.

        Traces not tagged with this experiment's name are ignored.

        Args:
            traces: Any window of monitor traces.

        Returns:
            Mapping of variant name to its traces (only known variants).
        """
        groups: dict[str, list[TraceEvent]] = {v: [] for v in self.experiment.variants}
        for trace in traces:
            if trace.metadata.get(EXPERIMENT_KEY) != self.experiment.name:
                continue
            variant = trace.metadata.get(VARIANT_KEY)
            if variant in groups:
                groups[variant].append(trace)
        return groups

    def analyze(
        self,
        traces: Sequence[TraceEvent],
        variant_a: str | None = None,
        variant_b: str | None = None,
    ) -> ABTestReport:
        """Analyze the experiment over a window of traces.

        Args:
            traces: Monitor traces (untagged and other-experiment traces
                are ignored).
            variant_a: First variant to compare (default: first declared).
            variant_b: Second variant to compare (default: second declared).

        Returns:
            An ``ABTestReport``. When either variant has fewer than
            ``config.min_samples`` traces, the report is flagged
            ``insufficient_data`` and carries no verdicts.

        Raises:
            ValueError: If a requested variant is not part of the experiment.
        """
        variant_a = variant_a or self.experiment.variants[0]
        variant_b = variant_b or self.experiment.variants[1]
        for v in (variant_a, variant_b):
            if v not in self.experiment.variants:
                msg = f"Unknown variant {v!r} for experiment {self.experiment.name!r}"
                raise ValueError(msg)

        groups = self.split(traces)
        traces_a = groups[variant_a]
        traces_b = groups[variant_b]

        if len(traces_a) < self.config.min_samples or len(traces_b) < self.config.min_samples:
            stats = {v: _variant_stats(v, g) for v, g in ((variant_a, traces_a), (variant_b, traces_b)) if g}
            return ABTestReport(
                experiment=self.experiment.name,
                variant_a=variant_a,
                variant_b=variant_b,
                stats=stats,
                verdicts=[],
                winner=None,
                insufficient_data=True,
            )

        stats_a = _variant_stats(variant_a, traces_a)
        stats_b = _variant_stats(variant_b, traces_b)
        verdicts = [
            self._success_rate_verdict(traces_a, traces_b, stats_a, stats_b),
            self._latency_verdict(traces_a, traces_b, stats_a, stats_b),
        ]

        return ABTestReport(
            experiment=self.experiment.name,
            variant_a=variant_a,
            variant_b=variant_b,
            stats={variant_a: stats_a, variant_b: stats_b},
            verdicts=verdicts,
            winner=self._overall_winner(verdicts, variant_a, variant_b),
            insufficient_data=False,
        )

    # ── Metric verdicts ──────────────────────────────────────────────────

    def _success_rate_verdict(
        self,
        traces_a: Sequence[TraceEvent],
        traces_b: Sequence[TraceEvent],
        stats_a: VariantStats,
        stats_b: VariantStats,
    ) -> MetricVerdict:
        """Two-proportion z-test on success rate (higher is better)."""
        p_value = _two_proportion_p(
            sum(1 for t in traces_a if t.success),
            len(traces_a),
            sum(1 for t in traces_b if t.success),
            len(traces_b),
        )
        significant = p_value < self.config.alpha
        winner = None
        if significant:
            winner = stats_a.variant if stats_a.success_rate > stats_b.success_rate else stats_b.variant
        return MetricVerdict(
            metric="success_rate",
            a_value=stats_a.success_rate,
            b_value=stats_b.success_rate,
            delta=stats_b.success_rate - stats_a.success_rate,
            p_value=p_value,
            significant=significant,
            winner=winner,
        )

    def _latency_verdict(
        self,
        traces_a: Sequence[TraceEvent],
        traces_b: Sequence[TraceEvent],
        stats_a: VariantStats,
        stats_b: VariantStats,
    ) -> MetricVerdict:
        """Welch's t-test on mean total latency (lower is better)."""
        p_value = _welch_p(
            [t.total_latency_ms for t in traces_a],
            [t.total_latency_ms for t in traces_b],
        )
        significant = p_value < self.config.alpha
        winner = None
        if significant:
            winner = stats_a.variant if stats_a.latency_mean < stats_b.latency_mean else stats_b.variant
        return MetricVerdict(
            metric="latency_mean",
            a_value=stats_a.latency_mean,
            b_value=stats_b.latency_mean,
            delta=stats_b.latency_mean - stats_a.latency_mean,
            p_value=p_value,
            significant=significant,
            winner=winner,
        )

    @staticmethod
    def _overall_winner(verdicts: Sequence[MetricVerdict], variant_a: str, variant_b: str) -> str | None:
        """The variant that wins at least one metric and loses none."""
        for candidate in (variant_a, variant_b):
            wins = sum(1 for v in verdicts if v.winner == candidate)
            losses = sum(1 for v in verdicts if v.winner not in (None, candidate))
            if wins >= 1 and losses == 0:
                return candidate
        return None
