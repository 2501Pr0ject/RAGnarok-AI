"""Comparison module for RAG pipelines.

This module provides functions to compare multiple RAG configurations
on the same testset and identify the best performing configuration.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.evaluate import EvaluationResult, ProgressInfo, evaluate

if TYPE_CHECKING:
    from ragnarok_ai.cache.base import CacheProtocol
    from ragnarok_ai.core.protocols import RAGProtocol
    from ragnarok_ai.core.types import TestSet

# Metrics where lower is better
LOWER_IS_BETTER: set[str] = {"latency_ms", "avg_latency_ms", "total_latency_ms"}


def _compute_testset_hash(testset: TestSet) -> str:
    """Compute deterministic hash of testset for tracking.

    Args:
        testset: TestSet to hash.

    Returns:
        16-character hash string.
    """
    content = "".join(q.text + "".join(q.ground_truth_docs) for q in testset.queries)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ConfigResult:
    """Result for a single config in comparison.

    Attributes:
        config_name: Name of the configuration.
        evaluation: Full evaluation result.
        summary_metrics: Aggregated metrics from evaluation.
        k: Value of k used for @K metrics.
    """

    config_name: str
    evaluation: EvaluationResult
    summary_metrics: dict[str, float]
    k: int


@dataclass
class ComparisonProgress:
    """Progress info for compare() callback.

    Attributes:
        current_config: Name of config being evaluated.
        config_index: 1-based index of current config.
        total_configs: Total number of configs to evaluate.
        evaluation_progress: Progress within current evaluation (if available).
    """

    current_config: str
    config_index: int
    total_configs: int
    evaluation_progress: ProgressInfo | None = None


# Type alias for comparison progress callback
ComparisonProgressCallback = (
    Callable[[ComparisonProgress], None] | Callable[[ComparisonProgress], Awaitable[None]]
)


@dataclass
class ComparisonResult:
    """Result of comparing multiple RAG configurations.

    Attributes:
        testset: The testset used for all evaluations.
        results: Dict mapping config name to ConfigResult.
        warnings: List of warnings (e.g., if k differs between configs).
        timestamp: When the comparison was run.
        testset_hash: Hash of testset for regression tracking.
        baseline_name: Name of config marked as baseline (for regression detection).
    """

    testset: TestSet
    results: dict[str, ConfigResult]
    warnings: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    testset_hash: str = ""
    baseline_name: str | None = None

    def __post_init__(self) -> None:
        """Compute testset_hash if not provided."""
        if not self.testset_hash:
            self.testset_hash = _compute_testset_hash(self.testset)

    def summary(self) -> str:
        """Generate a formatted comparison table.

        Returns:
            Multi-line string with comparison table.
        """
        if not self.results:
            return "No results to compare."

        # Find winner for default metric (ndcg)
        ndcg_winner = self.winner("ndcg")

        # Build header
        headers = ["Config", "Precision", "Recall", "MRR", "NDCG", "Latency"]
        col_widths = [max(14, max(len(name) for name in self.results) + 2)]
        col_widths.extend([10] * (len(headers) - 1))

        # Build separator
        sep_line = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        header_line = "|" + "|".join(f" {h:^{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

        lines = [sep_line, header_line, sep_line]

        # Build data rows
        for config_name, result in self.results.items():
            metrics = result.summary_metrics
            is_winner = config_name == ndcg_winner

            name_display = f"{config_name} *" if is_winner else config_name

            row_values = [
                name_display,
                f"{metrics.get('precision', 0):.2f}",
                f"{metrics.get('recall', 0):.2f}",
                f"{metrics.get('mrr', 0):.2f}",
                f"{metrics.get('ndcg', 0):.2f}",
                f"{result.evaluation.total_latency_ms:.0f}ms",
            ]

            row_line = "|" + "|".join(f" {v:^{col_widths[i]}} " for i, v in enumerate(row_values)) + "|"
            lines.append(row_line)

        lines.append(sep_line)
        lines.append("* = best NDCG score")

        # Add warnings if any
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)

    def winner(self, metric: str = "ndcg") -> str:
        """Return config name with best score for metric.

        Handles lower_is_better metrics (latency_ms, etc.).

        Args:
            metric: Metric name to use for determining winner.

        Returns:
            Name of the winning configuration.

        Raises:
            ValueError: If no results available.
        """
        if not self.results:
            raise ValueError("No results to compare")

        # Handle latency metrics from evaluation (use total_latency_ms)
        if metric == "latency_ms":
            metric = "total_latency_ms"

        scores: dict[str, float] = {}
        for name, result in self.results.items():
            if metric == "total_latency_ms":
                scores[name] = result.evaluation.total_latency_ms
            else:
                scores[name] = result.summary_metrics.get(metric, 0.0)

        if metric in LOWER_IS_BETTER:
            return min(scores, key=lambda x: scores[x])
        return max(scores, key=lambda x: scores[x])

    def rankings(self) -> dict[str, list[str]]:
        """Return rankings per metric.

        Returns:
            Dict mapping metric name to list of config names sorted by score.
        """
        metrics_to_rank = ["precision", "recall", "mrr", "ndcg", "total_latency_ms"]
        rankings: dict[str, list[str]] = {}

        for metric in metrics_to_rank:
            scores: dict[str, float] = {}
            for name, result in self.results.items():
                if metric == "total_latency_ms":
                    scores[name] = result.evaluation.total_latency_ms
                else:
                    scores[name] = result.summary_metrics.get(metric, 0.0)

            # Sort: descending for higher_is_better, ascending for lower_is_better
            reverse = metric not in LOWER_IS_BETTER
            rankings[metric] = sorted(scores.keys(), key=lambda x: scores[x], reverse=reverse)

        return rankings

    def pairwise(self, config_a: str, config_b: str) -> dict[str, dict[str, Any]]:
        """Detailed comparison between two configs.

        Args:
            config_a: First config name.
            config_b: Second config name.

        Returns:
            Dict mapping metric name to comparison details.

        Raises:
            ValueError: If config names not found.
        """
        if config_a not in self.results:
            raise ValueError(f"Config '{config_a}' not found")
        if config_b not in self.results:
            raise ValueError(f"Config '{config_b}' not found")

        result_a = self.results[config_a]
        result_b = self.results[config_b]

        metrics = ["precision", "recall", "mrr", "ndcg"]
        comparisons: dict[str, dict[str, Any]] = {}

        for metric in metrics:
            value_a = result_a.summary_metrics.get(metric, 0.0)
            value_b = result_b.summary_metrics.get(metric, 0.0)
            diff = value_a - value_b
            pct_change = (
                (diff / value_b) * 100
                if value_b != 0
                else (100.0 if diff > 0 else (-100.0 if diff < 0 else 0.0))
            )

            higher_is_better = metric not in LOWER_IS_BETTER
            is_a_better = (diff > 0) if higher_is_better else (diff < 0)

            comparisons[metric] = {
                "value_a": value_a,
                "value_b": value_b,
                "difference": diff,
                "percent_change": pct_change,
                "winner": config_a if is_a_better else (config_b if diff != 0 else "tie"),
            }

        # Add latency comparison
        latency_a = result_a.evaluation.total_latency_ms
        latency_b = result_b.evaluation.total_latency_ms
        latency_diff = latency_a - latency_b

        comparisons["total_latency_ms"] = {
            "value_a": latency_a,
            "value_b": latency_b,
            "difference": latency_diff,
            "percent_change": (latency_diff / latency_b * 100) if latency_b != 0 else 0,
            "winner": config_b if latency_diff > 0 else (config_a if latency_diff < 0 else "tie"),
        }

        return comparisons

    def to_dict(self) -> dict[str, Any]:
        """Export to dict for JSON serialization.

        Returns:
            Serializable dictionary.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "testset_hash": self.testset_hash,
            "testset_name": self.testset.name,
            "num_queries": len(self.testset.queries),
            "baseline_name": self.baseline_name,
            "warnings": self.warnings,
            "results": {
                name: {
                    "config_name": r.config_name,
                    "k": r.k,
                    "summary_metrics": r.summary_metrics,
                    "total_latency_ms": r.evaluation.total_latency_ms,
                }
                for name, r in self.results.items()
            },
            "rankings": self.rankings(),
            "winners": {
                metric: self.winner(metric)
                for metric in ["precision", "recall", "mrr", "ndcg", "total_latency_ms"]
            },
        }

    def export(self, path: str) -> None:
        """Export to HTML or JSON based on file extension.

        Args:
            path: Output file path.

        Raises:
            ValueError: If file extension is not supported.
        """
        path_obj = Path(path)

        if path_obj.suffix == ".json":
            with path_obj.open("w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        elif path_obj.suffix == ".html":
            # Simple HTML export (can be enhanced later)
            html = self._generate_html()
            with path_obj.open("w") as f:
                f.write(html)
        else:
            raise ValueError(f"Unsupported export format: {path_obj.suffix}")

    def _generate_html(self) -> str:
        """Generate simple HTML report.

        Returns:
            HTML string.
        """
        winners = {m: self.winner(m) for m in ["precision", "recall", "mrr", "ndcg", "total_latency_ms"]}

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<meta charset='utf-8'>",
            "<title>RAG Comparison Report</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
            "th { background-color: #f5f5f5; }",
            ".winner { background-color: #d4edda; font-weight: bold; }",
            ".warning { color: #856404; background-color: #fff3cd; padding: 10px; border-radius: 4px; }",
            "</style>",
            "</head><body>",
            "<h1>RAG Configuration Comparison</h1>",
            f"<p>Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>",
            f"<p>Testset: {self.testset.name or 'unnamed'} ({len(self.testset.queries)} queries)</p>",
            f"<p>Testset hash: <code>{self.testset_hash}</code></p>",
        ]

        if self.warnings:
            html_parts.append("<div class='warning'><strong>Warnings:</strong><ul>")
            for w in self.warnings:
                html_parts.append(f"<li>{w}</li>")
            html_parts.append("</ul></div>")

        # Results table
        html_parts.append("<h2>Results</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Config</th><th>Precision</th><th>Recall</th><th>MRR</th><th>NDCG</th><th>Latency</th><th>k</th></tr>")

        for name, result in self.results.items():
            m = result.summary_metrics
            is_ndcg_winner = name == winners["ndcg"]
            row_class = "winner" if is_ndcg_winner else ""

            html_parts.append(f"<tr class='{row_class}'>")
            html_parts.append(f"<td>{name}</td>")
            html_parts.append(f"<td>{m.get('precision', 0):.3f}</td>")
            html_parts.append(f"<td>{m.get('recall', 0):.3f}</td>")
            html_parts.append(f"<td>{m.get('mrr', 0):.3f}</td>")
            html_parts.append(f"<td>{m.get('ndcg', 0):.3f}</td>")
            html_parts.append(f"<td>{result.evaluation.total_latency_ms:.0f}ms</td>")
            html_parts.append(f"<td>{result.k}</td>")
            html_parts.append("</tr>")

        html_parts.append("</table>")

        # Winners
        html_parts.append("<h2>Winners by Metric</h2>")
        html_parts.append("<ul>")
        for metric, winner in winners.items():
            html_parts.append(f"<li><strong>{metric}</strong>: {winner}</li>")
        html_parts.append("</ul>")

        html_parts.append("</body></html>")

        return "\n".join(html_parts)


async def _call_progress(
    callback: ComparisonProgressCallback,
    progress: ComparisonProgress,
) -> None:
    """Call progress callback (sync or async).

    Args:
        callback: Progress callback function.
        progress: Progress information.
    """
    result = callback(progress)
    if asyncio.iscoroutine(result):
        await result


async def compare(
    rag_factory: Callable[[dict[str, Any]], RAGProtocol],
    configs: dict[str, dict[str, Any]],
    testset: TestSet,
    *,
    k: int = 10,
    max_concurrency: int = 1,
    on_progress: ComparisonProgressCallback | None = None,
    cache: CacheProtocol | None = None,
    baseline_name: str | None = None,
) -> ComparisonResult:
    """Compare multiple RAG configurations on the same testset.

    Evaluates each configuration sequentially on the SAME testset to ensure
    fair comparison. Results are keyed by config name for easy access.

    Args:
        rag_factory: Function that creates RAGProtocol from config dict.
        configs: Dict mapping config name to config params.
        testset: TestSet to evaluate all configs against (SAME for all).
        k: Default k for @K metrics (can be overridden per-config via params["k"]).
        max_concurrency: Parallelism within each evaluation.
        on_progress: Progress callback receiving ComparisonProgress.
        cache: Optional cache for results.
        baseline_name: Name of config to mark as baseline (for regression detection).

    Returns:
        ComparisonResult with all evaluations and comparison utilities.

    Raises:
        ValueError: If configs is empty.
        ValueError: If baseline_name not in configs.

    Example:
        >>> def create_rag(config: dict) -> MyRAG:
        ...     return MyRAG(chunk_size=config.get("chunk_size", 512))
        ...
        >>> result = await compare(
        ...     rag_factory=create_rag,
        ...     configs={
        ...         "baseline": {"chunk_size": 512},
        ...         "experiment": {"chunk_size": 256},
        ...     },
        ...     testset=testset,
        ...     baseline_name="baseline",
        ... )
        >>> print(result.winner("precision"))
        "experiment"
    """
    # Validation
    if not configs:
        raise ValueError("configs cannot be empty")

    if baseline_name is not None and baseline_name not in configs:
        raise ValueError(f"baseline_name '{baseline_name}' not in configs")

    warnings: list[str] = []
    results: dict[str, ConfigResult] = {}

    # Check for k differences
    k_values: dict[str, int] = {}
    for name, params in configs.items():
        config_k = params.get("k", k)
        k_values[name] = config_k

    unique_ks = set(k_values.values())
    if len(unique_ks) > 1:
        k_parts = ", ".join(f"{name}={kv}" for name, kv in k_values.items())
        k_warning = f"k differs across configs: {k_parts}"
        warnings.append(k_warning)

    total_configs = len(configs)

    # Evaluate each config sequentially (same testset)
    for idx, (config_name, params) in enumerate(configs.items(), 1):
        # Progress: starting config
        if on_progress:
            progress = ComparisonProgress(
                current_config=config_name,
                config_index=idx,
                total_configs=total_configs,
                evaluation_progress=None,
            )
            await _call_progress(on_progress, progress)

        # Create pipeline
        pipeline = rag_factory(params)

        # Get k for this config
        config_k = params.get("k", k)

        # Evaluate
        evaluation = await evaluate(
            rag_pipeline=pipeline,
            testset=testset,  # SAME testset for all
            k=config_k,
            max_concurrency=max_concurrency,
            cache=cache,
            pipeline_id=config_name,  # Isolate cache
        )

        # Store result
        results[config_name] = ConfigResult(
            config_name=config_name,
            evaluation=evaluation,
            summary_metrics=evaluation.summary(),
            k=config_k,
        )

    return ComparisonResult(
        testset=testset,
        results=results,
        warnings=warnings,
        baseline_name=baseline_name,
    )
