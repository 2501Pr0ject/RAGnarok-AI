"""Console reporter for ragnarok-ai.

This module provides terminal output for evaluation results,
with colored tables and status indicators.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from ragnarok_ai.evaluators.retrieval import RetrievalMetrics


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Status colors
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"

    # Background
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_RED = "\033[41m"


@dataclass
class Threshold:
    """Threshold configuration for metric status indicators.

    Attributes:
        good: Minimum value for good status (green).
        warning: Minimum value for warning status (yellow).
            Below this is considered bad (red).
    """

    good: float = 0.8
    warning: float = 0.6


# Default thresholds for each metric
DEFAULT_THRESHOLDS: dict[str, Threshold] = {
    "precision": Threshold(good=0.8, warning=0.6),
    "recall": Threshold(good=0.8, warning=0.6),
    "mrr": Threshold(good=0.8, warning=0.5),
    "ndcg": Threshold(good=0.8, warning=0.6),
    "faithfulness": Threshold(good=0.8, warning=0.6),
    "relevance": Threshold(good=0.8, warning=0.6),
    "hallucination": Threshold(good=0.2, warning=0.4),  # Lower is better
}


class ConsoleReporter:
    """Reporter that outputs evaluation results to the terminal.

    Provides formatted tables with color-coded status indicators
    based on configurable thresholds.

    Attributes:
        use_colors: Whether to use ANSI colors in output.
        thresholds: Threshold configuration for status indicators.
        output: Output stream (defaults to stdout).

    Example:
        >>> reporter = ConsoleReporter()
        >>> reporter.report_retrieval_metrics(metrics)
        ┌─────────────────┬───────┬────────┐
        │ Metric          │ Score │ Status │
        ├─────────────────┼───────┼────────┤
        │ Precision@10    │ 0.82  │ ✅      │
        │ Recall@10       │ 0.74  │ ⚠️      │
        │ MRR             │ 0.91  │ ✅      │
        │ NDCG@10         │ 0.85  │ ✅      │
        └─────────────────┴───────┴────────┘
    """

    def __init__(
        self,
        use_colors: bool = True,
        thresholds: dict[str, Threshold] | None = None,
        output: TextIO | None = None,
    ) -> None:
        """Initialize ConsoleReporter.

        Args:
            use_colors: Whether to use ANSI colors. Defaults to True.
            thresholds: Custom thresholds for status indicators.
            output: Output stream. Defaults to sys.stdout.
        """
        self.use_colors = use_colors and _supports_color(output or sys.stdout)
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.output = output or sys.stdout

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}{Colors.RESET}"
        return text

    def _get_status(self, metric_name: str, value: float) -> tuple[str, str]:
        """Get status indicator and color for a metric value.

        Args:
            metric_name: Name of the metric.
            value: The metric value.

        Returns:
            Tuple of (status_emoji, color_code).
        """
        threshold = self.thresholds.get(metric_name, Threshold())

        # Special case: hallucination (lower is better)
        if metric_name == "hallucination":
            if value <= threshold.good:
                return ("✅", Colors.GREEN)
            if value <= threshold.warning:
                return ("⚠️ ", Colors.YELLOW)
            return ("❌", Colors.RED)

        # Normal case: higher is better
        if value >= threshold.good:
            return ("✅", Colors.GREEN)
        if value >= threshold.warning:
            return ("⚠️ ", Colors.YELLOW)
        return ("❌", Colors.RED)

    def _print(self, text: str = "") -> None:
        """Print text to output stream."""
        print(text, file=self.output)

    def report_retrieval_metrics(
        self,
        metrics: RetrievalMetrics,
        title: str | None = None,
    ) -> None:
        """Report retrieval evaluation metrics.

        Args:
            metrics: The retrieval metrics to report.
            title: Optional title for the report section.

        Example:
            >>> reporter.report_retrieval_metrics(metrics, title="Query 1")
        """
        if title:
            self._print()
            self._print(self._color(f"  {title}", Colors.BOLD))

        k = metrics.k

        # Table data
        rows = [
            (f"Precision@{k}", metrics.precision, "precision"),
            (f"Recall@{k}", metrics.recall, "recall"),
            ("MRR", metrics.mrr, "mrr"),
            (f"NDCG@{k}", metrics.ndcg, "ndcg"),
        ]

        self._print_metrics_table(rows)

    def report_summary(
        self,
        metrics_list: list[RetrievalMetrics],
        title: str = "Evaluation Summary",
    ) -> None:
        """Report aggregated summary of multiple evaluations.

        Args:
            metrics_list: List of retrieval metrics from multiple queries.
            title: Title for the summary section.
        """
        if not metrics_list:
            self._print("No metrics to report.")
            return

        # Calculate averages
        n = len(metrics_list)
        avg_precision = sum(m.precision for m in metrics_list) / n
        avg_recall = sum(m.recall for m in metrics_list) / n
        avg_mrr = sum(m.mrr for m in metrics_list) / n
        avg_ndcg = sum(m.ndcg for m in metrics_list) / n
        k = metrics_list[0].k

        self._print()
        self._print(self._color(f"  {title}", Colors.BOLD))
        self._print(self._color(f"  ({n} queries evaluated)", Colors.DIM))

        rows = [
            (f"Avg Precision@{k}", avg_precision, "precision"),
            (f"Avg Recall@{k}", avg_recall, "recall"),
            ("Avg MRR", avg_mrr, "mrr"),
            (f"Avg NDCG@{k}", avg_ndcg, "ndcg"),
        ]

        self._print_metrics_table(rows)

    def _print_metrics_table(
        self,
        rows: list[tuple[str, float, str]],
    ) -> None:
        """Print a metrics table.

        Args:
            rows: List of (label, value, metric_name) tuples.
        """
        # Column widths
        col1_width = 17
        col2_width = 7
        col3_width = 8

        # Box drawing characters
        top_left = "┌"
        top_right = "┐"
        bottom_left = "└"
        bottom_right = "┘"
        horizontal = "─"
        vertical = "│"
        cross = "┼"
        t_down = "┬"
        t_up = "┴"
        t_right = "├"
        t_left = "┤"

        # Top border
        self._print(
            f"  {top_left}{horizontal * col1_width}{t_down}"
            f"{horizontal * col2_width}{t_down}{horizontal * col3_width}{top_right}"
        )

        # Header
        header = (
            f"  {vertical}"
            f"{self._color(' Metric', Colors.BOLD):<{col1_width + (len(Colors.BOLD) + len(Colors.RESET) if self.use_colors else 0)}}"
            f"{vertical}"
            f"{self._color(' Score', Colors.BOLD):<{col2_width + (len(Colors.BOLD) + len(Colors.RESET) if self.use_colors else 0)}}"
            f"{vertical}"
            f"{self._color(' Status', Colors.BOLD):<{col3_width + (len(Colors.BOLD) + len(Colors.RESET) if self.use_colors else 0)}}"
            f"{vertical}"
        )
        self._print(header)

        # Header separator
        self._print(
            f"  {t_right}{horizontal * col1_width}{cross}"
            f"{horizontal * col2_width}{cross}{horizontal * col3_width}{t_left}"
        )

        # Data rows
        for label, value, metric_name in rows:
            status_emoji, status_color = self._get_status(metric_name, value)
            value_str = f"{value:.2f}"
            colored_value = self._color(value_str, status_color)

            # Calculate padding accounting for color codes
            color_len = len(status_color) + len(Colors.RESET) if self.use_colors else 0

            row = (
                f"  {vertical} {label:<{col1_width - 2}} "
                f"{vertical} {colored_value:<{col2_width - 2 + color_len}}"
                f"{vertical} {status_emoji:<{col3_width - 2}}"
                f"{vertical}"
            )
            self._print(row)

        # Bottom border
        self._print(
            f"  {bottom_left}{horizontal * col1_width}{t_up}"
            f"{horizontal * col2_width}{t_up}{horizontal * col3_width}{bottom_right}"
        )
        self._print()

    def print_header(self, text: str) -> None:
        """Print a section header.

        Args:
            text: Header text to display.
        """
        self._print()
        self._print(self._color(f"{'=' * 50}", Colors.DIM))
        self._print(self._color(f"  {text}", Colors.BOLD + Colors.CYAN))
        self._print(self._color(f"{'=' * 50}", Colors.DIM))

    def print_success(self, text: str) -> None:
        """Print a success message.

        Args:
            text: Message to display.
        """
        self._print(self._color(f"  ✅ {text}", Colors.GREEN))

    def print_warning(self, text: str) -> None:
        """Print a warning message.

        Args:
            text: Message to display.
        """
        self._print(self._color(f"  ⚠️  {text}", Colors.YELLOW))

    def print_error(self, text: str) -> None:
        """Print an error message.

        Args:
            text: Message to display.
        """
        self._print(self._color(f"  ❌ {text}", Colors.RED))

    def print_info(self, text: str) -> None:
        """Print an info message.

        Args:
            text: Message to display.
        """
        self._print(self._color(f"  [i] {text}", Colors.BLUE))


def _supports_color(stream: TextIO) -> bool:
    """Check if the output stream supports ANSI colors.

    Args:
        stream: Output stream to check.

    Returns:
        True if colors are supported, False otherwise.
    """
    # Check if stream is a TTY
    if not hasattr(stream, "isatty"):
        return False
    if not stream.isatty():
        return False

    # Check for common environment variables that disable color
    import os

    if os.environ.get("NO_COLOR"):
        return False

    return os.environ.get("TERM") != "dumb"
