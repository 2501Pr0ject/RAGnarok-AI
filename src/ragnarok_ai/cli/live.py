"""Live terminal display for streaming evaluations.

Long local evaluations should not be a black box. This renders a
continuously updating panel — progress, running metric averages, the
latest per-query results — on top of ``evaluate_stream``, so you see
what is happening while it happens.

Falls back to plain line-by-line output when the terminal is not
interactive (CI logs, redirected output).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# Per-query precision bands for the status glyph
_PASS_PRECISION = 0.7
_WARN_PRECISION = 0.4

# Number of most-recent per-query rows kept visible in the live panel
_TAIL_SIZE = 8


@dataclass
class RunningStats:
    """Streaming averages over per-query retrieval metrics."""

    count: int = 0
    precision: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    passed: int = 0
    warned: int = 0
    failed: int = 0

    def add(self, precision: float, recall: float, mrr: float, ndcg: float) -> None:
        """Fold one query's metrics into the running averages."""
        self.count += 1
        n = self.count
        self.precision += (precision - self.precision) / n
        self.recall += (recall - self.recall) / n
        self.mrr += (mrr - self.mrr) / n
        self.ndcg += (ndcg - self.ndcg) / n
        if precision >= _PASS_PRECISION:
            self.passed += 1
        elif precision >= _WARN_PRECISION:
            self.warned += 1
        else:
            self.failed += 1

    @property
    def average(self) -> float:
        """Mean of the four running metric averages."""
        return (self.precision + self.recall + self.mrr + self.ndcg) / 4


@dataclass
class _QueryRow:
    index: int
    text: str
    precision: float
    recall: float
    mrr: float


def _status_glyph(precision: float) -> str:
    if precision >= _PASS_PRECISION:
        return "[green]✓[/green]"
    if precision >= _WARN_PRECISION:
        return "[yellow]○[/yellow]"
    return "[red]✗[/red]"


class LiveEvaluationDisplay:
    """Rich Live panel for a streaming evaluation.

    Usage:
        display = LiveEvaluationDisplay(title="legal-rag-v3", total=500)
        with display:
            for query, metrics, _answer in stream:
                display.record(query.text, metrics)
        stats = display.stats

    When the output is not a terminal (CI, pipes), each query prints as
    one plain line instead — same information, log-friendly.
    """

    def __init__(self, title: str, total: int, *, console: Console | None = None) -> None:
        """Initialize the display.

        Args:
            title: Test set / run name shown in the panel header.
            total: Number of queries to evaluate.
            console: Override console (mainly for tests).
        """
        self.title = title
        self.total = total
        self.stats = RunningStats()
        self.console = console or Console()
        self.interactive = self.console.is_terminal
        self._tail: list[_QueryRow] = []
        self._start = time.perf_counter()
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        self._task = self._progress.add_task("Evaluating", total=total)
        self._live: Live | None = None

    def __enter__(self) -> LiveEvaluationDisplay:
        if self.interactive:
            self._live = Live(self._render(), console=self.console, refresh_per_second=8)
            self._live.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(None, None, None)
            self._live = None

    def record(self, query_text: str, precision: float, recall: float, mrr: float, ndcg: float) -> None:
        """Record one evaluated query and refresh the display."""
        self.stats.add(precision, recall, mrr, ndcg)
        self._tail.append(
            _QueryRow(index=self.stats.count, text=query_text, precision=precision, recall=recall, mrr=mrr)
        )
        del self._tail[:-_TAIL_SIZE]
        self._progress.update(self._task, completed=self.stats.count)

        if self._live is not None:
            self._live.update(self._render())
        else:
            glyph = "✓" if precision >= _PASS_PRECISION else ("○" if precision >= _WARN_PRECISION else "✗")
            self.console.print(
                f"[{glyph}] {self.stats.count:4d}/{self.total}  "
                f"P={precision:.2f} R={recall:.2f} MRR={mrr:.2f}  "
                f"avg={self.stats.average:.3f}  {query_text[:60]}",
                highlight=False,
            )

    # ── Rendering ────────────────────────────────────────────────────────

    def _render(self) -> Panel:
        metrics = Table.grid(padding=(0, 3))
        metrics.add_column(style="bold")
        metrics.add_column(justify="right")
        metrics.add_row("Precision@K", f"{self.stats.precision:.3f}")
        metrics.add_row("Recall@K", f"{self.stats.recall:.3f}")
        metrics.add_row("MRR", f"{self.stats.mrr:.3f}")
        metrics.add_row("NDCG@K", f"{self.stats.ndcg:.3f}")
        metrics.add_row("Average", f"[bold]{self.stats.average:.3f}[/bold]")

        tail = Table.grid(padding=(0, 1))
        tail.add_column(width=2)
        tail.add_column(width=5, justify="right")
        tail.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
        tail.add_column(justify="right")
        for row in self._tail:
            tail.add_row(
                _status_glyph(row.precision),
                f"{row.index}",
                Text(row.text, style="dim"),
                f"P={row.precision:.2f}",
            )

        counts = Text.assemble(
            (f"✓ {self.stats.passed}  ", "green"),
            (f"○ {self.stats.warned}  ", "yellow"),
            (f"✗ {self.stats.failed}", "red"),
        )

        return Panel(
            Group(self._progress, "", metrics, "", tail, "", counts),
            title=f"RAGnarok · {self.title}",
            border_style="cyan",
        )
