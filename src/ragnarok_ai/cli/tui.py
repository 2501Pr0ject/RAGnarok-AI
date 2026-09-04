"""Interactive TUI for browsing evaluation results.

``ragnarok view results.json`` opens the per-query results of an
evaluation (produced by ``ragnarok evaluate --output``) in an
interactive terminal app: sortable table, keyboard navigation, and a
detail panel for the selected query.

Requires the optional ``tui`` extra: ``pip install ragnarok-ai[tui]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Static

# Precision bands matching the CLI status glyphs
_PASS_PRECISION = 0.7
_WARN_PRECISION = 0.4


def load_results(path: str | Path) -> dict[str, Any]:
    """Load an evaluation results file and validate its shape.

    Args:
        path: Path to a JSON file written by ``ragnarok evaluate --output``.

    Returns:
        The parsed results document.

    Raises:
        ValueError: If the file is missing, invalid JSON, or has no
            per-query rows (older result files predate them).
    """
    p = Path(path)
    if not p.exists():
        msg = f"Results file not found: {path}"
        raise ValueError(msg)
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in {path}: {e}"
        raise ValueError(msg) from e
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list) or not data["queries"]:
        msg = (
            f"{path} has no per-query results to browse. "
            "Re-run: ragnarok evaluate --testset ... --pipeline ... --output results.json"
        )
        raise ValueError(msg)
    return data


def _verdict(precision: float) -> str:
    if precision >= _PASS_PRECISION:
        return "PASS"
    if precision >= _WARN_PRECISION:
        return "WARN"
    return "FAIL"


class ResultsApp(App[None]):
    """Browse evaluation results: table of queries + detail panel."""

    TITLE = "RAGnarok Results"

    CSS = """
    #summary {
        height: auto;
        padding: 0 1;
        background: $boost;
        color: $text;
    }
    #table {
        height: 1fr;
    }
    #detail {
        height: auto;
        max-height: 40%;
        padding: 0 1;
        border-top: solid $accent;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("f", "filter_failing", "Failing only"),
        Binding("a", "show_all", "All"),
    ]

    def __init__(self, results: dict[str, Any]) -> None:
        super().__init__()
        self.results = results
        self.rows: list[dict[str, Any]] = list(results["queries"])
        self.showing: list[dict[str, Any]] = self.rows

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(self._summary_text(), id="summary")
            yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
            yield Static("", id="detail")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("#", "Status", "Precision", "Recall", "MRR", "NDCG", "Query")
        self._fill_table(self.rows)
        table.focus()

    # ── Table handling ───────────────────────────────────────────────────

    def _fill_table(self, rows: list[dict[str, Any]]) -> None:
        table = self.query_one(DataTable)
        table.clear()
        self.showing = rows
        for i, row in enumerate(rows):
            table.add_row(
                str(i + 1),
                _verdict(row["precision"]),
                f"{row['precision']:.2f}",
                f"{row['recall']:.2f}",
                f"{row['mrr']:.2f}",
                f"{row['ndcg']:.2f}",
                row["query"][:80],
                key=str(i),
            )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        row = self.showing[int(event.row_key.value)]
        detail = self.query_one("#detail", Static)
        answer = (row.get("answer") or "").strip()
        detail.update(
            f"[bold]Query:[/bold] {row['query']}\n"
            f"[bold]Answer:[/bold] {answer[:500] if answer else '(empty)'}\n"
            f"[bold]Metrics:[/bold] P={row['precision']:.2f} R={row['recall']:.2f} "
            f"MRR={row['mrr']:.2f} NDCG={row['ndcg']:.2f}"
        )

    # ── Actions ──────────────────────────────────────────────────────────

    def action_filter_failing(self) -> None:
        """Show only queries below the pass band."""
        self._fill_table([r for r in self.rows if r["precision"] < _PASS_PRECISION])

    def action_show_all(self) -> None:
        """Show every query."""
        self._fill_table(self.rows)

    # ── Rendering ────────────────────────────────────────────────────────

    def _summary_text(self) -> str:
        metrics = self.results.get("metrics", {})
        parts = [f"{k}: {v}" for k, v in metrics.items()]
        testset = self.results.get("testset", "?")
        n = self.results.get("queries_evaluated", len(self.rows))
        return f"[bold]{testset}[/bold] · {n} queries · " + " · ".join(parts)


def run_viewer(path: str | Path) -> None:
    """Load a results file and run the interactive viewer."""
    ResultsApp(load_results(path)).run()
