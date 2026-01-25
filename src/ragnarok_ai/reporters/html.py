"""HTML reporter for ragnarok-ai.

This module provides interactive HTML reports for evaluation results,
with drill-down capabilities for debugging RAG failures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragnarok_ai.reporters.console import DEFAULT_THRESHOLDS, Threshold

if TYPE_CHECKING:
    from ragnarok_ai.evaluators.retrieval import RetrievalMetrics


# HTML template with embedded CSS and JavaScript
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGnarok Evaluation Report</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --border-color: #30363d;
            --accent-cyan: #00d9ff;
            --accent-blue: #58a6ff;
            --accent-green: #27ca40;
            --accent-yellow: #f2c94c;
            --accent-red: #f85149;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }}

        header h1 {{
            color: var(--accent-cyan);
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}

        header .timestamp {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .metric-card {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
        }}

        .metric-card.pass {{
            border-color: var(--accent-green);
        }}

        .metric-card.warn {{
            border-color: var(--accent-yellow);
        }}

        .metric-card.fail {{
            border-color: var(--accent-red);
        }}

        .metric-card h3 {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }}

        .metric-card .value {{
            font-size: 2.5rem;
            font-weight: bold;
        }}

        .metric-card.pass .value {{
            color: var(--accent-green);
        }}

        .metric-card.warn .value {{
            color: var(--accent-yellow);
        }}

        .metric-card.fail .value {{
            color: var(--accent-red);
        }}

        .metric-card .status {{
            font-size: 0.8rem;
            margin-top: 0.25rem;
        }}

        .section {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 1.5rem;
            overflow: hidden;
        }}

        .section-header {{
            background-color: var(--bg-tertiary);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .section-header h2 {{
            font-size: 1.1rem;
            color: var(--text-primary);
        }}

        .section-content {{
            padding: 1.5rem;
        }}

        .controls {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }}

        .controls input {{
            flex: 1;
            padding: 0.5rem 1rem;
            background-color: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: var(--text-primary);
            font-size: 0.9rem;
        }}

        .controls input:focus {{
            outline: none;
            border-color: var(--accent-cyan);
        }}

        .controls select {{
            padding: 0.5rem 1rem;
            background-color: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: var(--text-primary);
            font-size: 0.9rem;
            cursor: pointer;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background-color: var(--bg-tertiary);
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            cursor: pointer;
            user-select: none;
        }}

        th:hover {{
            color: var(--accent-cyan);
        }}

        th.sorted-asc::after {{
            content: ' ▲';
        }}

        th.sorted-desc::after {{
            content: ' ▼';
        }}

        tr:hover {{
            background-color: var(--bg-tertiary);
        }}

        tr.expandable {{
            cursor: pointer;
        }}

        .status-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .status-badge.pass {{
            background-color: rgba(39, 202, 64, 0.15);
            color: var(--accent-green);
        }}

        .status-badge.warn {{
            background-color: rgba(242, 201, 76, 0.15);
            color: var(--accent-yellow);
        }}

        .status-badge.fail {{
            background-color: rgba(248, 81, 73, 0.15);
            color: var(--accent-red);
        }}

        .drill-down {{
            display: none;
            background-color: var(--bg-primary);
            padding: 1.5rem;
            border-top: 1px solid var(--border-color);
        }}

        .drill-down.expanded {{
            display: block;
        }}

        .drill-down h4 {{
            color: var(--accent-cyan);
            margin-bottom: 1rem;
            font-size: 0.95rem;
        }}

        .drill-down-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}

        .drill-down-section {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1rem;
        }}

        .drill-down-section h5 {{
            color: var(--text-secondary);
            font-size: 0.8rem;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }}

        .drill-down-section p {{
            color: var(--text-primary);
            font-size: 0.9rem;
        }}

        .chunk {{
            background-color: var(--bg-tertiary);
            border-left: 3px solid var(--accent-blue);
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 0 4px 4px 0;
            font-size: 0.85rem;
        }}

        .chunk .score {{
            color: var(--accent-cyan);
            font-weight: 600;
            font-size: 0.75rem;
            margin-bottom: 0.25rem;
        }}

        .empty-state {{
            text-align: center;
            padding: 3rem;
            color: var(--text-secondary);
        }}

        .empty-state p {{
            font-size: 1.1rem;
        }}

        footer {{
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}

        footer a {{
            color: var(--accent-cyan);
            text-decoration: none;
        }}

        footer a:hover {{
            text-decoration: underline;
        }}

        @media print {{
            body {{
                background-color: white;
                color: black;
            }}

            .controls {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>RAGnarok Evaluation Report</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>

        <section class="summary-grid">
            {metric_cards}
        </section>

        <section class="section">
            <div class="section-header">
                <h2>Evaluation Results</h2>
                <span class="result-count">{result_count} queries</span>
            </div>
            <div class="section-content">
                <div class="controls">
                    <input type="text" id="search" placeholder="Search queries..." />
                    <select id="status-filter">
                        <option value="all">All Status</option>
                        <option value="pass">Pass</option>
                        <option value="warn">Warning</option>
                        <option value="fail">Fail</option>
                    </select>
                </div>
                {results_table}
            </div>
        </section>

        <footer>
            <p>Generated by <a href="https://github.com/2501Pr0ject/RAGnarok-AI" target="_blank">RAGnarok-AI</a></p>
        </footer>
    </div>

    <script>
        // Sorting functionality
        document.querySelectorAll('th[data-sort]').forEach(th => {{
            th.addEventListener('click', () => {{
                const table = th.closest('table');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr.data-row'));
                const column = th.dataset.sort;
                const isNumeric = th.dataset.type === 'number';

                // Toggle sort direction
                const isAsc = th.classList.contains('sorted-asc');
                document.querySelectorAll('th').forEach(h => h.classList.remove('sorted-asc', 'sorted-desc'));
                th.classList.add(isAsc ? 'sorted-desc' : 'sorted-asc');

                rows.sort((a, b) => {{
                    let aVal = a.querySelector(`[data-col="${{column}}"]`)?.textContent || '';
                    let bVal = b.querySelector(`[data-col="${{column}}"]`)?.textContent || '';

                    if (isNumeric) {{
                        aVal = parseFloat(aVal) || 0;
                        bVal = parseFloat(bVal) || 0;
                    }}

                    if (aVal < bVal) return isAsc ? 1 : -1;
                    if (aVal > bVal) return isAsc ? -1 : 1;
                    return 0;
                }});

                rows.forEach(row => {{
                    const drillDown = row.nextElementSibling;
                    tbody.appendChild(row);
                    if (drillDown && drillDown.classList.contains('drill-down-row')) {{
                        tbody.appendChild(drillDown);
                    }}
                }});
            }});
        }});

        // Filter functionality
        const searchInput = document.getElementById('search');
        const statusFilter = document.getElementById('status-filter');

        function filterRows() {{
            const searchTerm = searchInput.value.toLowerCase();
            const statusValue = statusFilter.value;

            document.querySelectorAll('tr.data-row').forEach(row => {{
                const text = row.textContent.toLowerCase();
                const status = row.dataset.status;
                const matchesSearch = text.includes(searchTerm);
                const matchesStatus = statusValue === 'all' || status === statusValue;

                const drillDown = row.nextElementSibling;
                if (matchesSearch && matchesStatus) {{
                    row.style.display = '';
                    if (drillDown && drillDown.classList.contains('drill-down-row')) {{
                        drillDown.style.display = '';
                    }}
                }} else {{
                    row.style.display = 'none';
                    if (drillDown && drillDown.classList.contains('drill-down-row')) {{
                        drillDown.style.display = 'none';
                    }}
                }}
            }});
        }}

        searchInput.addEventListener('input', filterRows);
        statusFilter.addEventListener('change', filterRows);

        // Drill-down functionality
        document.querySelectorAll('tr.expandable').forEach(row => {{
            row.addEventListener('click', (e) => {{
                if (e.target.tagName === 'A') return;
                const drillDown = row.nextElementSibling;
                if (drillDown && drillDown.classList.contains('drill-down-row')) {{
                    drillDown.querySelector('.drill-down').classList.toggle('expanded');
                }}
            }});
        }});
    </script>
</body>
</html>"""

METRIC_CARD_TEMPLATE = """<div class="metric-card {status}">
    <h3>{name}</h3>
    <div class="value">{value}</div>
    <div class="status">{status_text}</div>
</div>"""

RESULTS_TABLE_TEMPLATE = """<table>
    <thead>
        <tr>
            <th data-sort="query">Query</th>
            <th data-sort="precision" data-type="number">Precision</th>
            <th data-sort="recall" data-type="number">Recall</th>
            <th data-sort="mrr" data-type="number">MRR</th>
            <th data-sort="ndcg" data-type="number">NDCG</th>
            <th data-sort="status">Status</th>
        </tr>
    </thead>
    <tbody>
        {rows}
    </tbody>
</table>"""

TABLE_ROW_TEMPLATE = """<tr class="data-row expandable" data-status="{status}">
    <td data-col="query">{query}</td>
    <td data-col="precision">{precision:.2f}</td>
    <td data-col="recall">{recall:.2f}</td>
    <td data-col="mrr">{mrr:.2f}</td>
    <td data-col="ndcg">{ndcg:.2f}</td>
    <td data-col="status"><span class="status-badge {status}">{status}</span></td>
</tr>
<tr class="drill-down-row">
    <td colspan="6">
        <div class="drill-down">
            {drill_down_content}
        </div>
    </td>
</tr>"""

DRILL_DOWN_TEMPLATE = """<div class="drill-down-grid">
    <div class="drill-down-section">
        <h5>Query</h5>
        <p>{query}</p>
    </div>
    <div class="drill-down-section">
        <h5>Metrics</h5>
        <p>Precision: {precision:.4f} | Recall: {recall:.4f} | MRR: {mrr:.4f} | NDCG: {ndcg:.4f}</p>
    </div>
    {chunks_section}
</div>"""

CHUNKS_SECTION_TEMPLATE = """<div class="drill-down-section" style="grid-column: 1 / -1;">
    <h5>Retrieved Chunks</h5>
    {chunks}
</div>"""

CHUNK_TEMPLATE = """<div class="chunk">
    <div class="score">Score: {score:.4f}</div>
    <div class="content">{content}</div>
</div>"""

EMPTY_STATE_TEMPLATE = """<div class="empty-state">
    <p>No evaluation results to display.</p>
</div>"""


class HTMLReporter:
    """Reporter that generates interactive HTML reports.

    Provides standalone HTML reports with:
    - Summary dashboard with overall metrics
    - Sortable and filterable results table
    - Drill-down view for each query

    Attributes:
        thresholds: Threshold configuration for status indicators.

    Example:
        >>> reporter = HTMLReporter()
        >>> reporter.report_to_file(results, "report.html")
    """

    def __init__(
        self,
        thresholds: dict[str, Threshold] | None = None,
    ) -> None:
        """Initialize HTMLReporter.

        Args:
            thresholds: Custom thresholds for status indicators.
        """
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def _get_status(self, metric_name: str, value: float) -> str:
        """Get status string for a metric value.

        Args:
            metric_name: Name of the metric.
            value: The metric value.

        Returns:
            Status string: "pass", "warn", or "fail".
        """
        threshold = self.thresholds.get(metric_name, Threshold())

        # Special case: hallucination (lower is better)
        if metric_name == "hallucination":
            if value <= threshold.good:
                return "pass"
            if value <= threshold.warning:
                return "warn"
            return "fail"

        # Normal case: higher is better
        if value >= threshold.good:
            return "pass"
        if value >= threshold.warning:
            return "warn"
        return "fail"

    def _get_overall_status(self, metrics: RetrievalMetrics) -> str:
        """Get overall status for a set of metrics.

        Args:
            metrics: The retrieval metrics.

        Returns:
            Status string based on worst metric.
        """
        statuses = [
            self._get_status("precision", metrics.precision),
            self._get_status("recall", metrics.recall),
            self._get_status("mrr", metrics.mrr),
            self._get_status("ndcg", metrics.ndcg),
        ]

        if "fail" in statuses:
            return "fail"
        if "warn" in statuses:
            return "warn"
        return "pass"

    def _get_timestamp(self) -> str:
        """Get current timestamp in human-readable format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: The text to escape.

        Returns:
            Escaped text safe for HTML.
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    def _generate_metric_cards(
        self,
        results: list[RetrievalMetrics],
    ) -> str:
        """Generate HTML for metric summary cards.

        Args:
            results: List of retrieval metrics.

        Returns:
            HTML string with metric cards.
        """
        if not results:
            return ""

        n = len(results)
        avg_precision = sum(m.precision for m in results) / n
        avg_recall = sum(m.recall for m in results) / n
        avg_mrr = sum(m.mrr for m in results) / n
        avg_ndcg = sum(m.ndcg for m in results) / n

        metrics = [
            ("Precision", avg_precision, "precision"),
            ("Recall", avg_recall, "recall"),
            ("MRR", avg_mrr, "mrr"),
            ("NDCG", avg_ndcg, "ndcg"),
        ]

        cards = []
        for name, value, metric_key in metrics:
            status = self._get_status(metric_key, value)
            status_text = {"pass": "Good", "warn": "Warning", "fail": "Needs Improvement"}[status]
            cards.append(
                METRIC_CARD_TEMPLATE.format(
                    name=name,
                    value=f"{value:.2f}",
                    status=status,
                    status_text=status_text,
                )
            )

        return "\n".join(cards)

    def _generate_table_row(
        self,
        metrics: RetrievalMetrics,
        query: str,
        chunks: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate HTML for a table row with drill-down.

        Args:
            metrics: The retrieval metrics for this query.
            query: The query text.
            chunks: Optional list of retrieved chunks with scores.

        Returns:
            HTML string with table row and drill-down.
        """
        status = self._get_overall_status(metrics)

        # Generate chunks section if available
        chunks_section = ""
        if chunks:
            chunk_html = []
            for chunk in chunks:
                content = self._escape_html(str(chunk.get("content", "")))
                score = float(chunk.get("score", 0))
                chunk_html.append(CHUNK_TEMPLATE.format(score=score, content=content))
            chunks_section = CHUNKS_SECTION_TEMPLATE.format(chunks="\n".join(chunk_html))

        drill_down = DRILL_DOWN_TEMPLATE.format(
            query=self._escape_html(query),
            precision=metrics.precision,
            recall=metrics.recall,
            mrr=metrics.mrr,
            ndcg=metrics.ndcg,
            chunks_section=chunks_section,
        )

        return TABLE_ROW_TEMPLATE.format(
            query=self._escape_html(query),
            precision=metrics.precision,
            recall=metrics.recall,
            mrr=metrics.mrr,
            ndcg=metrics.ndcg,
            status=status,
            drill_down_content=drill_down,
        )

    def _generate_results_table(
        self,
        results: list[tuple[str, RetrievalMetrics, list[dict[str, Any]] | None]],
    ) -> str:
        """Generate HTML for the results table.

        Args:
            results: List of (query, metrics, chunks) tuples.

        Returns:
            HTML string with results table.
        """
        if not results:
            return EMPTY_STATE_TEMPLATE

        rows = []
        for query, metrics, chunks in results:
            rows.append(self._generate_table_row(metrics, query, chunks))

        return RESULTS_TABLE_TEMPLATE.format(rows="\n".join(rows))

    def report(
        self,
        results: list[tuple[str, RetrievalMetrics, list[dict[str, Any]] | None]],
        metadata: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> str:
        """Generate HTML report for evaluation results.

        Args:
            results: List of (query, metrics, chunks) tuples.
            metadata: Optional metadata (unused, for API consistency).

        Returns:
            HTML string of the report.

        Example:
            >>> html = reporter.report([(query, metrics, chunks), ...])
        """
        # Extract just the metrics for summary calculations
        metrics_list = [m for _, m, _ in results] if results else []

        return HTML_TEMPLATE.format(
            timestamp=self._get_timestamp(),
            metric_cards=self._generate_metric_cards(metrics_list),
            result_count=len(results),
            results_table=self._generate_results_table(results),
        )

    def report_to_file(
        self,
        results: list[tuple[str, RetrievalMetrics, list[dict[str, Any]] | None]],
        path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write HTML report to a file.

        Args:
            results: List of (query, metrics, chunks) tuples.
            path: Path to the output file.
            metadata: Optional metadata (unused, for API consistency).

        Example:
            >>> reporter.report_to_file(results, Path("report.html"))
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        html = self.report(results, metadata)
        path.write_text(html, encoding="utf-8")

    def report_summary(
        self,
        results: list[RetrievalMetrics],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate HTML summary report for metrics only.

        This is a simpler report without query-level details.

        Args:
            results: List of retrieval metrics.
            metadata: Optional metadata (unused, for API consistency).

        Returns:
            HTML string of the summary report.

        Example:
            >>> html = reporter.report_summary(metrics_list)
        """
        # Convert to the full format with empty queries and no chunks
        full_results: list[tuple[str, RetrievalMetrics, list[dict[str, Any]] | None]] = [
            (f"Query {i + 1}", m, None) for i, m in enumerate(results)
        ]
        return self.report(full_results, metadata)

    def report_summary_to_file(
        self,
        results: list[RetrievalMetrics],
        path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write HTML summary report to a file.

        Args:
            results: List of retrieval metrics.
            path: Path to the output file.
            metadata: Optional metadata (unused, for API consistency).

        Example:
            >>> reporter.report_summary_to_file(results, Path("summary.html"))
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        html = self.report_summary(results, metadata)
        path.write_text(html, encoding="utf-8")
