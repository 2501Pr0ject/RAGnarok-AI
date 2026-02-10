"""Rich display functions for Jupyter notebooks.

This module provides HTML-rich display functions for RAGnarok-AI results
in Jupyter notebooks and IPython environments.

Theme: Terminal dark mode
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragnarok_ai.core.evaluate import EvaluationResult
    from ragnarok_ai.cost.tracker import CostSummary


# Terminal theme colors
_BG_DARK = "#0d1117"
_BG_LIGHTER = "#161b22"
_BG_HEADER = "#1f2428"
_TEXT_PRIMARY = "#e6edf3"
_TEXT_SECONDARY = "#8b949e"
_TEXT_MUTED = "#6e7681"
_GREEN = "#3fb950"
_YELLOW = "#d29922"
_RED = "#f85149"
_CYAN = "#58a6ff"
_PURPLE = "#a371f7"
_BORDER = "#30363d"


def _in_notebook() -> bool:
    """Check if we're running in a Jupyter notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        if shell_name == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        if shell_name == "TerminalInteractiveShell":
            return False  # Terminal IPython
        return False
    except (ImportError, NameError):
        return False


def _display_html(html: str) -> None:
    """Display HTML in notebook or print fallback."""
    if _in_notebook():
        from IPython.display import HTML
        from IPython.display import display as ipy_display

        ipy_display(HTML(html))
    else:
        print("[HTML output available in Jupyter notebook]")


def _format_number(value: float, decimals: int = 2) -> str:
    """Format a number for display."""
    if value == 0:
        return "0"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.{decimals}f}"


def _progress_bar(value: float, max_value: float = 1.0) -> str:
    """Generate a terminal-style progress bar."""
    percentage = min(100, max(0, (value / max_value) * 100))

    # Color based on value
    if percentage < 50:
        color = _RED
    elif percentage < 75:
        color = _YELLOW
    else:
        color = _GREEN

    filled = int(percentage / 100 * 20)
    empty = 20 - filled
    bar = "█" * filled + "░" * empty

    return f"""
    <span style="font-family: monospace; color: {color};">[{bar}]</span>
    <span style="font-family: monospace; color: {_TEXT_PRIMARY}; margin-left: 8px;">{_format_number(value)}</span>
    """


def display_metrics(result: EvaluationResult) -> None:
    """Display evaluation metrics with progress bars.

    Args:
        result: EvaluationResult from evaluate().

    Example:
        >>> from ragnarok_ai.notebook import display_metrics
        >>> display_metrics(results)
    """
    summary = result.summary()
    if not summary:
        _display_html(f'<p style="color: {_TEXT_MUTED}; font-family: monospace;">No metrics available.</p>')
        return

    html = f"""
    <div style="font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; padding: 16px; background: {_BG_DARK}; border-radius: 6px; border: 1px solid {_BORDER}; max-width: 500px;">
        <div style="color: {_CYAN}; font-size: 14px; margin-bottom: 16px; border-bottom: 1px solid {_BORDER}; padding-bottom: 8px;">
            $ ragnarok metrics
        </div>
        <table style="width: 100%; border-collapse: collapse;">
    """

    metrics = [
        ("precision", summary.get("precision", 0)),
        ("recall", summary.get("recall", 0)),
        ("mrr", summary.get("mrr", 0)),
        ("ndcg", summary.get("ndcg", 0)),
    ]

    for name, value in metrics:
        html += f"""
        <tr>
            <td style="padding: 6px 0; color: {_TEXT_SECONDARY}; width: 100px;">{name}</td>
            <td style="padding: 6px 0;">{_progress_bar(value)}</td>
        </tr>
        """

    num_queries = int(summary.get("num_queries", 0))
    html += f"""
        </table>
        <div style="margin-top: 12px; color: {_TEXT_MUTED}; font-size: 12px;">
            # {num_queries} queries evaluated
        </div>
    </div>
    """

    _display_html(html)


def display_cost(result: EvaluationResult) -> None:
    """Display cost summary with styled table.

    Args:
        result: EvaluationResult with cost tracking enabled.

    Example:
        >>> results = await evaluate(rag, testset, track_cost=True)
        >>> from ragnarok_ai.notebook import display_cost
        >>> display_cost(results)
    """
    if result.cost is None:
        _display_html(f"""
        <div style="font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; padding: 16px; background: {_BG_DARK}; border-radius: 6px; border: 1px solid {_YELLOW};">
            <span style="color: {_YELLOW};">WARNING:</span>
            <span style="color: {_TEXT_PRIMARY};"> Cost tracking not enabled. Use track_cost=True</span>
        </div>
        """)
        return

    _display_cost_summary(result.cost)


def _display_cost_summary(cost: CostSummary) -> None:
    """Display a CostSummary object."""
    html = f"""
    <div style="font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; padding: 16px; background: {_BG_DARK}; border-radius: 6px; border: 1px solid {_BORDER}; max-width: 500px;">
        <div style="color: {_CYAN}; font-size: 14px; margin-bottom: 16px; border-bottom: 1px solid {_BORDER}; padding-bottom: 8px;">
            $ ragnarok cost
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
            <thead>
                <tr style="border-bottom: 1px solid {_BORDER};">
                    <th style="padding: 8px 0; text-align: left; color: {_TEXT_SECONDARY}; font-weight: normal;">PROVIDER</th>
                    <th style="padding: 8px 0; text-align: right; color: {_TEXT_SECONDARY}; font-weight: normal;">TOKENS</th>
                    <th style="padding: 8px 0; text-align: right; color: {_TEXT_SECONDARY}; font-weight: normal;">COST</th>
                </tr>
            </thead>
            <tbody>
    """

    for _, usage in sorted(cost.by_provider.items(), key=lambda x: x[1].cost, reverse=True):
        provider_name = usage.provider
        if usage.is_local:
            provider_name += f' <span style="color: {_GREEN};">[local]</span>'

        cost_color = _GREEN if usage.cost == 0 else _TEXT_PRIMARY

        html += f"""
            <tr style="border-bottom: 1px solid {_BORDER};">
                <td style="padding: 8px 0; color: {_TEXT_PRIMARY};">{provider_name}</td>
                <td style="padding: 8px 0; text-align: right; color: {_TEXT_SECONDARY};">{usage.total_tokens:,}</td>
                <td style="padding: 8px 0; text-align: right; color: {cost_color};">${usage.cost:.4f}</td>
            </tr>
        """

    # Total row
    total_color = _GREEN if cost.total_cost == 0 else _TEXT_PRIMARY
    html += f"""
            </tbody>
            <tfoot>
                <tr>
                    <td style="padding: 10px 0; color: {_TEXT_PRIMARY}; font-weight: bold;">TOTAL</td>
                    <td style="padding: 10px 0; text-align: right; color: {_TEXT_PRIMARY};">{cost.total_tokens:,}</td>
                    <td style="padding: 10px 0; text-align: right; color: {total_color}; font-weight: bold;">${cost.total_cost:.4f}</td>
                </tr>
            </tfoot>
        </table>
    </div>
    """

    _display_html(html)


def display(result: EvaluationResult) -> None:
    """Display full evaluation results with metrics and cost.

    This is the main display function that shows:
    - Evaluation metrics with progress bars
    - Cost summary (if tracking was enabled)
    - Latency information

    Args:
        result: EvaluationResult from evaluate().

    Example:
        >>> from ragnarok_ai import evaluate
        >>> from ragnarok_ai.notebook import display
        >>>
        >>> results = await evaluate(rag, testset, track_cost=True)
        >>> display(results)
    """
    summary = result.summary()

    # Header
    html = f"""
    <div style="font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;">
        <div style="background: {_BG_HEADER}; color: {_CYAN}; padding: 16px; border-radius: 6px 6px 0 0; border: 1px solid {_BORDER}; border-bottom: none;">
            <span style="color: {_GREEN};">$</span> ragnarok evaluate --summary
        </div>
        <div style="background: {_BG_DARK}; padding: 20px; border-radius: 0 0 6px 6px; border: 1px solid {_BORDER}; border-top: none;">
    """

    # Quick stats
    num_queries = int(summary.get("num_queries", 0)) if summary else 0
    latency_s = result.total_latency_ms / 1000
    avg_latency = result.total_latency_ms / max(num_queries, 1)

    html += f"""
        <div style="display: flex; gap: 24px; margin-bottom: 20px; flex-wrap: wrap;">
            <div>
                <div style="font-size: 24px; font-weight: bold; color: {_CYAN};">{num_queries}</div>
                <div style="font-size: 11px; color: {_TEXT_MUTED};">queries</div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: bold; color: {_PURPLE};">{latency_s:.2f}s</div>
                <div style="font-size: 11px; color: {_TEXT_MUTED};">total_time</div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: bold; color: {_YELLOW};">{avg_latency:.0f}ms</div>
                <div style="font-size: 11px; color: {_TEXT_MUTED};">avg/query</div>
            </div>
    """

    # Cost stat if available
    if result.cost is not None:
        cost_display = f"${result.cost.total_cost:.4f}" if result.cost.total_cost > 0 else "$0.00"
        cost_color = _GREEN if result.cost.total_cost == 0 else _RED
        html += f"""
            <div>
                <div style="font-size: 24px; font-weight: bold; color: {cost_color};">{cost_display}</div>
                <div style="font-size: 11px; color: {_TEXT_MUTED};">cost</div>
            </div>
        """

    html += "</div>"

    # Metrics section
    if summary:
        html += f"""
        <div style="color: {_TEXT_MUTED}; font-size: 12px; margin: 16px 0 8px 0;"># RETRIEVAL METRICS</div>
        <table style="width: 100%; border-collapse: collapse;">
        """

        metrics = [
            ("precision", summary.get("precision", 0)),
            ("recall", summary.get("recall", 0)),
            ("mrr", summary.get("mrr", 0)),
            ("ndcg", summary.get("ndcg", 0)),
        ]

        for name, value in metrics:
            html += f"""
            <tr style="border-bottom: 1px solid {_BORDER};">
                <td style="padding: 8px 0; color: {_TEXT_SECONDARY}; width: 100px;">{name}</td>
                <td style="padding: 8px 0;">{_progress_bar(value)}</td>
            </tr>
            """

        html += "</table>"

    # Cost section
    if result.cost is not None and result.cost.by_provider:
        html += f"""
        <div style="color: {_TEXT_MUTED}; font-size: 12px; margin: 20px 0 8px 0;"># COST BREAKDOWN</div>
        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
            <thead>
                <tr style="border-bottom: 1px solid {_BORDER};">
                    <th style="padding: 8px 0; text-align: left; color: {_TEXT_MUTED}; font-weight: normal;">provider</th>
                    <th style="padding: 8px 0; text-align: right; color: {_TEXT_MUTED}; font-weight: normal;">model</th>
                    <th style="padding: 8px 0; text-align: right; color: {_TEXT_MUTED}; font-weight: normal;">tokens</th>
                    <th style="padding: 8px 0; text-align: right; color: {_TEXT_MUTED}; font-weight: normal;">cost</th>
                </tr>
            </thead>
            <tbody>
        """

        for _, usage in sorted(result.cost.by_provider.items(), key=lambda x: x[1].cost, reverse=True):
            local_badge = f' <span style="color: {_GREEN};">[local]</span>' if usage.is_local else ""
            cost_color = _GREEN if usage.cost == 0 else _TEXT_PRIMARY

            html += f"""
            <tr style="border-bottom: 1px solid {_BORDER};">
                <td style="padding: 8px 0; color: {_TEXT_PRIMARY};">{usage.provider}{local_badge}</td>
                <td style="padding: 8px 0; text-align: right; color: {_TEXT_SECONDARY};">{usage.model}</td>
                <td style="padding: 8px 0; text-align: right; color: {_TEXT_SECONDARY};">{usage.total_tokens:,}</td>
                <td style="padding: 8px 0; text-align: right; color: {cost_color};">${usage.cost:.4f}</td>
            </tr>
            """

        html += "</tbody></table>"

    # Errors section
    if result.errors:
        html += f"""
        <div style="margin-top: 20px; background: {_BG_LIGHTER}; border-left: 3px solid {_RED}; padding: 12px;">
            <span style="color: {_RED};">ERROR:</span>
            <span style="color: {_TEXT_PRIMARY};"> {len(result.errors)} error(s) occurred</span>
        </div>
        """

    html += """
        </div>
    </div>
    """

    _display_html(html)


def display_comparison(
    results: list[tuple[str, EvaluationResult]],
) -> None:
    """Display side-by-side comparison of multiple evaluation results.

    Args:
        results: List of (name, EvaluationResult) tuples.

    Example:
        >>> from ragnarok_ai.notebook import display_comparison
        >>> display_comparison([
        ...     ("Baseline", baseline_results),
        ...     ("New Model", new_results),
        ... ])
    """
    if not results:
        _display_html(f'<p style="color: {_TEXT_MUTED}; font-family: monospace;">No results to compare.</p>')
        return

    html = f"""
    <div style="font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;">
        <div style="background: {_BG_HEADER}; color: {_CYAN}; padding: 16px; border-radius: 6px 6px 0 0; border: 1px solid {_BORDER}; border-bottom: none;">
            <span style="color: {_GREEN};">$</span> ragnarok compare
        </div>
        <div style="background: {_BG_DARK}; padding: 20px; border-radius: 0 0 6px 6px; border: 1px solid {_BORDER}; border-top: none;">
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <thead>
                    <tr style="border-bottom: 1px solid {_BORDER};">
                        <th style="padding: 10px 0; text-align: left; color: {_TEXT_MUTED}; font-weight: normal;">metric</th>
    """

    for name, _ in results:
        html += f'<th style="padding: 10px; text-align: center; color: {_TEXT_PRIMARY};">{name}</th>'

    html += "</tr></thead><tbody>"

    metrics = ["precision", "recall", "mrr", "ndcg"]

    for metric in metrics:
        html += f'<tr style="border-bottom: 1px solid {_BORDER};"><td style="padding: 10px 0; color: {_TEXT_SECONDARY};">{metric}</td>'

        values = [r.summary().get(metric, 0) for _, r in results]
        max_val = max(values) if values else 0

        for (_, __), val in zip(results, values, strict=False):
            is_best = val == max_val and max_val > 0
            color = _GREEN if is_best else _TEXT_PRIMARY
            marker = " *" if is_best and len(results) > 1 else ""
            html += f'<td style="padding: 10px; text-align: center; color: {color};">{_format_number(val)}{marker}</td>'

        html += "</tr>"

    # Cost row
    html += f'<tr style="border-bottom: 1px solid {_BORDER};"><td style="padding: 10px 0; color: {_TEXT_SECONDARY};">cost</td>'
    for _, result in results:
        if result.cost:
            cost = result.cost.total_cost
            cost_str = f"${cost:.4f}" if cost > 0 else "$0.00"
            color = _GREEN if cost == 0 else _TEXT_PRIMARY
        else:
            cost_str = "N/A"
            color = _TEXT_MUTED
        html += f'<td style="padding: 10px; text-align: center; color: {color};">{cost_str}</td>'
    html += "</tr>"

    # Latency row
    html += f'<tr><td style="padding: 10px 0; color: {_TEXT_SECONDARY};">latency</td>'
    for _, result in results:
        latency = result.total_latency_ms / 1000
        html += f'<td style="padding: 10px; text-align: center; color: {_TEXT_PRIMARY};">{latency:.2f}s</td>'
    html += "</tr>"

    html += f"""
            </tbody>
        </table>
        <div style="margin-top: 12px; color: {_TEXT_MUTED}; font-size: 11px;">
            * = best score
        </div>
        </div>
    </div>
    """

    _display_html(html)
