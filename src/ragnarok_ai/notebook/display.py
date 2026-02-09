"""Rich display functions for Jupyter notebooks.

This module provides HTML-rich display functions for RAGnarok-AI results
in Jupyter notebooks and IPython environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragnarok_ai.core.evaluate import EvaluationResult
    from ragnarok_ai.cost.tracker import CostSummary


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


def _progress_bar(value: float, max_value: float = 1.0, width: int = 100) -> str:
    """Generate an HTML progress bar."""
    percentage = min(100, max(0, (value / max_value) * 100))

    # Color based on value (red -> yellow -> green)
    if percentage < 50:
        color = "#e74c3c"  # Red
    elif percentage < 75:
        color = "#f39c12"  # Orange
    else:
        color = "#27ae60"  # Green

    return f"""
    <div style="background: #ecf0f1; border-radius: 4px; width: {width}px; height: 16px; display: inline-block; vertical-align: middle;">
        <div style="background: {color}; width: {percentage}%; height: 100%; border-radius: 4px;"></div>
    </div>
    <span style="margin-left: 8px; font-weight: 500;">{_format_number(value)}</span>
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
        _display_html("<p><em>No metrics available.</em></p>")
        return

    html = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 16px; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 500px;">
        <h3 style="margin: 0 0 16px 0; color: #2c3e50; font-size: 16px; border-bottom: 2px solid #3498db; padding-bottom: 8px;">
            📊 Evaluation Metrics
        </h3>
        <table style="width: 100%; border-collapse: collapse;">
    """

    metrics = [
        ("Precision", summary.get("precision", 0)),
        ("Recall", summary.get("recall", 0)),
        ("MRR", summary.get("mrr", 0)),
        ("NDCG", summary.get("ndcg", 0)),
    ]

    for name, value in metrics:
        html += f"""
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 8px 0; color: #7f8c8d; width: 80px;">{name}</td>
            <td style="padding: 8px 0;">{_progress_bar(value)}</td>
        </tr>
        """

    num_queries = int(summary.get("num_queries", 0))
    html += f"""
        </table>
        <p style="margin: 12px 0 0 0; color: #95a5a6; font-size: 12px;">
            Based on {num_queries} queries
        </p>
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
        _display_html("""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 16px; background: #fef9e7; border-radius: 8px; border-left: 4px solid #f39c12;">
            <p style="margin: 0; color: #7d6608;">
                ⚠️ Cost tracking was not enabled. Use <code>track_cost=True</code> in evaluate().
            </p>
        </div>
        """)
        return

    _display_cost_summary(result.cost)


def _display_cost_summary(cost: CostSummary) -> None:
    """Display a CostSummary object."""
    html = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 16px; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 450px;">
        <h3 style="margin: 0 0 16px 0; color: #2c3e50; font-size: 16px; border-bottom: 2px solid #27ae60; padding-bottom: 8px;">
            💰 Cost Summary
        </h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <thead>
                <tr style="background: #f8f9fa;">
                    <th style="padding: 10px; text-align: left; color: #7f8c8d; font-weight: 500;">Provider</th>
                    <th style="padding: 10px; text-align: right; color: #7f8c8d; font-weight: 500;">Tokens</th>
                    <th style="padding: 10px; text-align: right; color: #7f8c8d; font-weight: 500;">Cost</th>
                </tr>
            </thead>
            <tbody>
    """

    for _, usage in sorted(cost.by_provider.items(), key=lambda x: x[1].cost, reverse=True):
        provider_name = usage.provider
        if usage.is_local:
            provider_name += ' <span style="background: #d5f5e3; color: #1e8449; padding: 2px 6px; border-radius: 4px; font-size: 11px;">local</span>'

        cost_color = "#27ae60" if usage.cost == 0 else "#2c3e50"

        html += f"""
            <tr style="border-bottom: 1px solid #ecf0f1;">
                <td style="padding: 10px; color: #2c3e50;">{provider_name}</td>
                <td style="padding: 10px; text-align: right; color: #7f8c8d;">{usage.total_tokens:,}</td>
                <td style="padding: 10px; text-align: right; color: {cost_color}; font-weight: 500;">${usage.cost:.2f}</td>
            </tr>
        """

    # Total row
    total_color = "#27ae60" if cost.total_cost == 0 else "#2c3e50"
    html += f"""
            </tbody>
            <tfoot>
                <tr style="background: #f8f9fa; font-weight: 600;">
                    <td style="padding: 10px; color: #2c3e50;">Total</td>
                    <td style="padding: 10px; text-align: right; color: #2c3e50;">{cost.total_tokens:,}</td>
                    <td style="padding: 10px; text-align: right; color: {total_color};">${cost.total_cost:.2f}</td>
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
    html = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px 8px 0 0;">
            <h2 style="margin: 0; font-size: 20px; font-weight: 600;">⚡ RAGnarok Evaluation Results</h2>
        </div>
        <div style="background: #fff; padding: 20px; border-radius: 0 0 8px 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    """

    # Quick stats
    num_queries = int(summary.get("num_queries", 0)) if summary else 0
    latency_s = result.total_latency_ms / 1000
    avg_latency = result.total_latency_ms / max(num_queries, 1)

    html += f"""
        <div style="display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;">
            <div style="background: #f8f9fa; padding: 12px 20px; border-radius: 6px; text-align: center;">
                <div style="font-size: 24px; font-weight: 600; color: #3498db;">{num_queries}</div>
                <div style="font-size: 12px; color: #7f8c8d;">Queries</div>
            </div>
            <div style="background: #f8f9fa; padding: 12px 20px; border-radius: 6px; text-align: center;">
                <div style="font-size: 24px; font-weight: 600; color: #9b59b6;">{latency_s:.1f}s</div>
                <div style="font-size: 12px; color: #7f8c8d;">Total Time</div>
            </div>
            <div style="background: #f8f9fa; padding: 12px 20px; border-radius: 6px; text-align: center;">
                <div style="font-size: 24px; font-weight: 600; color: #e67e22;">{avg_latency:.0f}ms</div>
                <div style="font-size: 12px; color: #7f8c8d;">Avg/Query</div>
            </div>
    """

    # Cost stat if available
    if result.cost is not None:
        cost_display = f"${result.cost.total_cost:.2f}" if result.cost.total_cost > 0 else "FREE"
        cost_color = "#27ae60" if result.cost.total_cost == 0 else "#e74c3c"
        html += f"""
            <div style="background: #f8f9fa; padding: 12px 20px; border-radius: 6px; text-align: center;">
                <div style="font-size: 24px; font-weight: 600; color: {cost_color};">{cost_display}</div>
                <div style="font-size: 12px; color: #7f8c8d;">Cost</div>
            </div>
        """

    html += "</div>"

    # Metrics section
    if summary:
        html += """
        <h4 style="margin: 20px 0 12px 0; color: #2c3e50; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Retrieval Metrics</h4>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
        """

        metrics = [
            ("Precision", summary.get("precision", 0), "Relevant docs / Retrieved docs"),
            ("Recall", summary.get("recall", 0), "Retrieved relevant / Total relevant"),
            ("MRR", summary.get("mrr", 0), "Mean Reciprocal Rank"),
            ("NDCG", summary.get("ndcg", 0), "Normalized DCG"),
        ]

        for name, value, tooltip in metrics:
            html += f'''
            <tr style="border-bottom: 1px solid #ecf0f1;">
                <td style="padding: 10px 0; color: #7f8c8d; width: 100px;" title="{tooltip}">{name}</td>
                <td style="padding: 10px 0;">{_progress_bar(value, width=200)}</td>
            </tr>
            '''

        html += "</table>"

    # Cost section
    if result.cost is not None and result.cost.by_provider:
        html += """
        <h4 style="margin: 20px 0 12px 0; color: #2c3e50; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Cost Breakdown</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background: #f8f9fa;">
                    <th style="padding: 10px; text-align: left; color: #7f8c8d; font-weight: 500;">Provider</th>
                    <th style="padding: 10px; text-align: right; color: #7f8c8d; font-weight: 500;">Model</th>
                    <th style="padding: 10px; text-align: right; color: #7f8c8d; font-weight: 500;">Tokens</th>
                    <th style="padding: 10px; text-align: right; color: #7f8c8d; font-weight: 500;">Cost</th>
                </tr>
            </thead>
            <tbody>
        """

        for _, usage in sorted(result.cost.by_provider.items(), key=lambda x: x[1].cost, reverse=True):
            local_badge = (
                ' <span style="background: #d5f5e3; color: #1e8449; padding: 2px 6px; border-radius: 4px; font-size: 10px;">local</span>'
                if usage.is_local
                else ""
            )
            cost_color = "#27ae60" if usage.cost == 0 else "#2c3e50"

            html += f"""
            <tr style="border-bottom: 1px solid #ecf0f1;">
                <td style="padding: 10px; color: #2c3e50;">{usage.provider}{local_badge}</td>
                <td style="padding: 10px; text-align: right; color: #7f8c8d; font-family: monospace; font-size: 12px;">{usage.model}</td>
                <td style="padding: 10px; text-align: right; color: #7f8c8d;">{usage.total_tokens:,}</td>
                <td style="padding: 10px; text-align: right; color: {cost_color}; font-weight: 500;">${usage.cost:.2f}</td>
            </tr>
            """

        html += "</tbody></table>"

    # Errors section
    if result.errors:
        html += f"""
        <div style="margin-top: 20px; background: #fdedec; border-left: 4px solid #e74c3c; padding: 12px; border-radius: 4px;">
            <strong style="color: #c0392b;">⚠️ {len(result.errors)} error(s) occurred</strong>
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
        _display_html("<p><em>No results to compare.</em></p>")
        return

    html = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 20px; border-radius: 8px 8px 0 0;">
            <h2 style="margin: 0; font-size: 20px; font-weight: 600;">📊 Pipeline Comparison</h2>
        </div>
        <div style="background: #fff; padding: 20px; border-radius: 0 0 8px 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: #f8f9fa;">
                        <th style="padding: 12px; text-align: left; color: #7f8c8d;">Metric</th>
    """

    for name, _ in results:
        html += f'<th style="padding: 12px; text-align: center; color: #2c3e50;">{name}</th>'

    html += "</tr></thead><tbody>"

    metrics = ["precision", "recall", "mrr", "ndcg"]
    metric_names = ["Precision", "Recall", "MRR", "NDCG"]

    for metric, metric_name in zip(metrics, metric_names, strict=False):
        html += f'<tr style="border-bottom: 1px solid #ecf0f1;"><td style="padding: 12px; color: #7f8c8d;">{metric_name}</td>'

        values = [r.summary().get(metric, 0) for _, r in results]
        max_val = max(values) if values else 0

        for (_, __), val in zip(results, values, strict=False):
            is_best = val == max_val and max_val > 0
            style = "font-weight: 600; color: #27ae60;" if is_best else "color: #2c3e50;"
            badge = " 🏆" if is_best and len(results) > 1 else ""
            html += f'<td style="padding: 12px; text-align: center; {style}">{_format_number(val)}{badge}</td>'

        html += "</tr>"

    # Cost row
    html += '<tr style="border-bottom: 1px solid #ecf0f1;"><td style="padding: 12px; color: #7f8c8d;">Cost</td>'
    for _, result in results:
        if result.cost:
            cost = result.cost.total_cost
            cost_str = f"${cost:.2f}" if cost > 0 else "FREE"
            color = "#27ae60" if cost == 0 else "#e74c3c"
        else:
            cost_str = "N/A"
            color = "#95a5a6"
        html += f'<td style="padding: 12px; text-align: center; color: {color};">{cost_str}</td>'
    html += "</tr>"

    # Latency row
    html += '<tr><td style="padding: 12px; color: #7f8c8d;">Latency</td>'
    for _, result in results:
        latency = result.total_latency_ms / 1000
        html += f'<td style="padding: 12px; text-align: center; color: #2c3e50;">{latency:.1f}s</td>'
    html += "</tr>"

    html += """
            </tbody>
        </table>
        </div>
    </div>
    """

    _display_html(html)
