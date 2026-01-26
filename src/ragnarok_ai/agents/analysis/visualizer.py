"""Trajectory visualization for agent evaluation.

This module provides visualization tools for agent trajectories:
- ASCII box drawing for CLI
- Mermaid flowcharts for GitHub/docs
- HTML reports for interactive viewing

Example:
    >>> from ragnarok_ai.agents.analysis import print_trajectory, format_trajectory_mermaid
    >>>
    >>> # ASCII output
    >>> print_trajectory(response)
    >>>
    >>> # Mermaid for GitHub
    >>> mermaid = format_trajectory_mermaid(response)
    >>> print(mermaid)
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragnarok_ai.agents.types import AgentResponse, AgentStep


# =============================================================================
# ASCII Visualization
# =============================================================================


def format_trajectory_ascii(response: AgentResponse, max_width: int = 60) -> str:
    """Format trajectory as ASCII box drawing.

    Args:
        response: Agent response with trajectory.
        max_width: Maximum width for content (default 60).

    Returns:
        ASCII formatted string with box drawing.

    Example:
        >>> print(format_trajectory_ascii(response))
        ┌──────────────────────────────────────────────────────────┐
        │ Step 1: thought                                          │
        │ "I need to search for information"                       │
        ├──────────────────────────────────────────────────────────┤
        │ Step 2: action [search]                                  │
        │ Input: {"query": "capital of France"}                    │
        │ Output: "Paris is the capital..."                        │
        └──────────────────────────────────────────────────────────┘
    """
    if not response.steps:
        return "┌─ Empty trajectory ─┐\n└────────────────────┘"

    lines: list[str] = []
    inner_width = max_width - 4  # Account for "│ " and " │"

    # Top border
    lines.append("┌" + "─" * (max_width - 2) + "┐")

    for i, step in enumerate(response.steps):
        if i > 0:
            # Separator between steps
            lines.append("├" + "─" * (max_width - 2) + "┤")

        # Step header
        header = _format_step_header(step, i)
        lines.append(_pad_line(header, inner_width))

        # Step content
        content_lines = _wrap_content(step.content, inner_width - 2)
        for content_line in content_lines[:3]:  # Limit to 3 lines
            lines.append(_pad_line(f'  "{content_line}"', inner_width))

        if len(content_lines) > 3:
            lines.append(_pad_line("  ...", inner_width))

        # Tool call details
        if step.tool_call:
            input_str = json.dumps(step.tool_call.input, ensure_ascii=False)
            if len(input_str) > inner_width - 10:
                input_str = input_str[: inner_width - 13] + "..."
            lines.append(_pad_line(f"  Input: {input_str}", inner_width))

            if step.tool_call.output:
                output_str = step.tool_call.output
                if len(output_str) > inner_width - 12:
                    output_str = output_str[: inner_width - 15] + "..."
                lines.append(_pad_line(f"  Output: {output_str}", inner_width))

            if step.tool_call.error:
                lines.append(_pad_line(f"  Error: {step.tool_call.error}", inner_width))

    # Bottom border
    lines.append("└" + "─" * (max_width - 2) + "┘")

    return "\n".join(lines)


def _format_step_header(step: AgentStep, index: int) -> str:
    """Format step header line."""
    header = f"Step {index + 1}: {step.step_type}"
    if step.tool_call:
        header += f" [{step.tool_call.name}]"
    return header


def _wrap_content(content: str, width: int) -> list[str]:
    """Wrap content to specified width."""
    content = content.replace("\n", " ").strip()
    if not content:
        return []

    lines: list[str] = []
    while content:
        if len(content) <= width:
            lines.append(content)
            break
        # Find last space before width
        split_idx = content.rfind(" ", 0, width)
        if split_idx == -1:
            split_idx = width
        lines.append(content[:split_idx])
        content = content[split_idx:].strip()

    return lines


def _pad_line(content: str, inner_width: int) -> str:
    """Pad content to fit box width."""
    if len(content) > inner_width:
        content = content[: inner_width - 3] + "..."
    return "│ " + content.ljust(inner_width) + " │"


def print_trajectory(response: AgentResponse, max_width: int = 60) -> None:
    """Print trajectory to stdout.

    Args:
        response: Agent response with trajectory.
        max_width: Maximum width for content.
    """
    print(format_trajectory_ascii(response, max_width))


# =============================================================================
# Mermaid Visualization
# =============================================================================


def format_trajectory_mermaid(response: AgentResponse) -> str:
    """Format trajectory as Mermaid flowchart.

    Renders natively on GitHub, GitLab, and most documentation sites.

    Args:
        response: Agent response with trajectory.

    Returns:
        Mermaid flowchart string.

    Example:
        >>> print(format_trajectory_mermaid(response))
        ```mermaid
        graph TD
            T1["💭 Thought: Need to search"] --> A1["🔧 search"]
            A1 --> O1["👁 Found: Paris..."]
            O1 --> F["✅ Answer: Paris"]
        ```
    """
    if not response.steps:
        return "```mermaid\ngraph TD\n    empty[No steps]\n```"

    lines = ["```mermaid", "graph TD"]

    node_ids: list[str] = []
    for i, step in enumerate(response.steps):
        node_id = _get_node_id(step, i)
        node_ids.append(node_id)

        label = _get_mermaid_label(step)
        lines.append(f"    {node_id}[{label}]")

    # Add connections
    for i in range(len(node_ids) - 1):
        lines.append(f"    {node_ids[i]} --> {node_ids[i + 1]}")

    # Style failed steps
    for i, step in enumerate(response.steps):
        if step.tool_call and not step.tool_call.success:
            lines.append(f"    style {node_ids[i]} fill:#ffcccc")

    lines.append("```")
    return "\n".join(lines)


def _get_node_id(step: AgentStep, index: int) -> str:
    """Generate unique node ID for Mermaid."""
    prefix_map = {
        "thought": "T",
        "action": "A",
        "observation": "O",
        "final_answer": "F",
    }
    prefix = prefix_map.get(step.step_type, "S")
    return f"{prefix}{index + 1}"


def _get_mermaid_label(step: AgentStep) -> str:
    """Generate Mermaid node label with emoji."""
    emoji_map = {
        "thought": "💭",
        "action": "🔧",
        "observation": "👁",
        "final_answer": "✅",
    }
    emoji = emoji_map.get(step.step_type, "📌")

    # Truncate content
    content = step.content.replace("\n", " ").strip()
    if len(content) > 30:
        content = content[:27] + "..."

    # Escape quotes for Mermaid
    content = content.replace('"', "'")

    if step.tool_call:
        return f'"{emoji} {step.tool_call.name}: {content}"'
    return f'"{emoji} {step.step_type.title()}: {content}"'


# =============================================================================
# HTML Visualization
# =============================================================================


def export_trajectory_html(
    response: AgentResponse,
    output_path: str | Path,
    title: str = "Agent Trajectory",
) -> None:
    """Export trajectory as interactive HTML report.

    Args:
        response: Agent response with trajectory.
        output_path: Path to write HTML file.
        title: Title for the HTML page.
    """
    html_content = generate_trajectory_html(response, title)
    Path(output_path).write_text(html_content, encoding="utf-8")


def generate_trajectory_html(response: AgentResponse, title: str = "Agent Trajectory") -> str:
    """Generate HTML content for trajectory visualization.

    Args:
        response: Agent response with trajectory.
        title: Title for the HTML page.

    Returns:
        Complete HTML document string.
    """
    steps_html = []
    for i, step in enumerate(response.steps):
        step_html = _generate_step_html(step, i)
        steps_html.append(step_html)

    # Summary stats
    tool_counts: dict[str, int] = {}
    for step in response.steps:
        if step.tool_call:
            name = step.tool_call.name
            tool_counts[name] = tool_counts.get(name, 0) + 1

    tools_summary = ", ".join(f"{name} ({count}x)" for name, count in sorted(tool_counts.items()))
    if not tools_summary:
        tools_summary = "None"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --border-color: #0f3460;
            --thought-color: #9b59b6;
            --action-color: #3498db;
            --observation-color: #2ecc71;
            --answer-color: #f39c12;
            --error-color: #e74c3c;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
        }}
        .summary {{
            background: var(--card-bg);
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }}
        .summary-item {{
            display: flex;
            flex-direction: column;
        }}
        .summary-label {{
            font-size: 0.85em;
            opacity: 0.7;
        }}
        .summary-value {{
            font-size: 1.2em;
            font-weight: 600;
        }}
        .timeline {{
            position: relative;
            padding-left: 30px;
        }}
        .timeline::before {{
            content: '';
            position: absolute;
            left: 10px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: var(--border-color);
        }}
        .step {{
            position: relative;
            background: var(--card-bg);
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 15px;
            border-left: 4px solid var(--border-color);
        }}
        .step::before {{
            content: '';
            position: absolute;
            left: -26px;
            top: 20px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--border-color);
        }}
        .step.thought {{ border-left-color: var(--thought-color); }}
        .step.thought::before {{ background: var(--thought-color); }}
        .step.action {{ border-left-color: var(--action-color); }}
        .step.action::before {{ background: var(--action-color); }}
        .step.observation {{ border-left-color: var(--observation-color); }}
        .step.observation::before {{ background: var(--observation-color); }}
        .step.final_answer {{ border-left-color: var(--answer-color); }}
        .step.final_answer::before {{ background: var(--answer-color); }}
        .step.error {{ border-left-color: var(--error-color); }}
        .step-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .step-type {{
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        .step-index {{
            opacity: 0.5;
            font-size: 0.85em;
        }}
        .step-content {{
            background: rgba(0,0,0,0.2);
            padding: 10px 15px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .tool-details {{
            margin-top: 10px;
            font-size: 0.9em;
        }}
        .tool-details code {{
            background: rgba(0,0,0,0.3);
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Consolas', monospace;
        }}
        .tool-name {{
            color: var(--action-color);
            font-weight: 600;
        }}
        .error-msg {{
            color: var(--error-color);
            margin-top: 5px;
        }}
        .answer-box {{
            background: linear-gradient(135deg, var(--card-bg), #1e3a5f);
            border: 2px solid var(--answer-color);
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .answer-label {{
            color: var(--answer-color);
            font-weight: 600;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{html.escape(title)}</h1>

        <div class="summary">
            <div class="summary-item">
                <span class="summary-label">Steps</span>
                <span class="summary-value">{len(response.steps)}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Tools Used</span>
                <span class="summary-value">{html.escape(tools_summary)}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Latency</span>
                <span class="summary-value">{response.total_latency_ms:.0f}ms</span>
            </div>
        </div>

        <div class="timeline">
            {"".join(steps_html)}
        </div>

        <div class="answer-box">
            <div class="answer-label">Final Answer</div>
            <div>{html.escape(response.answer)}</div>
        </div>
    </div>
</body>
</html>"""


def _generate_step_html(step: AgentStep, index: int) -> str:
    """Generate HTML for a single step."""
    step_class: str = step.step_type
    if step.tool_call and not step.tool_call.success:
        step_class += " error"

    tool_html = ""
    if step.tool_call:
        input_json = html.escape(json.dumps(step.tool_call.input, indent=2))
        output_escaped = html.escape(step.tool_call.output[:500] if step.tool_call.output else "")

        tool_html = f"""
        <div class="tool-details">
            <span class="tool-name">{html.escape(step.tool_call.name)}</span>
            <div>Input: <code>{input_json}</code></div>
            <div>Output: <code>{output_escaped}</code></div>
        </div>"""

        if step.tool_call.error:
            tool_html += f'<div class="error-msg">Error: {html.escape(step.tool_call.error)}</div>'

    return f"""
        <div class="step {step_class}">
            <div class="step-header">
                <span class="step-type">{step.step_type}</span>
                <span class="step-index">Step {index + 1}</span>
            </div>
            <div class="step-content">{html.escape(step.content)}</div>
            {tool_html}
        </div>"""
