"""Trajectory analysis and visualization for agents.

This module provides tools for analyzing and visualizing agent
execution trajectories.

Example:
    >>> from ragnarok_ai.agents.analysis import (
    ...     TrajectoryAnalyzer,
    ...     print_trajectory,
    ...     format_trajectory_mermaid,
    ... )
    >>>
    >>> # Analyze trajectory
    >>> analyzer = TrajectoryAnalyzer()
    >>> summary = analyzer.analyze(response)
    >>> print(summary)
    >>>
    >>> # ASCII visualization
    >>> print_trajectory(response)
    >>>
    >>> # Mermaid for GitHub/docs
    >>> print(format_trajectory_mermaid(response))
"""

from __future__ import annotations

from ragnarok_ai.agents.analysis.analyzer import (
    FailurePoint,
    TrajectoryAnalyzer,
    TrajectorySummary,
)
from ragnarok_ai.agents.analysis.visualizer import (
    export_trajectory_html,
    format_trajectory_ascii,
    format_trajectory_mermaid,
    generate_trajectory_html,
    print_trajectory,
)

__all__ = [
    "FailurePoint",
    "TrajectoryAnalyzer",
    "TrajectorySummary",
    "export_trajectory_html",
    "format_trajectory_ascii",
    "format_trajectory_mermaid",
    "generate_trajectory_html",
    "print_trajectory",
]
