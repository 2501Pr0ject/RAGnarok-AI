"""Jupyter notebook integration for RAGnarok-AI.

This module provides rich HTML display functions for evaluation results
in Jupyter notebooks and IPython environments.

Example:
    >>> from ragnarok_ai import evaluate
    >>> from ragnarok_ai.notebook import display
    >>>
    >>> results = await evaluate(rag, testset, track_cost=True)
    >>> display(results)  # Rich HTML output in notebook
"""

from ragnarok_ai.notebook.display import (
    display,
    display_comparison,
    display_cost,
    display_metrics,
)

__all__ = [
    "display",
    "display_comparison",
    "display_cost",
    "display_metrics",
]
