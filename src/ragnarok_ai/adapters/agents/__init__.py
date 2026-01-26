"""Agent adapters for ragnarok-ai.

This module provides adapters for common agent patterns:
- ReAct (Reason + Act)
- Chain-of-Thought (CoT)

Example:
    >>> from ragnarok_ai.adapters.agents import ReActParser, ReActAdapter
    >>>
    >>> # Parse ReAct output
    >>> parser = ReActParser()
    >>> steps = parser.parse(raw_output)
    >>>
    >>> # Wrap a ReAct agent
    >>> adapter = ReActAdapter(my_agent)
    >>> response = await adapter.query("What is X?")
"""

from __future__ import annotations

from ragnarok_ai.adapters.agents.cot import ChainOfThoughtAdapter
from ragnarok_ai.adapters.agents.react import ReActAdapter, ReActParser

__all__ = [
    "ChainOfThoughtAdapter",
    "ReActAdapter",
    "ReActParser",
]
