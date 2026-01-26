"""Agent evaluation module for ragnarok-ai.

This module provides the foundation for evaluating AI agents,
including the AgentProtocol interface and core types.

Example:
    >>> from ragnarok_ai.agents import AgentProtocol, AgentResponse, AgentStep, ToolCall
    >>>
    >>> class MyAgent:
    ...     async def query(self, question: str) -> AgentResponse:
    ...         steps = [
    ...             AgentStep(step_type="thought", content="Thinking..."),
    ...             AgentStep(step_type="final_answer", content="Done"),
    ...         ]
    ...         return AgentResponse(answer="42", steps=steps)
    ...
    >>> # MyAgent implements AgentProtocol via duck typing
    >>> isinstance(MyAgent(), AgentProtocol)
    True
"""

from __future__ import annotations

from ragnarok_ai.agents.protocols import AgentProtocol
from ragnarok_ai.agents.types import AgentResponse, AgentStep, ToolCall

__all__ = [
    "AgentProtocol",
    "AgentResponse",
    "AgentStep",
    "ToolCall",
]
