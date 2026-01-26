"""Protocol definitions for agent evaluation.

This module provides the AgentProtocol interface that all agents
must implement to be evaluated by ragnarok-ai.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ragnarok_ai.agents.types import AgentResponse


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for agent implementations.

    Any agent that implements async query() returning AgentResponse works.
    Supports: ReAct, CoT, tool-using, multi-turn agents.

    This protocol uses duck typing via @runtime_checkable, meaning any class
    that implements the required method signature will work without explicit
    inheritance.

    Example:
        >>> class MyAgent:
        ...     async def query(self, question: str) -> AgentResponse:
        ...         # Execute agent logic
        ...         return AgentResponse(answer="...", steps=[...])
        ...
        >>> # Works with evaluate_agent() without inheritance
        >>> agent = MyAgent()
        >>> isinstance(agent, AgentProtocol)  # True via duck typing
        True

    Attributes:
        None required. Only the query() method signature matters.

    See Also:
        - RAGProtocol: Similar protocol for RAG pipelines
        - AgentResponse: The return type containing trajectory
    """

    async def query(self, question: str) -> AgentResponse:
        """Execute agent and return response with trajectory.

        This is the only method required by the protocol. Implement your
        agent logic here, tracking steps and tool calls as you go.

        Args:
            question: The question or task for the agent to handle.

        Returns:
            AgentResponse containing:
                - answer: The final answer string
                - steps: List of AgentStep showing execution trajectory
                - total_latency_ms: Total execution time
                - metadata: Optional additional data

        Example:
            >>> async def query(self, question: str) -> AgentResponse:
            ...     steps = []
            ...     # Add thought step
            ...     steps.append(AgentStep(
            ...         step_type="thought",
            ...         content="I need to analyze this question.",
            ...     ))
            ...     # Add tool call step
            ...     result = await self.search(question)
            ...     steps.append(AgentStep(
            ...         step_type="action",
            ...         content="Searching...",
            ...         tool_call=ToolCall(name="search", input={...}, output=result),
            ...     ))
            ...     # Return response
            ...     return AgentResponse(answer="...", steps=steps)
        """
        ...
