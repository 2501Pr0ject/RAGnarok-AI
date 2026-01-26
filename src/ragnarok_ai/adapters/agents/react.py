"""ReAct pattern parser and adapter for agent evaluation.

This module provides tools to parse and evaluate ReAct-style agent outputs.

ReAct Format:
    Thought: I need to find information about X
    Action: search
    Action Input: {"query": "X"}
    Observation: [search results]
    Thought: Based on the results...
    Action: answer
    Action Input: {"response": "Y"}

Example:
    >>> from ragnarok_ai.adapters.agents import ReActParser, ReActAdapter
    >>>
    >>> # Parse raw output
    >>> parser = ReActParser()
    >>> steps = parser.parse(raw_output)
    >>>
    >>> # Wrap a ReAct agent
    >>> adapter = ReActAdapter(my_agent)
    >>> response = await adapter.query("What is X?")
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING, Any

from ragnarok_ai.agents.types import AgentResponse, AgentStep, ToolCall

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def _parse_action_input(raw_input: str) -> dict[str, Any]:
    """Parse action input string to dictionary.

    Handles various formats:
    - Valid JSON: {"query": "test"}
    - JSON with single quotes: {'query': 'test'}
    - Simple string: test query

    Args:
        raw_input: Raw action input string.

    Returns:
        Parsed dictionary. If parsing fails, returns {"raw": raw_input}.
    """
    raw_input = raw_input.strip()

    if not raw_input:
        return {}

    # Try standard JSON first
    try:
        result = json.loads(raw_input)
        if isinstance(result, dict):
            return result
        return {"value": result}
    except json.JSONDecodeError:
        pass

    # Try replacing single quotes with double quotes (Python dict style)
    try:
        # Replace single quotes, being careful with apostrophes
        converted = re.sub(r"'([^']*)':", r'"\1":', raw_input)
        converted = re.sub(r":\s*'([^']*)'", r': "\1"', converted)
        result = json.loads(converted)
        if isinstance(result, dict):
            return result
        return {"value": result}
    except json.JSONDecodeError:
        pass

    # Try extracting key-value pairs with regex
    # Pattern: key: value or key=value
    kv_pattern = r'(\w+)\s*[:=]\s*["\']?([^"\',:}]+)["\']?'
    matches = re.findall(kv_pattern, raw_input)
    if matches:
        return {k.strip(): v.strip() for k, v in matches}

    # Fallback: treat as raw string
    return {"raw": raw_input}


class ReActParser:
    """Parse ReAct-style text output to structured AgentStep list.

    Supports customizable prefixes for thought, action, action input,
    and observation sections.

    Attributes:
        thought_prefix: Prefix for thought lines.
        action_prefix: Prefix for action lines.
        action_input_prefix: Prefix for action input lines.
        observation_prefix: Prefix for observation lines.

    Example:
        >>> parser = ReActParser()
        >>> steps = parser.parse('''
        ... Thought: I need to search
        ... Action: search
        ... Action Input: {"query": "test"}
        ... Observation: Found results
        ... ''')
        >>> len(steps)
        3
    """

    def __init__(
        self,
        thought_prefix: str = "Thought:",
        action_prefix: str = "Action:",
        action_input_prefix: str = "Action Input:",
        observation_prefix: str = "Observation:",
        final_answer_prefix: str = "Final Answer:",
    ) -> None:
        """Initialize the parser.

        Args:
            thought_prefix: Prefix identifying thought lines.
            action_prefix: Prefix identifying action lines.
            action_input_prefix: Prefix identifying action input lines.
            observation_prefix: Prefix identifying observation lines.
            final_answer_prefix: Prefix identifying final answer lines.
        """
        self.thought_prefix = thought_prefix
        self.action_prefix = action_prefix
        self.action_input_prefix = action_input_prefix
        self.observation_prefix = observation_prefix
        self.final_answer_prefix = final_answer_prefix

    def parse(self, text: str) -> list[AgentStep]:
        """Parse ReAct trace into structured steps.

        Args:
            text: Raw ReAct-style output text.

        Returns:
            List of AgentStep objects representing the trace.
        """
        if not text or not text.strip():
            return []

        steps: list[AgentStep] = []
        lines = text.split("\n")

        current_thought: str | None = None
        current_action: str | None = None
        current_action_input: str | None = None
        collecting_observation = False
        observation_lines: list[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check for Final Answer
            if stripped.startswith(self.final_answer_prefix):
                # Flush pending observation
                if collecting_observation and observation_lines:
                    obs_content = "\n".join(observation_lines).strip()
                    if obs_content:
                        steps.append(AgentStep(step_type="observation", content=obs_content))
                        self._update_last_tool_output(steps, obs_content)
                    observation_lines = []
                    collecting_observation = False

                # Flush pending thought
                if current_thought:
                    steps.append(AgentStep(step_type="thought", content=current_thought))
                    current_thought = None

                answer = stripped[len(self.final_answer_prefix) :].strip()
                # Collect multiline final answer
                i += 1
                while i < len(lines) and not self._is_prefix_line(lines[i]):
                    answer += "\n" + lines[i]
                    i += 1
                steps.append(AgentStep(step_type="final_answer", content=answer.strip()))
                continue

            # Check for Thought
            if stripped.startswith(self.thought_prefix):
                # Flush pending observation
                if collecting_observation:
                    steps.append(AgentStep(step_type="observation", content="\n".join(observation_lines).strip()))
                    observation_lines = []
                    collecting_observation = False

                # Flush pending thought
                if current_thought:
                    steps.append(AgentStep(step_type="thought", content=current_thought))

                current_thought = stripped[len(self.thought_prefix) :].strip()
                i += 1
                continue

            # Check for Action (but not Action Input)
            if stripped.startswith(self.action_prefix) and not stripped.startswith(self.action_input_prefix):
                # Flush pending observation
                if collecting_observation:
                    steps.append(AgentStep(step_type="observation", content="\n".join(observation_lines).strip()))
                    observation_lines = []
                    collecting_observation = False

                current_action = stripped[len(self.action_prefix) :].strip()
                i += 1
                continue

            # Check for Action Input
            if stripped.startswith(self.action_input_prefix):
                current_action_input = stripped[len(self.action_input_prefix) :].strip()

                # Collect multiline input (for JSON that spans lines)
                i += 1
                while i < len(lines) and not self._is_prefix_line(lines[i]):
                    next_line = lines[i].strip()
                    if next_line:
                        current_action_input += " " + next_line
                    i += 1

                # Create action step with tool call
                if current_action:
                    # Flush pending thought first
                    if current_thought:
                        steps.append(AgentStep(step_type="thought", content=current_thought))
                        current_thought = None

                    parsed_input = _parse_action_input(current_action_input or "")
                    steps.append(
                        AgentStep(
                            step_type="action",
                            content=current_action,
                            tool_call=ToolCall(
                                name=current_action,
                                input=parsed_input,
                                output="",  # Will be filled by observation
                            ),
                        )
                    )
                    current_action = None
                    current_action_input = None
                continue

            # Check for Observation
            if stripped.startswith(self.observation_prefix):
                observation_content = stripped[len(self.observation_prefix) :].strip()
                observation_lines = [observation_content] if observation_content else []
                collecting_observation = True
                i += 1
                continue

            # Collecting multiline observation
            if collecting_observation:
                observation_lines.append(line)
                i += 1
                continue

            i += 1

        # Flush remaining items
        if collecting_observation and observation_lines:
            obs_content = "\n".join(observation_lines).strip()
            if obs_content:
                steps.append(AgentStep(step_type="observation", content=obs_content))
                # Update last action's tool call output
                self._update_last_tool_output(steps, obs_content)

        if current_thought:
            steps.append(AgentStep(step_type="thought", content=current_thought))

        return steps

    def _is_prefix_line(self, line: str) -> bool:
        """Check if line starts with any known prefix."""
        stripped = line.strip()
        return any(
            stripped.startswith(p)
            for p in [
                self.thought_prefix,
                self.action_prefix,
                self.observation_prefix,
                self.final_answer_prefix,
            ]
        )

    def _update_last_tool_output(self, steps: list[AgentStep], output: str) -> None:
        """Update the output of the last tool call in steps."""
        for i in range(len(steps) - 1, -1, -1):
            old_tc = steps[i].tool_call
            if old_tc is not None:
                # Create new ToolCall with updated output (ToolCall is frozen)
                new_tc = ToolCall(
                    name=old_tc.name,
                    input=old_tc.input,
                    output=output,
                    error=old_tc.error,
                    latency_ms=old_tc.latency_ms,
                )
                # Create new AgentStep with updated tool_call (AgentStep is frozen)
                steps[i] = AgentStep(
                    step_type=steps[i].step_type,
                    content=steps[i].content,
                    tool_call=new_tc,
                    latency_ms=steps[i].latency_ms,
                    metadata=steps[i].metadata,
                )
                break

    def parse_to_response(
        self,
        text: str,
        answer: str | None = None,
    ) -> AgentResponse:
        """Parse ReAct trace and wrap in AgentResponse.

        Args:
            text: Raw ReAct-style output text.
            answer: Optional explicit answer. If not provided, extracts
                   from Final Answer step or last observation.

        Returns:
            AgentResponse with parsed steps.
        """
        steps = self.parse(text)

        # Extract answer if not provided
        if answer is None:
            answer = self._extract_answer(steps, text)

        return AgentResponse(
            answer=answer,
            steps=steps,
        )

    def _extract_answer(self, steps: list[AgentStep], raw_text: str) -> str:
        """Extract answer from steps or raw text.

        Args:
            steps: Parsed steps.
            raw_text: Original raw text.

        Returns:
            Extracted answer string.
        """
        # Look for final_answer step
        for step in reversed(steps):
            if step.step_type == "final_answer":
                return step.content

        # Look for last observation
        for step in reversed(steps):
            if step.step_type == "observation":
                return step.content

        # Look for answer action
        for step in reversed(steps):
            if (
                step.tool_call
                and step.tool_call.name.lower() in ("answer", "respond", "reply")
                and step.tool_call.input
            ):
                return str(step.tool_call.input.get("response", step.tool_call.input.get("raw", "")))

        # Fallback: return last non-empty line
        for line in reversed(raw_text.split("\n")):
            if line.strip():
                return line.strip()

        return ""


class ReActAdapter:
    """Wrap a ReAct-style agent for evaluation.

    Executes the agent and parses its output into structured AgentResponse.

    Attributes:
        agent: The wrapped agent callable.
        parser: ReActParser instance for parsing output.

    Example:
        >>> async def my_agent(question: str) -> str:
        ...     return '''
        ...     Thought: I need to search
        ...     Action: search
        ...     Action Input: {"q": "test"}
        ...     Observation: Found it
        ...     Final Answer: The answer is X
        ...     '''
        >>>
        >>> adapter = ReActAdapter(my_agent)
        >>> response = await adapter.query("What is X?")
        >>> print(response.answer)
        The answer is X
    """

    def __init__(
        self,
        agent: Callable[[str], Awaitable[str]] | Callable[[str], str],
        parser: ReActParser | None = None,
        extract_answer: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            agent: Agent callable that takes a question and returns output.
                  Can be sync or async.
            parser: Optional custom ReActParser. Uses default if not provided.
            extract_answer: Optional function to extract answer from raw output.
        """
        self.agent = agent
        self.parser = parser or ReActParser()
        self.extract_answer = extract_answer

    async def query(self, question: str) -> AgentResponse:
        """Execute agent and return structured response.

        Args:
            question: The question to ask the agent.

        Returns:
            AgentResponse with parsed trajectory.
        """
        start_time = time.perf_counter()

        # Execute agent (handle sync/async)
        if asyncio.iscoroutinefunction(self.agent):
            raw_output = await self.agent(question)
        else:
            raw_output = await asyncio.to_thread(self.agent, question)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Extract answer if custom extractor provided
        answer = None
        if self.extract_answer:
            answer = self.extract_answer(raw_output)

        # Parse output
        response = self.parser.parse_to_response(raw_output, answer=answer)

        # Update latency
        return AgentResponse(
            answer=response.answer,
            steps=response.steps,
            total_latency_ms=elapsed_ms,
            metadata={
                "adapter": "react",
                "raw_output_length": len(raw_output),
            },
        )
