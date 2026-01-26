"""Chain-of-Thought adapter for agent evaluation.

This module provides an adapter for evaluating Chain-of-Thought reasoning.

Example:
    >>> from ragnarok_ai.adapters.agents import ChainOfThoughtAdapter
    >>>
    >>> adapter = ChainOfThoughtAdapter(llm)
    >>> response = await adapter.query("What is 2+2*3?")
    >>> print(response.reasoning_trace)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ragnarok_ai.agents.types import AgentResponse, AgentStep

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import LLMProtocol


class ChainOfThoughtAdapter:
    """Wrap an LLM for Chain-of-Thought evaluation.

    Prompts the LLM to reason step by step and parses the response
    into structured AgentSteps.

    Attributes:
        llm: The LLM provider.
        cot_prompt: Prompt to trigger step-by-step reasoning.
        step_separator: String used to separate reasoning steps.

    Example:
        >>> adapter = ChainOfThoughtAdapter(
        ...     llm,
        ...     cot_prompt="Let's solve this step by step:",
        ... )
        >>> response = await adapter.query("What is 15% of 80?")
        >>> for step in response.steps:
        ...     print(f"{step.step_type}: {step.content[:50]}...")
    """

    def __init__(
        self,
        llm: LLMProtocol,
        cot_prompt: str = "Let's think step by step.",
        step_separator: str = "\n\n",
        answer_prefix: str | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            llm: LLM provider implementing LLMProtocol.
            cot_prompt: Prompt appended to trigger CoT reasoning.
            step_separator: String that separates reasoning steps.
            answer_prefix: Optional prefix that marks the final answer.
                          If found, content after it becomes the answer.
        """
        self.llm = llm
        self.cot_prompt = cot_prompt
        self.step_separator = step_separator
        self.answer_prefix = answer_prefix

    async def query(self, question: str) -> AgentResponse:
        """Execute Chain-of-Thought reasoning and return response.

        Args:
            question: The question to reason about.

        Returns:
            AgentResponse with thought steps and final answer.
        """
        start_time = time.perf_counter()

        # Build prompt with CoT trigger
        prompt = f"{question}\n\n{self.cot_prompt}"

        # Generate response
        raw_response = await self.llm.generate(prompt)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Parse into steps
        steps, answer = self._parse_cot_response(raw_response)

        return AgentResponse(
            answer=answer,
            steps=steps,
            total_latency_ms=elapsed_ms,
            metadata={
                "adapter": "chain_of_thought",
                "cot_prompt": self.cot_prompt,
                "raw_response_length": len(raw_response),
            },
        )

    def _parse_cot_response(self, response: str) -> tuple[list[AgentStep], str]:
        """Parse CoT response into thought steps.

        Args:
            response: Raw LLM response.

        Returns:
            Tuple of (list of AgentSteps, final answer).
        """
        if not response or not response.strip():
            return [], ""

        # Check for explicit answer prefix
        answer = ""
        reasoning_text = response

        if self.answer_prefix:
            prefix_lower = self.answer_prefix.lower()
            response_lower = response.lower()
            if prefix_lower in response_lower:
                idx = response_lower.find(prefix_lower)
                reasoning_text = response[:idx]
                answer = response[idx + len(self.answer_prefix) :].strip()

        # Split into paragraphs/steps
        if self.step_separator in reasoning_text:
            paragraphs = reasoning_text.split(self.step_separator)
        else:
            # Fallback: split by numbered steps or sentences
            paragraphs = self._split_into_steps(reasoning_text)

        steps: list[AgentStep] = []
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If no explicit answer and this is the last paragraph
            is_last = i == len(paragraphs) - 1
            if is_last and not answer:
                # Check if this looks like a conclusion
                if self._looks_like_conclusion(paragraph):
                    answer = self._extract_answer_from_conclusion(paragraph)
                    steps.append(AgentStep(step_type="final_answer", content=paragraph))
                else:
                    answer = paragraph
                    steps.append(AgentStep(step_type="final_answer", content=paragraph))
            else:
                steps.append(AgentStep(step_type="thought", content=paragraph))

        # If still no answer, use last step content
        if not answer and steps:
            answer = steps[-1].content

        return steps, answer

    def _split_into_steps(self, text: str) -> list[str]:
        """Split text into steps when no clear separator.

        Args:
            text: Text to split.

        Returns:
            List of step strings.
        """
        # Try to split by numbered steps (1. 2. 3. or Step 1: Step 2:)
        numbered_pattern = r"(?:^|\n)\s*(?:\d+[.):]|Step\s+\d+[.:])"
        import re

        parts = re.split(numbered_pattern, text, flags=re.IGNORECASE)

        # Filter empty parts
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) > 1:
            return parts

        # Fallback: split by sentence boundaries for long responses
        if len(text) > 200:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            # Group sentences into chunks of ~2-3 sentences
            chunks: list[str] = []
            current_chunk: list[str] = []
            for sentence in sentences:
                current_chunk.append(sentence)
                if len(current_chunk) >= 2:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            if len(chunks) > 1:
                return chunks

        # Return as single step
        return [text]

    def _looks_like_conclusion(self, text: str) -> bool:
        """Check if text looks like a conclusion/answer.

        Args:
            text: Text to check.

        Returns:
            True if text appears to be a conclusion.
        """
        conclusion_indicators = [
            "therefore",
            "thus",
            "so the answer",
            "the answer is",
            "in conclusion",
            "finally",
            "hence",
            "the result is",
            "we get",
            "this gives us",
            "equals",
            "=",
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in conclusion_indicators)

    def _extract_answer_from_conclusion(self, text: str) -> str:
        """Extract the actual answer from a conclusion paragraph.

        Args:
            text: Conclusion text.

        Returns:
            Extracted answer.
        """
        import re

        # Look for patterns like "the answer is X" or "= X"
        patterns = [
            r"(?:the\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)",
            r"(?:equals?|=)\s*(.+?)(?:\.|$)",
            r"result\s+is[:\s]+(.+?)(?:\.|$)",
            r"we\s+get[:\s]+(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Return the whole text as answer
        return text
