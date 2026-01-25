"""Relevance evaluator for ragnarok-ai.

This module provides LLM-based evaluation of relevance,
measuring if generated answers address the original question.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ragnarok_ai.core.exceptions import EvaluationError

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import LLMProtocol


class RelevanceResult(BaseModel):
    """Detailed result of relevance evaluation.

    Attributes:
        score: Relevance score between 0.0 and 1.0.
        reasoning: Explanation for the score.
        aspects_covered: List of question aspects addressed by the answer.
        aspects_missing: List of question aspects not addressed by the answer.
    """

    model_config = {"frozen": True}

    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    reasoning: str = Field(..., description="Explanation for the score")
    aspects_covered: list[str] = Field(default_factory=list, description="Question aspects addressed")
    aspects_missing: list[str] = Field(default_factory=list, description="Question aspects not addressed")


# Prompt for evaluating relevance
RELEVANCE_EVALUATION_PROMPT = """Evaluate if the following answer addresses the given question.

Question: {question}

Answer: {answer}

Analyze the answer and determine:
1. Does the answer address the question directly?
2. What aspects of the question are covered?
3. What aspects of the question are missing or not addressed?
4. Overall relevance score from 0.0 (completely irrelevant) to 1.0 (fully addresses the question)

Return a JSON object with:
- "score": a number between 0.0 and 1.0
- "reasoning": brief explanation for the score
- "aspects_covered": list of question aspects that are addressed
- "aspects_missing": list of question aspects that are not addressed

Only return the JSON object, nothing else."""


class RelevanceEvaluator:
    """LLM-based relevance evaluator.

    Measures if generated answers address the original question
    using an LLM-as-judge approach.

    The evaluation checks:
    - Does the answer directly address the question?
    - What aspects of the question are covered?
    - What aspects are missing?

    Implements EvaluatorProtocol for use in evaluation pipelines.

    Attributes:
        llm: The LLM provider for relevance evaluation.

    Example:
        >>> from ragnarok_ai.adapters import OllamaLLM
        >>> async with OllamaLLM() as llm:
        ...     evaluator = RelevanceEvaluator(llm)
        ...     score = await evaluator.evaluate(
        ...         response="Paris is the capital of France.",
        ...         query="What is the capital of France?"
        ...     )
        ...     print(f"Relevance: {score:.2f}")
    """

    def __init__(self, llm: LLMProtocol) -> None:
        """Initialize RelevanceEvaluator.

        Args:
            llm: The LLM provider implementing LLMProtocol.
        """
        self.llm = llm

    async def evaluate(
        self,
        response: str,
        context: str | None = None,  # noqa: ARG002
        query: str | None = None,
    ) -> float:
        """Evaluate relevance of a response to the original query.

        Args:
            response: The generated response to evaluate.
            context: Optional context (unused, for protocol compatibility).
            query: The original query/question to evaluate against.

        Returns:
            Relevance score between 0.0 and 1.0.

        Raises:
            EvaluationError: If evaluation fails or query is not provided.
        """
        if query is None:
            msg = "Query is required for relevance evaluation"
            raise EvaluationError(msg)

        result = await self.evaluate_detailed(response, query)
        return result.score

    async def evaluate_detailed(
        self,
        response: str,
        query: str,
    ) -> RelevanceResult:
        """Evaluate relevance with detailed results.

        Args:
            response: The generated response to evaluate.
            query: The original query/question.

        Returns:
            RelevanceResult with score, reasoning, and aspect analysis.

        Raises:
            EvaluationError: If evaluation fails.
        """
        # Handle empty response
        if not response.strip():
            return RelevanceResult(
                score=0.0,
                reasoning="Empty response cannot address the question.",
                aspects_covered=[],
                aspects_missing=["entire question"],
            )

        # Handle empty query
        if not query.strip():
            return RelevanceResult(
                score=1.0,
                reasoning="No question provided to evaluate against.",
                aspects_covered=[],
                aspects_missing=[],
            )

        # Evaluate relevance using LLM
        prompt = RELEVANCE_EVALUATION_PROMPT.format(question=query, answer=response)

        try:
            llm_response = await self.llm.generate(prompt)
            result = self._parse_json_object(llm_response)

            raw_score = result.get("score", 0.0)
            try:
                score = float(raw_score)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                score = 0.0
            # Clamp score to valid range
            score = max(0.0, min(1.0, score))

            reasoning = str(result.get("reasoning", "No reasoning provided."))

            aspects_covered = result.get("aspects_covered", [])
            if not isinstance(aspects_covered, list):
                aspects_covered = []
            aspects_covered = [str(a) for a in aspects_covered]

            aspects_missing = result.get("aspects_missing", [])
            if not isinstance(aspects_missing, list):
                aspects_missing = []
            aspects_missing = [str(a) for a in aspects_missing]

            return RelevanceResult(
                score=score,
                reasoning=reasoning,
                aspects_covered=aspects_covered,
                aspects_missing=aspects_missing,
            )
        except EvaluationError:
            raise
        except Exception as e:
            msg = f"Failed to evaluate relevance: {e}"
            raise EvaluationError(msg) from e

    def _parse_json_object(self, text: str) -> dict[str, object]:
        """Parse a JSON object from LLM response.

        Args:
            text: The LLM response text.

        Returns:
            Parsed dictionary.
        """
        text = text.strip()

        # Try direct parse first
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return {}
