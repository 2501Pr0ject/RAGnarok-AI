"""Hallucination detector for ragnarok-ai.

This module provides LLM-based detection of hallucinations,
identifying fabricated or unsupported information in generated answers.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ragnarok_ai.core.exceptions import EvaluationError

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import LLMProtocol


class Hallucination(BaseModel):
    """A detected hallucination in the response.

    Attributes:
        claim: The hallucinated claim from the response.
        reason: Explanation of why this is considered a hallucination.
    """

    model_config = {"frozen": True}

    claim: str = Field(..., description="The hallucinated claim")
    reason: str = Field(..., description="Why this is a hallucination")


class HallucinationResult(BaseModel):
    """Detailed result of hallucination detection.

    Attributes:
        score: Hallucination score between 0.0 (no hallucination) and 1.0 (full hallucination).
        hallucinations: List of detected hallucinations with explanations.
        total_claims: Total number of claims extracted from the response.
        reasoning: Overall reasoning for the score.
    """

    model_config = {"frozen": True}

    score: float = Field(..., ge=0.0, le=1.0, description="Hallucination score (0=good, 1=bad)")
    hallucinations: list[Hallucination] = Field(default_factory=list, description="Detected hallucinations")
    total_claims: int = Field(default=0, ge=0, description="Total claims in response")
    reasoning: str = Field(..., description="Overall reasoning")


# Prompt for extracting claims from the response
CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the following response.
A claim is a single, atomic statement that can be verified as true or false.

Response: {response}

Return a JSON array of claims. Example format:
["Paris is the capital of France", "The Eiffel Tower is 330 meters tall"]

Only return the JSON array, nothing else."""

# Prompt for detecting hallucinations in claims
HALLUCINATION_DETECTION_PROMPT = """Analyze if the following claim is a hallucination.

A claim is a hallucination if:
1. It is NOT supported by the given context (fabricated information)
2. It contradicts the context (false statement)

Claim: {claim}

Context: {context}

Answer with a JSON object containing:
- "is_hallucination": true if the claim is a hallucination, false if it's supported by the context
- "reason": brief explanation for your decision

Only return the JSON object, nothing else."""


class HallucinationDetector:
    """LLM-based hallucination detector.

    Detects fabricated or unsupported information in generated answers
    using an LLM-as-judge approach.

    The detection process:
    1. Extract claims from the generated response
    2. Check each claim against the provided context
    3. Identify claims not supported by or contradicting the context
    4. Calculate score as: hallucinated_claims / total_claims

    A score of 0.0 means no hallucinations (good), 1.0 means all claims
    are hallucinations (bad).

    Implements EvaluatorProtocol for use in evaluation pipelines.

    Attributes:
        llm: The LLM provider for claim extraction and verification.

    Example:
        >>> from ragnarok_ai.adapters import OllamaLLM
        >>> async with OllamaLLM() as llm:
        ...     detector = HallucinationDetector(llm)
        ...     score = await detector.evaluate(
        ...         response="Paris, founded in 500 BC, is the capital of France.",
        ...         context="France is a country in Europe. Its capital is Paris."
        ...     )
        ...     print(f"Hallucination rate: {score:.2f}")
    """

    def __init__(self, llm: LLMProtocol) -> None:
        """Initialize HallucinationDetector.

        Args:
            llm: The LLM provider implementing LLMProtocol.
        """
        self.llm = llm

    async def evaluate(
        self,
        response: str,
        context: str,
        query: str | None = None,  # noqa: ARG002
    ) -> float:
        """Evaluate hallucination rate of a response.

        Args:
            response: The generated response to evaluate.
            context: The retrieved context used for generation.
            query: Optional original query (unused, for protocol compatibility).

        Returns:
            Hallucination score between 0.0 (no hallucinations) and 1.0 (all hallucinations).

        Raises:
            EvaluationError: If evaluation fails.
        """
        result = await self.evaluate_detailed(response, context)
        return result.score

    async def evaluate_detailed(
        self,
        response: str,
        context: str,
    ) -> HallucinationResult:
        """Detect hallucinations with detailed results.

        Args:
            response: The generated response to evaluate.
            context: The retrieved context used for generation.

        Returns:
            HallucinationResult with score, hallucinations list, and reasoning.

        Raises:
            EvaluationError: If evaluation fails.
        """
        # Handle empty response
        if not response.strip():
            return HallucinationResult(
                score=0.0,
                hallucinations=[],
                total_claims=0,
                reasoning="Empty response has no claims to check for hallucinations.",
            )

        # Handle empty context
        if not context.strip():
            return HallucinationResult(
                score=1.0,
                hallucinations=[],
                total_claims=0,
                reasoning="No context provided - all claims are potentially hallucinations.",
            )

        # Extract claims from response
        claims = await self._extract_claims(response)

        if not claims:
            return HallucinationResult(
                score=0.0,
                hallucinations=[],
                total_claims=0,
                reasoning="No verifiable claims found in the response.",
            )

        # Check each claim for hallucination
        hallucinations: list[Hallucination] = []
        for claim in claims:
            result = await self._check_hallucination(claim, context)
            if result is not None:
                hallucinations.append(result)

        # Calculate score
        score = len(hallucinations) / len(claims)

        # Generate overall reasoning
        if not hallucinations:
            reasoning = "No hallucinations detected. All claims are supported by the context."
        elif len(hallucinations) == len(claims):
            reasoning = "All claims in the response are hallucinations (not supported by context)."
        else:
            reasoning = f"{len(hallucinations)} out of {len(claims)} claims are hallucinations."

        return HallucinationResult(
            score=score,
            hallucinations=hallucinations,
            total_claims=len(claims),
            reasoning=reasoning,
        )

    async def _extract_claims(self, response: str) -> list[str]:
        """Extract factual claims from a response.

        Args:
            response: The response to extract claims from.

        Returns:
            List of extracted claims.

        Raises:
            EvaluationError: If claim extraction fails.
        """
        prompt = CLAIM_EXTRACTION_PROMPT.format(response=response)

        try:
            llm_response = await self.llm.generate(prompt)
            claims = self._parse_json_array(llm_response)
            return [str(claim) for claim in claims if claim]
        except Exception as e:
            msg = f"Failed to extract claims: {e}"
            raise EvaluationError(msg) from e

    async def _check_hallucination(self, claim: str, context: str) -> Hallucination | None:
        """Check if a claim is a hallucination.

        Args:
            claim: The claim to check.
            context: The context to verify against.

        Returns:
            Hallucination object if the claim is a hallucination, None otherwise.

        Raises:
            EvaluationError: If verification fails.
        """
        prompt = HALLUCINATION_DETECTION_PROMPT.format(claim=claim, context=context)

        try:
            llm_response = await self.llm.generate(prompt)
            result = self._parse_json_object(llm_response)

            is_hallucination = bool(result.get("is_hallucination", False))
            reason = str(result.get("reason", "No reason provided."))

            if is_hallucination:
                return Hallucination(claim=claim, reason=reason)
            return None
        except Exception as e:
            msg = f"Failed to check hallucination for claim '{claim}': {e}"
            raise EvaluationError(msg) from e

    def _parse_json_array(self, text: str) -> list[str]:
        """Parse a JSON array from LLM response.

        Args:
            text: The LLM response text.

        Returns:
            Parsed list of strings.
        """
        text = text.strip()

        # Try direct parse first
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try to extract JSON array from text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        return []

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
