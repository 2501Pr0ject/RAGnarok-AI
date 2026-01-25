"""Faithfulness evaluator for ragnarok-ai.

This module provides LLM-based evaluation of faithfulness,
measuring if generated answers are grounded in retrieved context.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ragnarok_ai.core.exceptions import EvaluationError

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import LLMProtocol


class ClaimVerification(BaseModel):
    """Verification result for a single claim.

    Attributes:
        claim: The extracted claim from the response.
        supported: Whether the claim is supported by the context.
        reasoning: Explanation for the verification decision.
    """

    model_config = {"frozen": True}

    claim: str = Field(..., description="The extracted claim")
    supported: bool = Field(..., description="Whether the claim is supported by context")
    reasoning: str = Field(..., description="Explanation for the decision")


class FaithfulnessResult(BaseModel):
    """Detailed result of faithfulness evaluation.

    Attributes:
        score: Faithfulness score between 0.0 and 1.0.
        claims: List of extracted claims with verification results.
        reasoning: Overall reasoning for the score.
    """

    model_config = {"frozen": True}

    score: float = Field(..., ge=0.0, le=1.0, description="Faithfulness score")
    claims: list[ClaimVerification] = Field(..., description="Claim verifications")
    reasoning: str = Field(..., description="Overall reasoning")


# Prompt for extracting claims from the response
CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the following response.
A claim is a single, atomic statement that can be verified as true or false.

Response: {response}

Return a JSON array of claims. Example format:
["Paris is the capital of France", "The Eiffel Tower is 330 meters tall"]

Only return the JSON array, nothing else."""

# Prompt for verifying claims against context
CLAIM_VERIFICATION_PROMPT = """Verify if the following claim is supported by the given context.

Claim: {claim}

Context: {context}

Answer with a JSON object containing:
- "supported": true if the claim is clearly supported by the context, false otherwise
- "reasoning": brief explanation for your decision

Only return the JSON object, nothing else."""


class FaithfulnessEvaluator:
    """LLM-based faithfulness evaluator.

    Measures if generated answers are grounded in retrieved context
    using an LLM-as-judge approach.

    The evaluation process:
    1. Extract claims from the generated response
    2. Verify each claim against the provided context
    3. Calculate score as: supported_claims / total_claims

    Implements EvaluatorProtocol for use in evaluation pipelines.

    Attributes:
        llm: The LLM provider for claim extraction and verification.

    Example:
        >>> from ragnarok_ai.adapters import OllamaLLM
        >>> async with OllamaLLM() as llm:
        ...     evaluator = FaithfulnessEvaluator(llm)
        ...     score = await evaluator.evaluate(
        ...         response="Paris is the capital of France.",
        ...         context="France is a country in Europe. Its capital is Paris."
        ...     )
        ...     print(f"Faithfulness: {score:.2f}")
    """

    def __init__(self, llm: LLMProtocol) -> None:
        """Initialize FaithfulnessEvaluator.

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
        """Evaluate faithfulness of a response to its context.

        Args:
            response: The generated response to evaluate.
            context: The retrieved context used for generation.
            query: Optional original query (unused, for protocol compatibility).

        Returns:
            Faithfulness score between 0.0 and 1.0.

        Raises:
            EvaluationError: If evaluation fails.
        """
        result = await self.evaluate_detailed(response, context)
        return result.score

    async def evaluate_detailed(
        self,
        response: str,
        context: str,
    ) -> FaithfulnessResult:
        """Evaluate faithfulness with detailed claim-level results.

        Args:
            response: The generated response to evaluate.
            context: The retrieved context used for generation.

        Returns:
            FaithfulnessResult with score, claims, and reasoning.

        Raises:
            EvaluationError: If evaluation fails.
        """
        # Handle empty response
        if not response.strip():
            return FaithfulnessResult(
                score=1.0,
                claims=[],
                reasoning="Empty response has no claims to verify.",
            )

        # Handle empty context
        if not context.strip():
            return FaithfulnessResult(
                score=0.0,
                claims=[],
                reasoning="No context provided to verify claims against.",
            )

        # Extract claims from response
        claims = await self._extract_claims(response)

        if not claims:
            return FaithfulnessResult(
                score=1.0,
                claims=[],
                reasoning="No verifiable claims found in the response.",
            )

        # Verify each claim against context
        verifications: list[ClaimVerification] = []
        for claim in claims:
            verification = await self._verify_claim(claim, context)
            verifications.append(verification)

        # Calculate score
        supported_count = sum(1 for v in verifications if v.supported)
        score = supported_count / len(verifications)

        # Generate overall reasoning
        if score == 1.0:
            reasoning = "All claims in the response are supported by the context."
        elif score == 0.0:
            reasoning = "None of the claims in the response are supported by the context."
        else:
            reasoning = f"{supported_count} out of {len(verifications)} claims are supported by the context."

        return FaithfulnessResult(
            score=score,
            claims=verifications,
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

    async def _verify_claim(self, claim: str, context: str) -> ClaimVerification:
        """Verify a single claim against context.

        Args:
            claim: The claim to verify.
            context: The context to verify against.

        Returns:
            ClaimVerification with supported status and reasoning.

        Raises:
            EvaluationError: If verification fails.
        """
        prompt = CLAIM_VERIFICATION_PROMPT.format(claim=claim, context=context)

        try:
            llm_response = await self.llm.generate(prompt)
            result = self._parse_json_object(llm_response)

            supported = bool(result.get("supported", False))
            reasoning = str(result.get("reasoning", "No reasoning provided."))

            return ClaimVerification(
                claim=claim,
                supported=supported,
                reasoning=reasoning,
            )
        except Exception as e:
            msg = f"Failed to verify claim '{claim}': {e}"
            raise EvaluationError(msg) from e

    def _parse_json_array(self, text: str) -> list[str]:
        """Parse a JSON array from LLM response.

        Args:
            text: The LLM response text.

        Returns:
            Parsed list of strings.
        """
        # Try to find JSON array in the response
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
        # Try to find JSON object in the response
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
