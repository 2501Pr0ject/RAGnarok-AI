"""LLM-as-Judge evaluator using Prometheus 2.

This module provides multi-criteria evaluation of RAG responses
using Prometheus 2, a specialized open-source judge model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import LLMProtocol

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus 2 Prompts (Rubric-based format)
# =============================================================================

FAITHFULNESS_PROMPT = """###Task Description:
Evaluate if the answer is faithful to the given context.

###Context:
{context}

###Question:
{question}

###Answer to evaluate:
{answer}

###Evaluation Criteria:
[Faithfulness] Does the answer contain only information supported by the context?
Score 1: Completely unfaithful - contradicts or fabricates information
Score 2: Mostly unfaithful - significant unsupported claims
Score 3: Partially faithful - mix of supported and unsupported claims
Score 4: Mostly faithful - minor unsupported details
Score 5: Completely faithful - all claims supported by context

###Output Format:
Score: [1-5]
Verdict: [PASS if score >= 4, FAIL if score <= 2, PARTIAL otherwise]
Explanation: [Detailed reasoning for the score]

###Evaluation:"""

HALLUCINATION_PROMPT = """###Task Description:
Detect if the answer contains hallucinations (claims not supported by context).

###Context:
{context}

###Answer to evaluate:
{answer}

###Evaluation Criteria:
[Hallucination Detection] Are there factual claims not grounded in the context?
Score 1: Severe hallucinations - multiple fabricated facts
Score 2: Significant hallucinations - key claims unsupported
Score 3: Minor hallucinations - some details not in context
Score 4: Negligible hallucinations - trivial unsupported details only
Score 5: No hallucinations - every claim traceable to context

###Output Format:
Score: [1-5]
Verdict: [PASS if score >= 4, FAIL if score <= 2, PARTIAL otherwise]
Explanation: [List specific hallucinations found, if any]

###Evaluation:"""

RELEVANCE_PROMPT = """###Task Description:
Evaluate if the answer is relevant to the question asked.

###Question:
{question}

###Answer to evaluate:
{answer}

###Evaluation Criteria:
[Relevance] Does the answer address what was asked?
Score 1: Completely irrelevant - does not address the question
Score 2: Mostly irrelevant - misses the main point
Score 3: Partially relevant - addresses some aspects
Score 4: Mostly relevant - addresses main point with minor gaps
Score 5: Completely relevant - fully addresses the question

###Output Format:
Score: [1-5]
Verdict: [PASS if score >= 4, FAIL if score <= 2, PARTIAL otherwise]
Explanation: [Reasoning for relevance assessment]

###Evaluation:"""

COMPLETENESS_PROMPT = """###Task Description:
Evaluate if the answer is complete given the available context.

###Context:
{context}

###Question:
{question}

###Answer to evaluate:
{answer}

###Evaluation Criteria:
[Completeness] Does the answer cover all relevant information from the context?
Score 1: Very incomplete - misses most relevant information
Score 2: Incomplete - misses significant information
Score 3: Partially complete - covers main points but misses details
Score 4: Mostly complete - covers most relevant information
Score 5: Fully complete - comprehensive answer using all relevant context

###Output Format:
Score: [1-5]
Verdict: [PASS if score >= 4, FAIL if score <= 2, PARTIAL otherwise]
Explanation: [What was covered and what was missed]

###Evaluation:"""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class JudgeResult:
    """Result from LLM judge evaluation.

    Attributes:
        verdict: PASS, FAIL, or PARTIAL verdict.
        score: Normalized score from 0.0 to 1.0.
        explanation: Detailed reasoning for the score.
        criteria: The evaluation criteria used.
        raw_score: Original 1-5 score from Prometheus.
    """

    verdict: Literal["PASS", "FAIL", "PARTIAL"]
    score: float  # 0.0 - 1.0 (normalized from 1-5)
    explanation: str
    criteria: str
    raw_score: int  # 1-5 original


@dataclass(frozen=True)
class JudgeResults:
    """Combined results from all evaluation criteria.

    Attributes:
        faithfulness: Result from faithfulness evaluation.
        relevance: Result from relevance evaluation.
        hallucination: Result from hallucination detection.
        completeness: Result from completeness evaluation.
        overall_score: Average normalized score across all criteria.
        overall_verdict: Combined verdict (FAIL if any fails, PARTIAL if any partial).
    """

    faithfulness: JudgeResult
    relevance: JudgeResult
    hallucination: JudgeResult
    completeness: JudgeResult

    @property
    def overall_score(self) -> float:
        """Average score across all criteria."""
        scores = [
            self.faithfulness.score,
            self.relevance.score,
            self.hallucination.score,
            self.completeness.score,
        ]
        return sum(scores) / len(scores)

    @property
    def overall_verdict(self) -> Literal["PASS", "FAIL", "PARTIAL"]:
        """Combined verdict based on all criteria."""
        verdicts = [
            self.faithfulness.verdict,
            self.relevance.verdict,
            self.hallucination.verdict,
            self.completeness.verdict,
        ]
        if "FAIL" in verdicts:
            return "FAIL"
        if "PARTIAL" in verdicts:
            return "PARTIAL"
        return "PASS"

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "faithfulness": {
                "verdict": self.faithfulness.verdict,
                "score": self.faithfulness.score,
                "raw_score": self.faithfulness.raw_score,
                "explanation": self.faithfulness.explanation,
            },
            "relevance": {
                "verdict": self.relevance.verdict,
                "score": self.relevance.score,
                "raw_score": self.relevance.raw_score,
                "explanation": self.relevance.explanation,
            },
            "hallucination": {
                "verdict": self.hallucination.verdict,
                "score": self.hallucination.score,
                "raw_score": self.hallucination.raw_score,
                "explanation": self.hallucination.explanation,
            },
            "completeness": {
                "verdict": self.completeness.verdict,
                "score": self.completeness.score,
                "raw_score": self.completeness.raw_score,
                "explanation": self.completeness.explanation,
            },
            "overall": {
                "verdict": self.overall_verdict,
                "score": self.overall_score,
            },
        }


# =============================================================================
# Response Parser
# =============================================================================

def parse_judge_response(response: str, criteria: str) -> JudgeResult:
    """Parse Prometheus 2 response into JudgeResult.

    Args:
        response: Raw response from the judge model.
        criteria: The evaluation criteria name.

    Returns:
        Parsed JudgeResult with score, verdict, and explanation.
    """
    # Extract score
    score_match = re.search(r"Score:\s*(\d)", response)
    raw_score = int(score_match.group(1)) if score_match else 3

    # Clamp score to valid range
    raw_score = max(1, min(5, raw_score))

    # Extract verdict
    verdict_match = re.search(r"Verdict:\s*(PASS|FAIL|PARTIAL)", response, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
    else:
        # Infer verdict from score
        if raw_score >= 4:
            verdict = "PASS"
        elif raw_score <= 2:
            verdict = "FAIL"
        else:
            verdict = "PARTIAL"

    # Extract explanation
    explanation_match = re.search(r"Explanation:\s*(.+)", response, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else response.strip()

    # Clean up explanation (remove trailing artifacts)
    explanation = explanation.split("###")[0].strip()

    return JudgeResult(
        verdict=verdict,  # type: ignore[arg-type]
        score=(raw_score - 1) / 4,  # Normalize 1-5 to 0-1
        explanation=explanation,
        criteria=criteria,
        raw_score=raw_score,
    )


# =============================================================================
# LLM Judge
# =============================================================================

class LLMJudge:
    """Multi-criteria LLM-as-Judge using Prometheus 2.

    Provides comprehensive evaluation of RAG responses across multiple
    quality dimensions: faithfulness, relevance, hallucination, and completeness.

    Attributes:
        llm: The LLM provider for judge evaluations.
        model: Model name being used.

    Example:
        >>> from ragnarok_ai.evaluators import LLMJudge
        >>> judge = LLMJudge()  # Uses Prometheus 2 by default
        >>> result = await judge.evaluate_faithfulness(
        ...     context="Paris is the capital of France.",
        ...     question="What is the capital of France?",
        ...     answer="Paris is the capital of France."
        ... )
        >>> print(result.verdict)
        'PASS'

    Installation:
        Install Prometheus 2 (Q5_K_M quantized, ~4.8GB) via Ollama:
        >>> ollama pull hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M

        Or create a shorter alias:
        >>> ollama cp hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M prometheus2
    """

    # Prometheus 2 7B Q5_K_M from HuggingFace (~4.8GB, fits 16GB RAM)
    # Q5 provides better precision for subtle judgments (hallucination, faithfulness)
    DEFAULT_MODEL = "hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M"
    # Alternative names users might create
    PROMETHEUS_ALIASES = ["prometheus2", "prometheus", "prometheus-7b"]
    FALLBACK_MODELS = ["mistral", "llama3.2", "llama3", "llama2"]

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        llm: LLMProtocol | None = None,
    ) -> None:
        """Initialize LLMJudge.

        Args:
            model: Model name for Ollama. Defaults to Prometheus 2.
            base_url: Ollama API base URL.
            timeout: Request timeout in seconds. Default 300s for thorough evaluations.
            llm: Optional pre-configured LLM instance. If provided, model/base_url are ignored.
        """
        if llm is not None:
            self.llm = llm
            self.model = getattr(llm, "model", "custom")
        else:
            from ragnarok_ai.adapters.llm.ollama import OllamaLLM

            selected_model = model or self.DEFAULT_MODEL

            # Check model availability and fallback if needed
            available = self._get_available_models(base_url)
            if selected_model not in available:
                fallback = self._find_fallback_model(available)
                if fallback:
                    logger.warning(
                        f"Model '{selected_model}' not found. Falling back to '{fallback}'. "
                        f"For best results, install Prometheus 2: ollama pull {self.DEFAULT_MODEL}"
                    )
                    selected_model = fallback
                else:
                    msg = (
                        f"No suitable judge model found. "
                        f"Install Prometheus 2 with: ollama pull {self.DEFAULT_MODEL}"
                    )
                    raise RuntimeError(msg)

            self.llm = OllamaLLM(
                model=selected_model,
                base_url=base_url,
                timeout=timeout,
            )
            self.model = selected_model

    def _get_available_models(self, base_url: str) -> list[str]:
        """Get list of available Ollama models.

        Args:
            base_url: Ollama API base URL.

        Returns:
            List of model names.
        """
        import httpx

        try:
            response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def _find_fallback_model(self, available: list[str]) -> str | None:
        """Find a fallback model from available models.

        Prioritizes Prometheus aliases, then falls back to general models.

        Args:
            available: List of available model names.

        Returns:
            Fallback model name or None if no suitable model found.
        """
        # First, check for Prometheus aliases (user might have created one)
        for alias in self.PROMETHEUS_ALIASES:
            for model in available:
                if alias in model.lower():
                    return model

        # Then check for general fallback models
        for fallback in self.FALLBACK_MODELS:
            for model in available:
                if fallback in model.lower():
                    return model
        return None

    async def evaluate_faithfulness(
        self,
        context: str,
        question: str,
        answer: str,
    ) -> JudgeResult:
        """Evaluate if the answer is faithful to the context.

        Args:
            context: The source context/documents.
            question: The user question.
            answer: The generated answer to evaluate.

        Returns:
            JudgeResult with faithfulness assessment.
        """
        prompt = FAITHFULNESS_PROMPT.format(
            context=context,
            question=question,
            answer=answer,
        )
        response = await self.llm.generate(prompt)
        return parse_judge_response(response, "faithfulness")

    async def evaluate_relevance(
        self,
        question: str,
        answer: str,
    ) -> JudgeResult:
        """Evaluate if the answer is relevant to the question.

        Args:
            question: The user question.
            answer: The generated answer to evaluate.

        Returns:
            JudgeResult with relevance assessment.
        """
        prompt = RELEVANCE_PROMPT.format(
            question=question,
            answer=answer,
        )
        response = await self.llm.generate(prompt)
        return parse_judge_response(response, "relevance")

    async def detect_hallucination(
        self,
        context: str,
        answer: str,
    ) -> JudgeResult:
        """Detect hallucinations in the answer.

        Args:
            context: The source context/documents.
            answer: The generated answer to evaluate.

        Returns:
            JudgeResult with hallucination detection results.
        """
        prompt = HALLUCINATION_PROMPT.format(
            context=context,
            answer=answer,
        )
        response = await self.llm.generate(prompt)
        return parse_judge_response(response, "hallucination")

    async def evaluate_completeness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> JudgeResult:
        """Evaluate if the answer is complete.

        Args:
            question: The user question.
            answer: The generated answer to evaluate.
            context: The source context/documents.

        Returns:
            JudgeResult with completeness assessment.
        """
        prompt = COMPLETENESS_PROMPT.format(
            context=context,
            question=question,
            answer=answer,
        )
        response = await self.llm.generate(prompt)
        return parse_judge_response(response, "completeness")

    async def evaluate_all(
        self,
        context: str,
        question: str,
        answer: str,
    ) -> JudgeResults:
        """Run all evaluation criteria.

        Args:
            context: The source context/documents.
            question: The user question.
            answer: The generated answer to evaluate.

        Returns:
            JudgeResults with all evaluation results.
        """
        faithfulness = await self.evaluate_faithfulness(context, question, answer)
        relevance = await self.evaluate_relevance(question, answer)
        hallucination = await self.detect_hallucination(context, answer)
        completeness = await self.evaluate_completeness(question, answer, context)

        return JudgeResults(
            faithfulness=faithfulness,
            relevance=relevance,
            hallucination=hallucination,
            completeness=completeness,
        )

    async def evaluate_batch(
        self,
        items: list[dict],
    ) -> list[JudgeResults]:
        """Evaluate a batch of items.

        Args:
            items: List of dicts with 'context', 'question', 'answer' keys.

        Returns:
            List of JudgeResults for each item.
        """
        results = []
        for item in items:
            result = await self.evaluate_all(
                context=item["context"],
                question=item["question"],
                answer=item["answer"],
            )
            results.append(result)
        return results
