"""LLM-as-Judge evaluator using Prometheus 2.

This module provides multi-criteria evaluation of RAG responses
using Prometheus 2, a specialized open-source judge model.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import re
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import LLMProtocol

from ragnarok_ai.core.exceptions import (
    JudgeConnectionError,
    JudgeDependencyError,
    JudgeModelNotFoundError,
    JudgeParseError,
)
from ragnarok_ai.judges.contract import (
    CallTrace,
    JudgeConfig,
    JudgeOutput,
    OrdinalBands,
    OrdinalScore,
    UnitJudgment,
    canonical_json,
)

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
# Prometheus 2 native prompts
# =============================================================================
# Prometheus 2 is trained on its own absolute-grading format and does not
# follow the "Score: N" output format above: it answers
# "(feedback) [RESULT] N". The system prompt and the no-reference templates
# below are copied verbatim from prometheus-eval 0.1.20
# (prometheus_eval/prompts.py: ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT_WO_REF,
# SCORE_RUBRIC_TEMPLATE), https://github.com/prometheus-eval/prometheus-eval,
# Apache License 2.0.

PROMETHEUS_ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

PROMETHEUS_ABSOLUTE_PROMPT_WO_REF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """

PROMETHEUS_SCORE_RUBRIC_TEMPLATE = """
[{criteria}]
Score 1: {score1_description}
Score 2: {score2_description}
Score 3: {score3_description}
Score 4: {score4_description}
Score 5: {score5_description}
""".strip()

# Score rubrics per criterion (same criteria and score levels as the prompts above)
PROMETHEUS_RUBRICS: dict[str, dict[str, str]] = {
    "faithfulness": {
        "criteria": "Does the answer contain only information supported by the context?",
        "score1_description": "Completely unfaithful - contradicts or fabricates information",
        "score2_description": "Mostly unfaithful - significant unsupported claims",
        "score3_description": "Partially faithful - mix of supported and unsupported claims",
        "score4_description": "Mostly faithful - minor unsupported details",
        "score5_description": "Completely faithful - all claims supported by context",
    },
    "hallucination": {
        "criteria": "Are there factual claims not grounded in the context?",
        "score1_description": "Severe hallucinations - multiple fabricated facts",
        "score2_description": "Significant hallucinations - key claims unsupported",
        "score3_description": "Minor hallucinations - some details not in context",
        "score4_description": "Negligible hallucinations - trivial unsupported details only",
        "score5_description": "No hallucinations - every claim traceable to context",
    },
    "relevance": {
        "criteria": "Does the answer address what was asked?",
        "score1_description": "Completely irrelevant - does not address the question",
        "score2_description": "Mostly irrelevant - misses the main point",
        "score3_description": "Partially relevant - addresses some aspects",
        "score4_description": "Mostly relevant - addresses main point with minor gaps",
        "score5_description": "Completely relevant - fully addresses the question",
    },
    "completeness": {
        "criteria": "Does the answer cover all relevant information from the context?",
        "score1_description": "Very incomplete - misses most relevant information",
        "score2_description": "Incomplete - misses significant information",
        "score3_description": "Partially complete - covers main points but misses details",
        "score4_description": "Mostly complete - covers most relevant information",
        "score5_description": "Fully complete - comprehensive answer using all relevant context",
    },
}


def build_prometheus_prompt(criteria: str, instruction: str, response: str) -> str:
    """Build a Prometheus 2 absolute-grading prompt (no reference answer).

    Args:
        criteria: The evaluation criteria name (a key of PROMETHEUS_RUBRICS).
        instruction: What the answer responds to (question, with context when relevant).
        response: The answer to evaluate.

    Returns:
        System prompt followed by the filled prometheus-eval template.
    """
    rubric = PROMETHEUS_SCORE_RUBRIC_TEMPLATE.format(**PROMETHEUS_RUBRICS[criteria])
    user = PROMETHEUS_ABSOLUTE_PROMPT_WO_REF.format(instruction=instruction, response=response, rubric=rubric)
    return f"{PROMETHEUS_ABS_SYSTEM_PROMPT}\n\n{user}"


# =============================================================================
# Truncation heuristic
# =============================================================================


def looks_truncated(prompt_eval_count: int, num_ctx: int) -> bool:
    """Whether Ollama's reported prompt token count betrays a truncated prompt.

    Ollama does not report truncation. When the prompt exceeds ``num_ctx`` it
    keeps a few leading tokens plus the last half of the window, so the
    reported ``prompt_eval_count`` lands just above ``num_ctx // 2``
    (measured on Ollama 0.40.13: 259 for 512, 515 for 1024, 1027 for 2048,
    2051 for 4096). A count at the very top of the window is flagged too, in
    case a server version truncates to the window instead. A legitimate
    prompt whose length falls in one of these narrow bands is a false
    positive, hence "probably" in the warning.
    """
    half = num_ctx // 2
    return (half - 8 <= prompt_eval_count <= half + 16) or prompt_eval_count >= num_ctx - 16


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
        output: The judge's native output (``OrdinalScore`` for ``LLMJudge``).
        config: The run configuration that produced the result.
        trace: What the call cost and reported.
        units: Unit judgments (sentences, claims) when the judge works below
            answer level; ``score`` and ``verdict`` stay at answer level.
    """

    verdict: Literal["PASS", "FAIL", "PARTIAL"]
    score: float  # 0.0 - 1.0 (normalized from 1-5)
    explanation: str
    criteria: str
    raw_score: int  # 1-5 original
    output: JudgeOutput | None = None
    config: JudgeConfig | None = None
    trace: CallTrace | None = None
    units: tuple[UnitJudgment, ...] | None = None

    def to_dict(self, detailed: bool = False) -> dict[str, Any]:
        """Dictionary view: the five historical fields, plus the contract fields when *detailed*."""
        data: dict[str, Any] = {
            "verdict": self.verdict,
            "score": self.score,
            "raw_score": self.raw_score,
            "explanation": self.explanation,
            "criteria": self.criteria,
        }
        if detailed:
            from ragnarok_ai.judges.contract import _jsonable

            data["output"] = _jsonable(self.output) if self.output is not None else None
            data["config"] = self.config.to_dict() if self.config is not None else None
            data["trace"] = _jsonable(self.trace) if self.trace is not None else None
            data["units"] = [_jsonable(u) for u in self.units] if self.units is not None else None
        return data


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

    def to_dict(self) -> dict[str, Any]:
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


def verdict_from_raw_score(raw_score: int) -> Literal["PASS", "FAIL", "PARTIAL"]:
    """Map a 1-5 score to a verdict: 4-5 PASS, 3 PARTIAL, 1-2 FAIL.

    This is the only verdict rule. Both prompt formats use it, so a verdict
    can always be recomputed from ``JudgeResult.raw_score``.
    """
    if raw_score >= 4:
        return "PASS"
    if raw_score <= 2:
        return "FAIL"
    return "PARTIAL"


def parse_judge_response(response: str, criteria: str) -> JudgeResult:
    """Parse a "Score: N" judge response into JudgeResult.

    Args:
        response: Raw response from the judge model.
        criteria: The evaluation criteria name.

    Returns:
        Parsed JudgeResult with score, verdict, and explanation. The verdict
        is derived from the score (see ``verdict_from_raw_score``), never
        read from the response.

    Raises:
        EvaluationError: If the response contains no "Score: N". No default
            score is ever substituted.
    """
    # Extract score
    score_match = re.search(r"Score:\s*(\d)", response)
    if score_match is None:
        msg = f"Unparseable {criteria} judgment: no 'Score: N' in the judge response"
        raise JudgeParseError(msg, criterion=criteria, raw_output=response)
    raw_score = int(score_match.group(1))

    # Clamp score to valid range
    raw_score = max(1, min(5, raw_score))

    # The verdict is always derived from the score. A "Verdict:" line written
    # by the model is ignored: when it contradicts the score, the score wins.
    verdict = verdict_from_raw_score(raw_score)

    # Extract explanation
    explanation_match = re.search(r"Explanation:\s*(.+)", response, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else response.strip()

    # Clean up explanation (remove trailing artifacts)
    explanation = explanation.split("###")[0].strip()

    return JudgeResult(
        verdict=verdict,
        score=(raw_score - 1) / 4,  # Normalize 1-5 to 0-1
        explanation=explanation,
        criteria=criteria,
        raw_score=raw_score,
        output=OrdinalScore(raw_score, 1, 5),
    )


_PROMETHEUS_RESULT = re.compile(r"\[RESULT\]\s*\(?\s*([1-5])\s*\)?")


def parse_prometheus_response(response: str, criteria: str) -> JudgeResult:
    """Parse a Prometheus 2 "(feedback) [RESULT] N" response into JudgeResult.

    Args:
        response: Raw response from the judge model.
        criteria: The evaluation criteria name.

    Returns:
        Parsed JudgeResult; the explanation is the feedback before [RESULT].

    Raises:
        EvaluationError: If the response contains no "[RESULT] N" with N in
            1-5. No default score is ever substituted.
    """
    matches = list(_PROMETHEUS_RESULT.finditer(response))
    if not matches:
        msg = f"Unparseable {criteria} judgment: no '[RESULT] N' (1-5) in the judge response"
        raise JudgeParseError(msg, criterion=criteria, raw_output=response)
    last = matches[-1]
    raw_score = int(last.group(1))
    verdict = verdict_from_raw_score(raw_score)
    return JudgeResult(
        verdict=verdict,
        score=(raw_score - 1) / 4,  # Normalize 1-5 to 0-1
        explanation=response[: last.start()].strip(),
        criteria=criteria,
        raw_score=raw_score,
        output=OrdinalScore(raw_score, 1, 5),
    )


# =============================================================================
# LLM Judge
# =============================================================================


_SCORE_N_TEMPLATES: dict[str, str] = {
    "faithfulness": FAITHFULNESS_PROMPT,
    "relevance": RELEVANCE_PROMPT,
    "hallucination": HALLUCINATION_PROMPT,
    "completeness": COMPLETENESS_PROMPT,
}


def _library_version() -> str:
    from ragnarok_ai import __version__

    return str(__version__)


def _medical_table_version() -> str:
    """Fingerprint of the abbreviation tables the medical normalizer relies on."""
    from ragnarok_ai.evaluators.medical import abbreviations as tables

    names = sorted(n for n in dir(tables) if n.isupper() and isinstance(getattr(tables, n), dict))
    payload = {n: getattr(tables, n) for n in names}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:12]


class LLMJudge:
    """Multi-criteria LLM-as-Judge using Prometheus 2.

    Provides comprehensive evaluation of RAG responses across multiple
    quality dimensions: faithfulness, relevance, hallucination, and completeness.

    Attributes:
        llm: The LLM provider for judge evaluations.
        model: Model name being used.
        medical_mode: If True, normalizes medical abbreviations before evaluation.

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

        Medical mode example:
        >>> judge = LLMJudge(medical_mode=True)
        >>> result = await judge.evaluate_faithfulness(
        ...     context="Patient has CHF",
        ...     question="What condition does the patient have?",
        ...     answer="Patient has congestive heart failure"
        ... )
        >>> # CHF and "congestive heart failure" normalized to same form

    Installation:
        Install Prometheus 2 (Q5_K_M quantized, ~4.8GB) via Ollama:
        >>> ollama pull hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M

        Or create a shorter alias:
        >>> ollama cp hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M prometheus2
    """

    # Prometheus 2 7B Q5_K_M from HuggingFace (~4.8GB, fits 16GB RAM)
    # Q5 provides better precision for subtle judgments (hallucination, faithfulness)
    DEFAULT_MODEL = "hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M"
    PROMPT_FORMATS: tuple[str, ...] = ("auto", "prometheus", "score-n")
    # Sampling options imposed on the Ollama client the judge builds itself.
    # Greedy decoding and a fixed seed make a judgment reproducible on the same
    # machine and Ollama version; the explicit context window is what the
    # truncation warning is measured against. A client passed via ``llm=`` is
    # used as given.
    JUDGE_TEMPERATURE: float = 0.0
    JUDGE_SEED: int = 42
    JUDGE_NUM_CTX: int = 8192

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        llm: LLMProtocol | None = None,
        medical_mode: bool = False,
        disambiguation_llm: LLMProtocol | None = None,
        prompt_format: Literal["auto", "prometheus", "score-n"] = "auto",
        keep_raw_output: bool = False,
    ) -> None:
        """Initialize LLMJudge.

        Args:
            model: Model name for Ollama. Defaults to Prometheus 2.
            base_url: Ollama API base URL.
            timeout: Request timeout in seconds. Default 300s for thorough evaluations.
            llm: Optional pre-configured LLM instance. If provided, model/base_url are ignored.
            medical_mode: If True, normalize medical abbreviations before evaluation.
                         Reduces false positives from abbreviation ambiguity in medical contexts.
            disambiguation_llm: Optional small LLM (e.g. Ollama qwen2.5:0.5b) used to
                         resolve ambiguous medical abbreviations when the surrounding
                         context gives the keyword scorer no signal. Only used when
                         medical_mode is True.
            prompt_format: Which prompt and parser to use. ``"prometheus"``: the
                         prometheus-eval absolute-grading template and the
                         ``[RESULT] N`` parser; ``"score-n"``: the ``Score: N``
                         prompts; ``"auto"`` (default): ``"prometheus"`` when the
                         model name contains "prometheus", ``"score-n"`` otherwise.
                         Set it explicitly for a Prometheus 2 model served under
                         another name (an ``ollama cp`` alias, a gateway) or for a
                         client without a ``model`` attribute. The format in use is
                         ``effective_prompt_format``.
            keep_raw_output: Keep the model's raw text in each result's
                         ``trace.raw_output``. Off by default: the text can copy
                         sensitive context.

        Raises:
            ValueError: If *prompt_format* is not one of ``PROMPT_FORMATS``.
        """
        if prompt_format not in self.PROMPT_FORMATS:
            msg = f"Unknown prompt_format {prompt_format!r}; expected one of {self.PROMPT_FORMATS}"
            raise ValueError(msg)
        self.prompt_format = prompt_format
        self.keep_raw_output = keep_raw_output
        self.model_requested: str | None = None
        self._disambiguation_model: str | None = None

        self._model_digests: dict[str, str] = {}
        self.model_digest: str | None = None
        if llm is not None:
            self.llm = llm
            self.model = getattr(llm, "model", "custom")
        else:
            try:
                from ragnarok_ai.adapters.llm.ollama import OllamaLLM
            except ModuleNotFoundError as exc:
                # httpx is only installed with an extra; say so instead of a bare ImportError
                raise JudgeDependencyError(extra="ollama", module="httpx") from exc

            selected_model = model or self.DEFAULT_MODEL
            self.model_requested = selected_model

            # Dependency missing raises JudgeDependencyError (above); server unreachable
            # raises JudgeConnectionError; model not installed raises
            # JudgeModelNotFoundError. No other model is ever substituted.
            available = self._get_available_models(base_url)
            selected_model = self._resolve_model(selected_model, available, base_url)

            self.llm = OllamaLLM(
                model=selected_model,
                base_url=base_url,
                timeout=timeout,
                temperature=self.JUDGE_TEMPERATURE,
                seed=self.JUDGE_SEED,
                num_ctx=self.JUDGE_NUM_CTX,
            )
            self.model = selected_model
            # Read once, from the same /api/tags answer that listed the models
            self.model_digest = self._model_digests.get(selected_model)

        self.truncation_warnings = 0

        # Prometheus 2 ignores the "Score: N" format and needs its native template
        if prompt_format == "auto":
            self._effective_prompt_format = "prometheus" if "prometheus" in str(self.model).lower() else "score-n"
        else:
            self._effective_prompt_format = prompt_format
        self._prometheus = self._effective_prompt_format == "prometheus"
        logger.info(
            f"Judge model '{self.model}': prompt format '{self._effective_prompt_format}' (requested '{prompt_format}')"
        )

        # Initialize medical abbreviation normalizer if medical_mode is enabled
        self.medical_mode = medical_mode
        self._normalizer: Any = None
        if medical_mode:
            from ragnarok_ai.evaluators.medical.medical_normalizer import MedicalAbbreviationNormalizer

            disambiguator = None
            if disambiguation_llm is not None:
                from ragnarok_ai.evaluators.medical.disambiguation import SLMDisambiguator

                disambiguator = SLMDisambiguator(disambiguation_llm)
                self._disambiguation_model = str(getattr(disambiguation_llm, "model", "custom"))
            self._normalizer = MedicalAbbreviationNormalizer(disambiguator=disambiguator)

    async def _generate(self, prompt: str) -> str:
        """Call the LLM and return its text (see ``_generate_traced`` for the counters)."""
        text, _ = await self._generate_traced(prompt)
        return text

    async def _generate_traced(self, prompt: str) -> tuple[str, CallTrace]:
        """Call the LLM; when it reports token counters, check for truncation and trace the call."""
        started = time.perf_counter()
        detailed = getattr(self.llm, "generate_detailed", None)
        if detailed is None or not inspect.iscoroutinefunction(detailed):
            text = str(await self.llm.generate(prompt))
            latency = (time.perf_counter() - started) * 1000
            return text, CallTrace(latency_ms=latency, raw_output=text if self.keep_raw_output else None)
        generation = await detailed(prompt)
        measured = (time.perf_counter() - started) * 1000
        num_ctx = getattr(self.llm, "num_ctx", None)
        # Generic name first (LLMResponse, OllamaGeneration property), Ollama's raw name as fallback
        prompt_tokens = getattr(generation, "prompt_tokens", None)
        if prompt_tokens is None:
            prompt_tokens = getattr(generation, "prompt_eval_count", None)
        completion_tokens = getattr(generation, "completion_tokens", None)
        if completion_tokens is None:
            completion_tokens = getattr(generation, "eval_count", None)
        truncated: bool | None = None
        if num_ctx and prompt_tokens is not None:
            truncated = looks_truncated(prompt_tokens, num_ctx)
        if truncated:
            self.truncation_warnings += 1
            logger.warning(
                f"Judge prompt probably truncated by Ollama: {prompt_tokens} prompt tokens evaluated for a "
                f"context window of {num_ctx} ({len(prompt)} characters sent). The judge scored a cut "
                "context. Raise num_ctx or shorten the context."
            )
        text = str(generation.text)
        reported_latency = getattr(generation, "latency_ms", None)
        trace = CallTrace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            context_truncated=truncated,
            latency_ms=float(reported_latency) if reported_latency is not None else measured,
            raw_output=text if self.keep_raw_output else None,
        )
        return text, trace

    @property
    def effective_prompt_format(self) -> str:
        """The prompt format actually in use: ``"prometheus"`` or ``"score-n"``."""
        return self._effective_prompt_format

    def _get_available_models(self, base_url: str) -> list[str]:
        """Get the list of models installed on the Ollama server.

        Args:
            base_url: Ollama API base URL.

        Also records each model's digest in ``self._model_digests``, so the
        constructor can set ``model_digest`` without a second request.

        Returns:
            List of model names, possibly empty.

        Raises:
            JudgeConnectionError: If the server cannot be reached, times out,
                or answers with an error status. An unreachable server is
                never reported as "model not installed".
        """
        import httpx

        url = f"{base_url}/api/tags"
        try:
            response = httpx.get(url, timeout=5.0)
        except httpx.TimeoutException as exc:
            msg = f"Ollama at {base_url} did not answer within 5s ({url}): {exc}"
            raise JudgeConnectionError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"Cannot reach Ollama at {base_url} ({url}): {exc}. Is the server running?"
            raise JudgeConnectionError(msg) from exc
        except Exception as exc:
            msg = f"Unexpected error while listing Ollama models at {base_url}: {exc}"
            raise JudgeConnectionError(msg) from exc
        if response.status_code != 200:
            msg = f"Ollama at {base_url} answered {response.status_code} to {url}: {response.text[:200]}"
            raise JudgeConnectionError(msg)
        data = response.json()
        models = data.get("models", [])
        self._model_digests = {str(m["name"]): str(m["digest"]) for m in models if m.get("digest")}
        return [str(m["name"]) for m in models]

    @staticmethod
    def _resolve_model(requested: str, available: list[str], base_url: str) -> str:
        """Return the installed name serving *requested*, or raise.

        Accepted resolutions, in order:

        1. exact name;
        2. tag normalization: ``mistral`` resolves to ``mistral:latest``;
        3. Prometheus aliases: when the requested name is a Prometheus 2
           model (its name contains "prometheus", e.g. the default or an
           ``ollama cp`` alias such as ``prometheus2``) and another installed
           model name contains "prometheus", that model is used, with an
           INFO log.

        Anything else raises: the judge never substitutes a different
        model for the one requested.

        Raises:
            JudgeModelNotFoundError: If no installed model serves *requested*.
        """
        if requested in available:
            return requested
        if ":" not in requested and f"{requested}:latest" in available:
            return f"{requested}:latest"
        if "prometheus" in requested.lower():
            for name in available:
                if "prometheus" in name.lower():
                    logger.info(
                        f"Judge model '{requested}' is not installed; using the installed Prometheus 2 model '{name}'"
                    )
                    return name
        raise JudgeModelNotFoundError(requested, available, base_url)

    # ------------------------------------------------------------------
    # JudgeProtocol
    # ------------------------------------------------------------------

    CRITERIA: frozenset[str] = frozenset({"faithfulness", "relevance", "hallucination", "completeness"})
    VERDICT_POLICY = OrdinalBands(pass_at=4, fail_at=2)
    # Criteria whose context and answer go through the medical normalizer
    _NORMALIZED_CRITERIA: frozenset[str] = frozenset({"faithfulness", "hallucination", "completeness"})

    @property
    def criteria(self) -> frozenset[str]:
        """The four criteria of the Prometheus prompts."""
        return self.CRITERIA

    def _input_transforms(self, criterion: str) -> tuple[str, ...]:
        """Identifiers of the transformations applied to the inputs for *criterion*."""
        if not (self.medical_mode and self._normalizer) or criterion not in self._NORMALIZED_CRITERIA:
            return ()
        transforms = [f"medical_normalizer:{_medical_table_version()}"]
        if self._disambiguation_model is not None:
            transforms.append(f"disambiguation:{self._disambiguation_model}")
        return tuple(transforms)

    def _prompt_template_sha256(self, criterion: str) -> str:
        """Hash of the prompt template for *criterion*, before any sample is inserted."""
        if self._prometheus:
            rubric = PROMETHEUS_SCORE_RUBRIC_TEMPLATE.format(**PROMETHEUS_RUBRICS[criterion])
            template = f"{PROMETHEUS_ABS_SYSTEM_PROMPT}\n\n" + PROMETHEUS_ABSOLUTE_PROMPT_WO_REF.replace(
                "{rubric}", rubric
            )
        else:
            template = _SCORE_N_TEMPLATES[criterion]
        return hashlib.sha256(template.encode("utf-8")).hexdigest()

    def config(self, criterion: str) -> JudgeConfig:
        """The run-level configuration for *criterion*.

        Raises:
            ValueError: If *criterion* is unknown.
        """
        if criterion not in self.CRITERIA:
            msg = f"Unknown criterion {criterion!r}; expected one of {sorted(self.CRITERIA)}"
            raise ValueError(msg)
        llm_options = getattr(self.llm, "options", None) or {}
        backend = "ollama" if type(self.llm).__module__.startswith("ragnarok_ai.adapters.llm.ollama") else None
        return JudgeConfig(
            judge_class=f"{type(self).__module__}.{type(self).__qualname__}",
            judge_kind="llm",
            model=str(self.model),
            criterion=criterion,
            verdict_policy=self.VERDICT_POLICY,
            ragnarok_ai_version=_library_version(),
            backend=backend,
            model_requested=self.model_requested,
            model_digest=self.model_digest,
            options=tuple(sorted((str(k), v) for k, v in dict(llm_options).items())),
            prompt_format=self._effective_prompt_format,
            prompt_template_sha256=self._prompt_template_sha256(criterion),
            input_transforms=self._input_transforms(criterion),
        )

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:
        """Judge one answer on one criterion.

        The prompt is exactly the one the named methods have always sent:
        ``evaluate_faithfulness``, ``evaluate_relevance``,
        ``detect_hallucination`` and ``evaluate_completeness`` are wrappers
        of this method. In medical mode, context and answer are normalized
        for faithfulness, hallucination and completeness, not for relevance.

        Raises:
            ValueError: If *criterion* is unknown.
            JudgeParseError: If the model's output cannot be parsed.
        """
        if criterion not in self.CRITERIA:
            msg = f"Unknown criterion {criterion!r}; expected one of {sorted(self.CRITERIA)}"
            raise ValueError(msg)
        if self.medical_mode and self._normalizer and criterion in self._NORMALIZED_CRITERIA:
            context, _ = await self._normalizer.normalize_text_async(context)
            answer, _ = await self._normalizer.normalize_text_async(answer)

        if self._prometheus:
            if criterion == "relevance":
                instruction = question
            elif criterion == "hallucination":
                instruction = f"Context:\n{context}"
            else:
                instruction = f"Context:\n{context}\n\nQuestion:\n{question}"
            prompt = build_prometheus_prompt(criterion, instruction, answer)
        elif criterion == "relevance":
            prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        elif criterion == "hallucination":
            prompt = HALLUCINATION_PROMPT.format(context=context, answer=answer)
        else:
            prompt = _SCORE_N_TEMPLATES[criterion].format(context=context, question=question, answer=answer)

        text, trace = await self._generate_traced(prompt)
        result = (
            parse_prometheus_response(text, criterion) if self._prometheus else parse_judge_response(text, criterion)
        )
        return replace(result, config=self.config(criterion), trace=trace)

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
        return await self.judge("faithfulness", question=question, context=context, answer=answer)

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
        return await self.judge("relevance", question=question, context="", answer=answer)

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
        return await self.judge("hallucination", question="", context=context, answer=answer)

    async def evaluate_completeness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> JudgeResult:
        """Evaluate if the answer completely addresses the question.

        Args:
            question: The user question.
            answer: The generated answer to evaluate.
            context: The source context/documents.

        Returns:
            JudgeResult with completeness assessment.
        """
        return await self.judge("completeness", question=question, context=context, answer=answer)

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
        items: list[dict[str, Any]],
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
