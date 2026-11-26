"""Claim-level evaluators as judges: ``FaithfulnessEvaluator`` and ``HallucinationDetector`` under the judge contract.

These two evaluators extract the claims of an answer with an LLM, then check
each claim against the context with one more call per claim. Their own
results (``FaithfulnessResult``, ``HallucinationResult``) are claim-level
pydantic models; this module turns them into ``JudgeResult`` values and
wraps the evaluators as ``JudgeProtocol`` judges, so they run through
``judge_many``, the ``judge-bench`` harness and ``JudgeCalibrator`` like
any other judge.

What a claim-level result is, under the contract:

- one ``UnitJudgment`` per claim: PASS and score 1 when the claim holds
  (supported, not hallucinated), FAIL and score 0 otherwise. A claim is a
  rewording produced by the model, not a slice of the answer: its ``span``
  is ``None``;
- the answer's score is the share of claims that hold (``fraction_pass``)
  and its verdict the worst claim's (``worst``): PASS only when every claim
  holds. ``aggregate_units`` computes both, as for sentences;
- **the score always reads "higher is better"**. ``HallucinationResult.score``
  is the share of hallucinated claims (0 is good); the judge result carries
  its complement, and the criterion is named ``hallucination_free`` so that
  the name says which way it goes;
- an answer without any claim (empty answer, or nothing extracted) is not
  scored: ``NoUnitsError``, the ``no_units`` failure of the contract;
- an empty context is not such a case. The claims are extracted and all
  marked FAIL without a verification call: nothing can support them.

Limits. The number of claims is decided by the model at each run, so two
runs of the same judge on the same answer need not have the same units:
comparisons between runs are made on answers, not on claims. The model's
digest and token counts are not known through ``LLMProtocol``; the trace
carries the latency only.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from ragnarok_ai.core.exceptions import ConfigurationError, NoUnitsError
from ragnarok_ai.core.provenance import library_commit
from ragnarok_ai.judges.aggregation import aggregate_units
from ragnarok_ai.judges.contract import AggregationPolicy, CallTrace, JudgeConfig, ScoreThreshold, UnitJudgment

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ragnarok_ai.evaluators.judge import JudgeResult

# Share of claims that hold, and the worst claim's verdict: PASS only when every claim holds
CLAIM_AGGREGATION = AggregationPolicy(score_rule="fraction_pass", verdict_rule="worst")
# The same rule read on the score, for what works from a threshold (calibration, benchmark): PASS at 1.0
CLAIM_VERDICT_POLICY = ScoreThreshold(pass_threshold=1.0)
PROMPT_FORMAT = "claims-json"
LABEL_HOLDS = "HOLDS"
LABEL_FAILS = "FAILS"


def claims_to_judge_result(
    criterion: str,
    claims: Sequence[tuple[str, bool, str]],
    *,
    config: JudgeConfig | None,
    nothing_to_judge: str,
) -> JudgeResult:
    """A ``JudgeResult`` from checked claims, given as ``(text, holds, reasoning)``.

    Raises:
        NoUnitsError: If there is no claim; *nothing_to_judge* says why.
    """
    from ragnarok_ai.evaluators.judge import JudgeResult

    if not claims:
        raise NoUnitsError(nothing_to_judge)
    units = tuple(
        UnitJudgment(
            text=text,
            span=None,  # a claim is a rewording by the model: it has no offsets in the answer
            score=1.0 if holds else 0.0,
            verdict="PASS" if holds else "FAIL",
            label=LABEL_HOLDS if holds else LABEL_FAILS,
        )
        for text, holds, _ in claims
    )
    score, verdict = aggregate_units(units, CLAIM_AGGREGATION, CLAIM_VERDICT_POLICY)
    held = sum(1 for _, holds, _ in claims if holds)
    lines = [f"{held} of {len(claims)} claim(s) hold."]
    lines += [f"[{'PASS' if holds else 'FAIL'}] {text}: {reasoning}".rstrip(": ") for text, holds, reasoning in claims]
    return JudgeResult(
        verdict=verdict,
        score=score,
        explanation="\n".join(lines),
        criteria=criterion,
        raw_score=None,  # an aggregate has no single native score
        output=None,
        config=config,
        units=units,
    )


def template_sha256(*templates: str) -> str:
    """Hash of prompt templates, before any sample is inserted."""
    return hashlib.sha256("\n\n".join(templates).encode("utf-8")).hexdigest()


def model_name(llm: object, model: str | None) -> str:
    """The model's name: *model* when given, else ``llm.model``.

    Raises:
        ConfigurationError: If neither is available. ``LLMProtocol`` does
            not expose a model name, and two runs with different unnamed
            models would otherwise share one configuration and be declared
            comparable.
    """
    if model:
        return str(model)
    found = getattr(llm, "model", None)
    if isinstance(found, str) and found:
        return found
    msg = (
        f"{type(llm).__name__} does not expose a model name (no `model` attribute): pass it explicitly, "
        'as_judge(model="<name>"). The name is recorded in the judge configuration, which decides whether '
        "two runs are comparable."
    )
    raise ConfigurationError(msg)


def llm_options(llm: object) -> tuple[tuple[str, Any], ...]:
    """Options the LLM sends to its backend, as sorted pairs: ``llm.options``, and ``llm.temperature`` when set."""
    options = dict(getattr(llm, "options", None) or {})
    temperature = getattr(llm, "temperature", None)
    if temperature is not None and not callable(temperature):
        options.setdefault("temperature", temperature)
    return tuple(sorted((str(key), value) for key, value in options.items()))


def claim_judge_config(
    evaluator: object,
    llm: object,
    criterion: str,
    *,
    model: str,
    extraction_template: str,
    check_template: str,
    input_transforms: tuple[str, ...] = (),
    judge_specific: tuple[tuple[str, Any], ...] = (),
) -> JudgeConfig:
    """The run-level configuration of a claim-level evaluator used as a judge."""
    from ragnarok_ai import __version__

    return JudgeConfig(
        judge_class=f"{type(evaluator).__module__}.{type(evaluator).__qualname__}",
        judge_kind="llm",
        model=model,
        criterion=criterion,
        verdict_policy=CLAIM_VERDICT_POLICY,
        ragnarok_ai_version=str(__version__),
        backend="ollama" if type(llm).__module__.startswith("ragnarok_ai.adapters.llm.ollama") else None,
        options=llm_options(llm),
        prompt_format=PROMPT_FORMAT,
        prompt_template_sha256=template_sha256(extraction_template, check_template),
        unit="claim",
        aggregation=CLAIM_AGGREGATION,
        input_transforms=input_transforms,
        # What cuts the answer into units is the extraction prompt, run by the judge's own model
        segmenter=f"llm-claim-extraction:{template_sha256(extraction_template)[:16]}",
        judge_specific=tuple(sorted((("empty_context", "claims extracted, all FAIL, no check call"), *judge_specific))),
        source_commit=library_commit(),
    )


class ClaimEvaluator(Protocol):
    """What ``ClaimLevelJudge`` needs from an evaluator."""

    JUDGE_CRITERION: str

    def config(self, criterion: str | None = None, *, model: str | None = None) -> JudgeConfig:
        """The judge configuration of the evaluator."""
        ...  # pragma: no cover

    async def _judge_result(self, *, context: str, answer: str, config: JudgeConfig | None = None) -> JudgeResult:
        """One judgment under the contract (private to the evaluator: ``as_judge()`` is the public way in)."""
        ...  # pragma: no cover


class ClaimLevelJudge:
    """A claim-level evaluator behind ``JudgeProtocol``, with one criterion.

    Built by ``FaithfulnessEvaluator.as_judge()`` (criterion
    ``faithfulness``) and ``HallucinationDetector.as_judge()`` (criterion
    ``hallucination_free``). The question is not part of the evaluators'
    prompts and is not used.
    """

    def __init__(self, evaluator: ClaimEvaluator, *, model: str | None = None) -> None:
        """Wrap *evaluator*.

        Raises:
            ConfigurationError: If the model's name is neither given nor readable on the evaluator's LLM.
        """
        self.evaluator = evaluator
        self.criterion = evaluator.JUDGE_CRITERION
        self._config = evaluator.config(self.criterion, model=model)  # a missing model name surfaces here

    @property
    def criteria(self) -> frozenset[str]:
        """The evaluator's single criterion."""
        return frozenset({self.criterion})

    def config(self, criterion: str) -> JudgeConfig:
        """The run-level configuration for *criterion*.

        Raises:
            ValueError: If *criterion* is not the evaluator's.
        """
        self._check(criterion)
        return self._config

    def _check(self, criterion: str) -> None:
        if criterion != self.criterion:
            msg = f"Unknown criterion {criterion!r}; expected {self.criterion!r}"
            raise ValueError(msg)

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:  # noqa: ARG002
        """Judge *answer* against *context*; *question* is not part of the evaluator's prompts.

        Raises:
            ValueError: If *criterion* is not the evaluator's.
            NoUnitsError: If the answer holds no claim (empty answer, or none extracted).
            EvaluationError: If a call to the model fails or its output cannot be read.
        """
        self._check(criterion)
        started = time.perf_counter()
        result = await self.evaluator._judge_result(context=context, answer=answer, config=self._config)
        # Extraction plus one call per claim: the latency is the whole judgment's. Token counts are not
        # available through LLMProtocol
        return replace(result, trace=CallTrace(latency_ms=(time.perf_counter() - started) * 1000.0))


__all__ = [
    "CLAIM_AGGREGATION",
    "CLAIM_VERDICT_POLICY",
    "LABEL_FAILS",
    "LABEL_HOLDS",
    "ClaimLevelJudge",
    "claim_judge_config",
    "claims_to_judge_result",
]
