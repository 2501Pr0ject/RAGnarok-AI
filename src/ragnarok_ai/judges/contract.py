"""Common contract for every judge: native outputs, verdict policies, configuration, traces.

These types are pure data. Nothing in the library is wired to them yet; the
judges adopt them in later changes. All dataclasses are frozen and validated
at construction, so a value that exists is a value that is well formed.

Vocabulary:

- A **native output** is what a judge actually produces: an integer on a
  scale (Prometheus), a probability (HHEM, MiniCheck), or probabilities over
  labels (an NLI model). Each projects to a ``score`` in ``[0, 1]``.
- A **verdict policy** turns a native output into PASS, FAIL or PARTIAL. The
  verdict is always computed; it is never read from a model's text.
- A **JudgeConfig** holds everything that makes two runs comparable: model,
  options, prompt template, criterion, unit, aggregation, policy, version.
  Two results are comparable when their configs have the same hash.
- A **CallTrace** holds what one call cost and reported.
- A **JudgeAttempt** is either a valid result or a described failure, never
  both, never neither.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ragnarok_ai.evaluators.judge import JudgeResult

Verdict = Literal["PASS", "FAIL", "PARTIAL"]
VERDICTS: tuple[Verdict, ...] = ("PASS", "FAIL", "PARTIAL")

Unit = Literal["answer", "sentence", "claim"]
UNITS: tuple[Unit, ...] = ("answer", "sentence", "claim")

JudgeKind = Literal["llm", "classifier", "custom"]
JUDGE_KINDS: tuple[JudgeKind, ...] = ("llm", "classifier", "custom")

_PROB_TOLERANCE = 1e-6


# =============================================================================
# Native outputs
# =============================================================================


@dataclass(frozen=True)
class OrdinalScore:
    """An integer on a closed scale, e.g. Prometheus's 1-5.

    Attributes:
        value: The score produced by the judge.
        min: Lowest value of the scale.
        max: Highest value of the scale.
    """

    value: int
    min: int
    max: int
    kind: Literal["ordinal"] = field(default="ordinal", init=False)

    def __post_init__(self) -> None:
        if self.min >= self.max:
            msg = f"OrdinalScore scale must satisfy min < max, got min={self.min} max={self.max}"
            raise ValueError(msg)
        if not self.min <= self.value <= self.max:
            msg = f"OrdinalScore value {self.value} is outside the scale [{self.min}, {self.max}]"
            raise ValueError(msg)

    @property
    def score(self) -> float:
        """Projection to ``[0, 1]``: ``(value - min) / (max - min)``."""
        return (self.value - self.min) / (self.max - self.min)


@dataclass(frozen=True)
class BinaryProbability:
    """The probability of the positive class, e.g. "supported".

    Attributes:
        p_positive: Probability in ``[0, 1]``.
    """

    p_positive: float
    kind: Literal["binary"] = field(default="binary", init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_positive <= 1.0:
            msg = f"BinaryProbability p_positive must be in [0, 1], got {self.p_positive}"
            raise ValueError(msg)

    @property
    def score(self) -> float:
        """Projection to ``[0, 1]``: the probability itself."""
        return self.p_positive


@dataclass(frozen=True)
class ClassProbabilities:
    """A probability distribution over labels, e.g. an NLI model's three classes.

    Attributes:
        labels: Distinct label names.
        probs: One probability per label, summing to 1.
        positive_label: The label whose probability is the score.
    """

    labels: tuple[str, ...]
    probs: tuple[float, ...]
    positive_label: str
    kind: Literal["classes"] = field(default="classes", init=False)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.probs):
            msg = f"ClassProbabilities needs one probability per label, got {len(self.labels)} labels and {len(self.probs)} probs"
            raise ValueError(msg)
        if not self.labels:
            msg = "ClassProbabilities needs at least one label"
            raise ValueError(msg)
        if len(set(self.labels)) != len(self.labels):
            msg = f"ClassProbabilities labels must be unique, got {self.labels}"
            raise ValueError(msg)
        if self.positive_label not in self.labels:
            msg = f"ClassProbabilities positive_label {self.positive_label!r} is not one of {self.labels}"
            raise ValueError(msg)
        if any(p < 0.0 or p > 1.0 for p in self.probs):
            msg = f"ClassProbabilities probs must be in [0, 1], got {self.probs}"
            raise ValueError(msg)
        total = sum(self.probs)
        if abs(total - 1.0) > _PROB_TOLERANCE:
            msg = f"ClassProbabilities probs must sum to 1 (tolerance {_PROB_TOLERANCE}), got {total}"
            raise ValueError(msg)

    @property
    def score(self) -> float:
        """Projection to ``[0, 1]``: the probability of ``positive_label``."""
        return self.probs[self.labels.index(self.positive_label)]

    @property
    def predicted(self) -> str:
        """The label with the highest probability (first one on ties)."""
        best = max(range(len(self.probs)), key=lambda i: (self.probs[i], -i))
        return self.labels[best]


JudgeOutput = OrdinalScore | BinaryProbability | ClassProbabilities


# =============================================================================
# Verdict policies
# =============================================================================


@dataclass(frozen=True)
class OrdinalBands:
    """Verdict bands on an ordinal scale: PASS at or above, FAIL at or below.

    Prometheus: ``OrdinalBands(pass_at=4, fail_at=2)`` on the 1-5 scale, which
    is the rule of ``verdict_from_raw_score``.

    Attributes:
        pass_at: Lowest value that is a PASS.
        fail_at: Highest value that is a FAIL. Values strictly between
            ``fail_at`` and ``pass_at`` are PARTIAL.
        scale_min: Lowest value of the scale, used to express the pass
            threshold on the ``[0, 1]`` score.
        scale_max: Highest value of the scale.
    """

    pass_at: int
    fail_at: int
    scale_min: int = 1
    scale_max: int = 5
    kind: Literal["ordinal_bands"] = field(default="ordinal_bands", init=False)

    def __post_init__(self) -> None:
        if self.scale_min >= self.scale_max:
            msg = f"OrdinalBands scale must satisfy scale_min < scale_max, got {self.scale_min} and {self.scale_max}"
            raise ValueError(msg)
        if not self.scale_min <= self.fail_at < self.pass_at <= self.scale_max:
            msg = (
                f"OrdinalBands needs scale_min <= fail_at < pass_at <= scale_max, got fail_at={self.fail_at} "
                f"pass_at={self.pass_at} on [{self.scale_min}, {self.scale_max}]"
            )
            raise ValueError(msg)

    def verdict(self, output: JudgeOutput) -> Verdict:
        """Apply the bands to an ``OrdinalScore``."""
        if not isinstance(output, OrdinalScore):
            msg = f"OrdinalBands applies to OrdinalScore, got {type(output).__name__}"
            raise TypeError(msg)
        if (output.min, output.max) != (self.scale_min, self.scale_max):
            msg = (
                f"OrdinalBands is defined on [{self.scale_min}, {self.scale_max}] "
                f"but the output is on [{output.min}, {output.max}]"
            )
            raise ValueError(msg)
        if output.value >= self.pass_at:
            return "PASS"
        if output.value <= self.fail_at:
            return "FAIL"
        return "PARTIAL"

    def pass_threshold_on_score(self) -> float:
        """The pass band's lower edge projected on the ``[0, 1]`` score (0.75 for 4 on 1-5)."""
        return (self.pass_at - self.scale_min) / (self.scale_max - self.scale_min)


@dataclass(frozen=True)
class ScoreThreshold:
    """Verdict by thresholds on the ``[0, 1]`` score.

    Attributes:
        pass_threshold: PASS when ``score >= pass_threshold``.
        fail_threshold: FAIL when ``score <= fail_threshold``; PARTIAL in
            between. ``None`` gives a binary verdict: FAIL below
            ``pass_threshold``.
    """

    pass_threshold: float
    fail_threshold: float | None = None
    kind: Literal["score_threshold"] = field(default="score_threshold", init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.pass_threshold <= 1.0:
            msg = f"ScoreThreshold pass_threshold must be in [0, 1], got {self.pass_threshold}"
            raise ValueError(msg)
        if self.fail_threshold is not None and not 0.0 <= self.fail_threshold < self.pass_threshold:
            msg = (
                f"ScoreThreshold fail_threshold must be in [0, pass_threshold), got {self.fail_threshold} "
                f"with pass_threshold={self.pass_threshold}"
            )
            raise ValueError(msg)

    def verdict(self, output: JudgeOutput) -> Verdict:
        """Apply the thresholds to the output's score."""
        score = output.score
        if score >= self.pass_threshold:
            return "PASS"
        if self.fail_threshold is None or score <= self.fail_threshold:
            return "FAIL"
        return "PARTIAL"

    def pass_threshold_on_score(self) -> float:
        """The pass threshold itself."""
        return self.pass_threshold


@dataclass(frozen=True)
class LabelMap:
    """Verdict by the predicted label of a ``ClassProbabilities``.

    Unsupported information is a failure, never PARTIAL: a grounding judge
    maps ``SUPPORTED`` to PASS and both ``UNSUPPORTED`` and ``CONTRADICTED``
    to FAIL; an NLI judge maps ``entailment`` to PASS and both ``neutral``
    and ``contradiction`` to FAIL.

    Attributes:
        mapping: ``(label, verdict)`` pairs, one per label the judge can predict.
    """

    mapping: tuple[tuple[str, Verdict], ...]
    kind: Literal["label_map"] = field(default="label_map", init=False)

    def __post_init__(self) -> None:
        if not self.mapping:
            msg = "LabelMap needs at least one (label, verdict) pair"
            raise ValueError(msg)
        labels = [label for label, _ in self.mapping]
        if len(set(labels)) != len(labels):
            msg = f"LabelMap labels must be unique, got {labels}"
            raise ValueError(msg)
        for label, verdict in self.mapping:
            if verdict not in VERDICTS:
                msg = f"LabelMap verdict for {label!r} must be one of {VERDICTS}, got {verdict!r}"
                raise ValueError(msg)

    @property
    def as_dict(self) -> dict[str, Verdict]:
        """The mapping as a dictionary."""
        return dict(self.mapping)

    def verdict(self, output: JudgeOutput) -> Verdict:
        """Apply the mapping to the predicted label of a ``ClassProbabilities``."""
        if not isinstance(output, ClassProbabilities):
            msg = f"LabelMap applies to ClassProbabilities, got {type(output).__name__}"
            raise TypeError(msg)
        predicted = output.predicted
        try:
            return self.as_dict[predicted]
        except KeyError:
            msg = f"LabelMap has no verdict for predicted label {predicted!r} (known: {[k for k, _ in self.mapping]})"
            raise ValueError(msg) from None

    def pass_threshold_on_score(self) -> float:
        """0.5 on the score, which is P(positive label).

        The verdict itself follows the predicted label (argmax), not this
        threshold. With two classes the two agree. With three or more they
        do not: the positive label can be the argmax with a probability
        below 0.5 (e.g. 0.4 / 0.3 / 0.3), and a calibrator sweeping the
        score may recommend a threshold that no label rule reproduces.
        """
        return 0.5


VerdictPolicy = OrdinalBands | ScoreThreshold | LabelMap


def apply_verdict_policy(policy: VerdictPolicy, output: JudgeOutput) -> Verdict:
    """Compute the verdict of *output* under *policy*."""
    return policy.verdict(output)


# =============================================================================
# Aggregation of units
# =============================================================================

ScoreRule = Literal["fraction_pass", "mean", "min"]
SCORE_RULES: tuple[ScoreRule, ...] = ("fraction_pass", "mean", "min")
VerdictRule = Literal["worst", "from_score"]
VERDICT_RULES: tuple[VerdictRule, ...] = ("worst", "from_score")


@dataclass(frozen=True)
class AggregationPolicy:
    """How unit judgments (sentences, claims) become an answer-level result.

    Attributes:
        score_rule: ``fraction_pass`` (share of units with verdict PASS),
            ``mean`` or ``min`` of the unit scores.
        verdict_rule: ``worst`` (FAIL if any unit is FAIL, else PARTIAL if any
            is PARTIAL, else PASS) or ``from_score`` (the judge's verdict policy
            applied to the aggregated score).
    """

    score_rule: ScoreRule
    verdict_rule: VerdictRule

    def __post_init__(self) -> None:
        if self.score_rule not in SCORE_RULES:
            msg = f"AggregationPolicy score_rule must be one of {SCORE_RULES}, got {self.score_rule!r}"
            raise ValueError(msg)
        if self.verdict_rule not in VERDICT_RULES:
            msg = f"AggregationPolicy verdict_rule must be one of {VERDICT_RULES}, got {self.verdict_rule!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class UnitJudgment:
    """The judgment of one unit of an answer (a sentence or a claim).

    Attributes:
        text: The unit's text.
        span: ``(start, end)`` offsets of the unit in the answer.
        output: The judge's native output for this unit, if judged.
        score: The unit's score in ``[0, 1]``, if judged.
        verdict: The unit's verdict, if judged.
        label: The predicted label, for label-producing judges.
        skipped: Why the unit was not judged (e.g. "no alphanumeric content"),
            or ``None`` when it was. Skipped units are excluded from aggregation.
    """

    text: str
    span: tuple[int, int]
    output: JudgeOutput | None = None
    score: float | None = None
    verdict: Verdict | None = None
    label: str | None = None
    skipped: str | None = None

    def __post_init__(self) -> None:
        start, end = self.span
        if start < 0 or end < start:
            msg = f"UnitJudgment span must satisfy 0 <= start <= end, got {self.span}"
            raise ValueError(msg)
        if self.skipped is None and self.score is None:
            msg = "UnitJudgment needs a score unless it is skipped (set `skipped` to the reason)"
            raise ValueError(msg)
        if self.skipped is not None and (self.score is not None or self.verdict is not None):
            msg = "A skipped UnitJudgment carries no score and no verdict"
            raise ValueError(msg)
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            msg = f"UnitJudgment score must be in [0, 1], got {self.score}"
            raise ValueError(msg)
        if self.verdict is not None and self.verdict not in VERDICTS:
            msg = f"UnitJudgment verdict must be one of {VERDICTS}, got {self.verdict!r}"
            raise ValueError(msg)


# =============================================================================
# Configuration (run level) and trace (call level)
# =============================================================================


def _jsonable(value: Any) -> Any:
    """Convert dataclasses and tuples to JSON-friendly structures.

    Dataclasses carry their type name under ``"type"`` so that two policies
    with the same fields but different semantics hash differently. Numbers
    are normalized: an integral float becomes an int, so ``0`` and ``0.0``
    (or ``8192`` and ``8192.0``) give the same hash. Booleans stay booleans.
    """
    if is_dataclass(value) and not isinstance(value, type):
        payload = {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
        payload["type"] = type(value).__name__
        return payload
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    return str(value)


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, dataclasses expanded, numbers normalized."""
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _check_json_pairs(name: str, pairs: tuple[tuple[str, Any], ...]) -> None:
    """Keys sorted and unique, values JSON-serializable (no NaN, no arbitrary objects)."""
    keys = [k for k, _ in pairs]
    if any(not isinstance(k, str) for k in keys):
        msg = f"JudgeConfig {name} keys must be strings, got {keys}"
        raise ValueError(msg)
    if keys != sorted(keys):
        msg = f"JudgeConfig {name} must be sorted by key, got {keys}"
        raise ValueError(msg)
    if len(set(keys)) != len(keys):
        msg = f"JudgeConfig {name} keys must be unique, got {keys}"
        raise ValueError(msg)
    for key, value in pairs:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            msg = f"JudgeConfig {name}[{key!r}] is not JSON-serializable: {exc}"
            raise ValueError(msg) from exc


@dataclass(frozen=True)
class JudgeConfig:
    """Everything that makes two judge runs comparable. One per judge and criterion.

    Attributes:
        judge_class: Qualified class name of the judge.
        judge_kind: ``llm``, ``classifier`` or ``custom``.
        model: Model actually used, after alias resolution.
        criterion: The judged criterion.
        verdict_policy: How verdicts are computed from native outputs.
        ragnarok_ai_version: Library version that produced the results.
        backend: ``ollama``, ``transformers``, ... or ``None``.
        model_requested: What the user asked for, when it differs from ``model``.
        model_digest: Ollama digest, pinned Hugging Face revision, or hash of the weights.
        options: Options sent to the backend, as sorted ``(key, value)`` pairs.
        prompt_format: ``prometheus``, ``score-n``, or ``None`` for a classifier.
        prompt_template_sha256: Hash of the template before the sample is
            inserted (system prompt, template and rubric; no question, context
            or answer).
        unit: ``answer`` (default), ``sentence`` or ``claim``.
        aggregation: How units are aggregated; ``None`` when ``unit == "answer"``.
        input_transforms: Transformations applied to inputs, e.g.
            ``medical_normalizer:<version>``. A run with a transform is never
            comparable to a run without it.
        segmenter: Segmenter id and version when ``unit != "answer"``.
        judge_specific: Parameters specific to one judge, as sorted pairs.
    """

    judge_class: str
    judge_kind: JudgeKind
    model: str
    criterion: str
    verdict_policy: VerdictPolicy
    ragnarok_ai_version: str
    backend: str | None = None
    model_requested: str | None = None
    model_digest: str | None = None
    options: tuple[tuple[str, Any], ...] = ()
    prompt_format: str | None = None
    prompt_template_sha256: str | None = None
    unit: Unit = "answer"
    aggregation: AggregationPolicy | None = None
    input_transforms: tuple[str, ...] = ()
    segmenter: str | None = None
    judge_specific: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.judge_kind not in JUDGE_KINDS:
            msg = f"JudgeConfig judge_kind must be one of {JUDGE_KINDS}, got {self.judge_kind!r}"
            raise ValueError(msg)
        if self.unit not in UNITS:
            msg = f"JudgeConfig unit must be one of {UNITS}, got {self.unit!r}"
            raise ValueError(msg)
        if self.unit == "answer" and self.aggregation is not None:
            msg = "JudgeConfig aggregation must be None when unit is 'answer'"
            raise ValueError(msg)
        if self.unit != "answer" and self.aggregation is None:
            msg = f"JudgeConfig unit {self.unit!r} requires an aggregation policy"
            raise ValueError(msg)
        if self.unit != "answer" and not self.segmenter:
            msg = f"JudgeConfig unit {self.unit!r} requires a segmenter id"
            raise ValueError(msg)
        _check_json_pairs("options", self.options)
        _check_json_pairs("judge_specific", self.judge_specific)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view of the configuration (policies carry their type name)."""
        payload = _jsonable(self)
        assert isinstance(payload, dict)
        payload.pop("type", None)
        return payload

    @property
    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON of every field."""
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def comparable_with(self, other: JudgeConfig) -> bool:
        """Two runs are comparable when their configurations hash identically."""
        return self.config_hash == other.config_hash

    def __hash__(self) -> int:
        return hash(self.config_hash)


@dataclass(frozen=True)
class CallTrace:
    """What one judge call cost and reported.

    Attributes:
        prompt_tokens: Prompt tokens evaluated, as reported by the backend.
        completion_tokens: Generated tokens, as reported by the backend.
        context_truncated: Whether the prompt was probably truncated (the
            1.12.3 detection), ``None`` when it could not be assessed.
        latency_ms: Wall-clock latency of the call.
        raw_output: The model's raw text, kept only on request: it can copy
            sensitive context.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    context_truncated: bool | None = None
    latency_ms: float | None = None
    raw_output: str | None = None

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                msg = f"CallTrace {name} cannot be negative, got {value}"
                raise ValueError(msg)
        if self.latency_ms is not None and self.latency_ms < 0:
            msg = f"CallTrace latency_ms cannot be negative, got {self.latency_ms}"
            raise ValueError(msg)


# =============================================================================
# Attempt: a valid result or a described failure
# =============================================================================

FailureKind = Literal["parse", "backend", "timeout", "no_units"]
FAILURE_KINDS: tuple[FailureKind, ...] = ("parse", "backend", "timeout", "no_units")


@dataclass(frozen=True)
class JudgeFailure:
    """Why a sample produced no result.

    Attributes:
        kind: ``parse`` (unreadable output), ``backend`` (the model or server
            failed), ``timeout``, or ``no_units`` (nothing to judge after
            segmentation).
        message: Human-readable detail.
        raw_output: The model's raw text when available and kept.
    """

    kind: FailureKind
    message: str
    raw_output: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in FAILURE_KINDS:
            msg = f"JudgeFailure kind must be one of {FAILURE_KINDS}, got {self.kind!r}"
            raise ValueError(msg)
        if not self.message:
            msg = "JudgeFailure needs a message"
            raise ValueError(msg)


@dataclass(frozen=True)
class JudgeAttempt:
    """One sample judged on one criterion: a result or a failure, never both.

    Attributes:
        sample_id: The sample's identifier.
        criterion: The judged criterion.
        config: The run configuration the attempt belongs to.
        result: The judgment, when the call succeeded. Always a valid
            ``JudgeResult`` with a score.
        error: The failure, when it did not.
        trace: What the call cost and reported, when known.
    """

    sample_id: str
    criterion: str
    config: JudgeConfig
    result: JudgeResult | None
    error: JudgeFailure | None
    trace: CallTrace | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            msg = "JudgeAttempt needs exactly one of result and error"
            raise ValueError(msg)
        if self.config.criterion != self.criterion:
            msg = f"JudgeAttempt criterion {self.criterion!r} differs from its config's {self.config.criterion!r}"
            raise ValueError(msg)

    @property
    def ok(self) -> bool:
        """True when the attempt produced a result."""
        return self.result is not None


__all__ = [
    "FAILURE_KINDS",
    "JUDGE_KINDS",
    "SCORE_RULES",
    "UNITS",
    "VERDICTS",
    "VERDICT_RULES",
    "AggregationPolicy",
    "BinaryProbability",
    "CallTrace",
    "ClassProbabilities",
    "FailureKind",
    "JudgeAttempt",
    "JudgeConfig",
    "JudgeFailure",
    "JudgeKind",
    "JudgeOutput",
    "LabelMap",
    "OrdinalBands",
    "OrdinalScore",
    "ScoreRule",
    "ScoreThreshold",
    "Unit",
    "UnitJudgment",
    "Verdict",
    "VerdictPolicy",
    "VerdictRule",
    "apply_verdict_policy",
    "canonical_json",
]
