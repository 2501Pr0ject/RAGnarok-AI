"""Simulated judges for the harness tests: deterministic, no model, no data."""

from __future__ import annotations

from ragnarok_ai.core.exceptions import JudgeParseError
from ragnarok_ai.evaluators.judge import JudgeResult
from ragnarok_ai.judges import (
    BinaryProbability,
    CallTrace,
    ClassProbabilities,
    JudgeConfig,
    LabelMap,
    OrdinalBands,
    OrdinalScore,
    ScoreThreshold,
)

CRITERIA = frozenset({"faithfulness"})


def _p_from_text(text: str) -> float:
    """A probability of support encoded in the text: 'p=0.8' marker, default 0.9."""
    if "p=" in text:
        return float(text.split("p=", 1)[1].split()[0].rstrip(".,;"))
    return 0.9


class ScriptedOrdinalJudge:
    """Prometheus-like: a 1-5 score read from a '(raw N)' marker, default 5; '(fail-parse)' raises."""

    criteria = CRITERIA

    def __init__(self) -> None:
        self.received: list[dict[str, str]] = []

    def config(self, criterion: str) -> JudgeConfig:
        return JudgeConfig(
            "tests.ScriptedOrdinalJudge", "custom", "scripted-ordinal", criterion, OrdinalBands(4, 2), "test"
        )

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:
        self.received.append({"question": question, "context": context, "answer": answer})
        if "(fail-parse)" in answer:
            raise JudgeParseError("Unparseable", criterion=criterion, raw_output=answer)
        raw = 5
        if "(raw " in answer:
            raw = int(answer.split("(raw ", 1)[1].split(")")[0])
        return JudgeResult(
            verdict="PASS" if raw >= 4 else ("FAIL" if raw <= 2 else "PARTIAL"),
            score=(raw - 1) / 4,
            explanation="scripted",
            criteria=criterion,
            raw_score=raw,
            output=OrdinalScore(raw, 1, 5),
        )


class ScriptedProbabilityJudge:
    """HHEM-like: P(supported) read from a 'p=0.x' marker, default 0.9; threshold policy 0.5.

    With ``training_length`` set, the configuration declares it and each
    trace counts the whitespace tokens of context and answer, as a classifier
    with a known training length would.
    """

    criteria = CRITERIA

    def __init__(self, training_length: int | None = None, training_length_kind: str | None = None) -> None:
        self.training_length = training_length
        self.training_length_kind = training_length_kind

    def config(self, criterion: str) -> JudgeConfig:
        specific: tuple[tuple[str, object], ...] = ()
        if self.training_length is not None:
            specific = (("training_length", self.training_length),)
            if self.training_length_kind is not None:
                specific += (("training_length_kind", self.training_length_kind),)
        return JudgeConfig(
            "tests.ScriptedProbabilityJudge",
            "classifier",
            "scripted-prob",
            criterion,
            ScoreThreshold(0.5),
            "test",
            judge_specific=specific,
        )

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:  # noqa: ARG002
        p = _p_from_text(answer)
        trace = None
        if self.training_length is not None:
            trace = CallTrace(prompt_tokens=len(context.split()) + len(answer.split()), context_truncated=False)
        return JudgeResult(
            verdict="PASS" if p >= 0.5 else "FAIL",
            score=p,
            explanation="",
            criteria=criterion,
            raw_score=None,
            output=BinaryProbability(p),
            trace=trace,
        )


class ScriptedThreeClassJudge:
    """NLI-like: predicts the class named in a '[C]', '[U]' or '[S]' marker, default SUPPORTED."""

    criteria = CRITERIA
    LABELS = ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED")

    def config(self, criterion: str) -> JudgeConfig:
        policy = LabelMap((("SUPPORTED", "PASS"), ("CONTRADICTED", "FAIL"), ("UNSUPPORTED", "FAIL")))
        return JudgeConfig("tests.ScriptedThreeClassJudge", "classifier", "scripted-3c", criterion, policy, "test")

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:  # noqa: ARG002
        probs = {"SUPPORTED": 0.1, "CONTRADICTED": 0.1, "UNSUPPORTED": 0.1}
        predicted = "CONTRADICTED" if "[C]" in answer else ("UNSUPPORTED" if "[U]" in answer else "SUPPORTED")
        probs[predicted] = 0.8
        output = ClassProbabilities(self.LABELS, tuple(probs[label] for label in self.LABELS), "SUPPORTED")
        return JudgeResult(
            verdict="PASS" if predicted == "SUPPORTED" else "FAIL",
            score=output.score,
            explanation="",
            criteria=criterion,
            raw_score=None,
            output=output,
        )
