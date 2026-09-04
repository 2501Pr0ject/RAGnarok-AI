"""Tests for judge calibration."""

from __future__ import annotations

import pytest

from ragnarok_ai.calibration.calibrator import (
    JudgeCalibrator,
    _cohens_kappa,
    _wilson_interval,
)
from ragnarok_ai.calibration.models import (
    CalibrationSample,
    CalibrationSet,
    interpret_kappa,
)
from ragnarok_ai.evaluators.judge import JudgeResult


def make_sample(human_pass: bool = True, **overrides: object) -> CalibrationSample:
    """Build a labeled sample."""
    defaults: dict = {
        "question": "What is CHF?",
        "context": "CHF is congestive heart failure.",
        "answer": "Congestive heart failure.",
        "human_pass": human_pass,
    }
    defaults.update(overrides)
    return CalibrationSample(**defaults)


class ScriptedJudge:
    """Fake LLMJudge returning scripted scores per criterion, in order."""

    def __init__(self, scores: dict[str, list[float]]) -> None:
        self.scores = {k: list(v) for k, v in scores.items()}
        self.calls: list[str] = []

    def _result(self, criterion: str) -> JudgeResult:
        self.calls.append(criterion)
        score = self.scores[criterion].pop(0)
        verdict = "PASS" if score >= 0.7 else ("FAIL" if score < 0.4 else "PARTIAL")
        return JudgeResult(
            criteria=criterion,
            verdict=verdict,
            score=score,
            explanation="scripted",
            raw_score=max(1, min(5, round(score * 4 + 1))),
        )

    async def evaluate_faithfulness(self, context: str, question: str, answer: str) -> JudgeResult:  # noqa: ARG002
        return self._result("faithfulness")

    async def evaluate_relevance(self, question: str, answer: str) -> JudgeResult:  # noqa: ARG002
        return self._result("relevance")

    async def detect_hallucination(self, context: str, answer: str) -> JudgeResult:  # noqa: ARG002
        return self._result("hallucination")

    async def evaluate_completeness(self, question: str, answer: str, context: str) -> JudgeResult:  # noqa: ARG002
        return self._result("completeness")


class TestStatistics:
    """Test suite for the statistical helpers."""

    def test_kappa_perfect_agreement(self) -> None:
        assert _cohens_kappa([True, False, True], [True, False, True]) == 1.0

    def test_kappa_no_better_than_chance(self) -> None:
        # Judge always passes; human split 50/50 — kappa 0
        judge = [True, True, True, True]
        human = [True, False, True, False]
        assert _cohens_kappa(judge, human) == pytest.approx(0.0)

    def test_kappa_unanimous_raters(self) -> None:
        assert _cohens_kappa([True, True], [True, True]) == 1.0
        assert _cohens_kappa([True, True], [False, False]) == 0.0

    def test_wilson_interval_brackets_the_proportion(self) -> None:
        low, high = _wilson_interval(80, 100)
        assert low < 0.8 < high
        assert (high - low) < 0.2

    def test_wilson_interval_widens_with_small_n(self) -> None:
        low_small, high_small = _wilson_interval(4, 5)
        low_large, high_large = _wilson_interval(80, 100)
        assert (high_small - low_small) > (high_large - low_large)

    @pytest.mark.parametrize(
        ("kappa", "label"),
        [
            (0.9, "almost perfect"),
            (0.7, "substantial"),
            (0.5, "moderate"),
            (0.3, "fair"),
            (0.1, "slight"),
            (-0.2, "poor"),
        ],
    )
    def test_kappa_interpretation(self, kappa: float, label: str) -> None:
        assert interpret_kappa(kappa) == label


class TestCalibration:
    """Test suite for JudgeCalibrator.calibrate."""

    @pytest.mark.asyncio
    async def test_perfect_judge_scores_kappa_one(self) -> None:
        samples = [make_sample(human_pass=True)] * 10 + [make_sample(human_pass=False)] * 10
        judge = ScriptedJudge({"faithfulness": [0.9] * 10 + [0.2] * 10})

        report = await JudgeCalibrator(judge).calibrate(samples)  # type: ignore[arg-type]

        c = report.criteria["faithfulness"]
        assert c.accuracy == 1.0
        assert c.kappa == 1.0
        assert c.kappa_interpretation == "almost perfect"
        assert c.disagreements == []
        assert c.false_accept_rate == 0.0
        assert c.false_reject_rate == 0.0

    @pytest.mark.asyncio
    async def test_error_rates_split_by_direction(self) -> None:
        # 2 human-fails that the judge passes (false accepts)
        # 1 human-pass that the judge fails (false reject)
        samples = (
            [make_sample(human_pass=False)] * 2 + [make_sample(human_pass=True)] + [make_sample(human_pass=True)] * 7
        )
        judge = ScriptedJudge({"faithfulness": [0.9, 0.9, 0.2] + [0.9] * 7})

        report = await JudgeCalibrator(judge).calibrate(samples)  # type: ignore[arg-type]

        c = report.criteria["faithfulness"]
        assert c.false_accept_rate == 1.0  # both human-fails slipped through
        assert c.false_reject_rate == pytest.approx(1 / 8)
        assert sorted(c.disagreements) == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_threshold_recommendation_moves_when_data_supports_it(self) -> None:
        # Judge is consistent but shifted: bad answers get ~0.75, good ~0.95.
        # At the default 0.7 everything passes; at 0.8+ agreement is perfect.
        samples = [make_sample(human_pass=True)] * 10 + [make_sample(human_pass=False)] * 10
        judge = ScriptedJudge({"faithfulness": [0.95] * 10 + [0.75] * 10})

        report = await JudgeCalibrator(judge).calibrate(samples)  # type: ignore[arg-type]

        c = report.criteria["faithfulness"]
        assert c.kappa == pytest.approx(0.0)  # at default threshold: useless
        assert c.recommended_threshold > 0.75
        assert c.kappa_at_recommended == 1.0

    @pytest.mark.asyncio
    async def test_threshold_recommendation_stays_put_on_ties(self) -> None:
        samples = [make_sample(human_pass=True)] * 10 + [make_sample(human_pass=False)] * 10
        judge = ScriptedJudge({"faithfulness": [0.9] * 10 + [0.2] * 10})

        report = await JudgeCalibrator(judge).calibrate(samples)  # type: ignore[arg-type]

        # Many thresholds give kappa 1.0; the current one wins ties
        assert report.criteria["faithfulness"].recommended_threshold == 0.7

    @pytest.mark.asyncio
    async def test_per_criterion_human_overrides(self) -> None:
        # Grounded but off-topic: faithful yes, relevant no
        samples = [make_sample(human_pass=True, human_labels={"relevance": False}) for _ in range(20)]
        judge = ScriptedJudge(
            {"faithfulness": [0.9] * 20, "relevance": [0.2] * 20},
        )

        report = await JudgeCalibrator(judge).calibrate(samples, criteria=["faithfulness", "relevance"])  # type: ignore[arg-type]

        assert report.criteria["faithfulness"].accuracy == 1.0
        assert report.criteria["relevance"].accuracy == 1.0  # judge failed them, human agreed

    @pytest.mark.asyncio
    async def test_small_sample_is_flagged(self) -> None:
        samples = [make_sample()] * 5
        judge = ScriptedJudge({"faithfulness": [0.9] * 5})

        report = await JudgeCalibrator(judge).calibrate(samples)  # type: ignore[arg-type]

        assert report.insufficient_data is True
        assert "WARNING" in report.summary()

    @pytest.mark.asyncio
    async def test_empty_samples_raise(self) -> None:
        judge = ScriptedJudge({"faithfulness": []})

        with pytest.raises(ValueError, match="zero labeled samples"):
            await JudgeCalibrator(judge).calibrate([])  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_unknown_criterion_raises(self) -> None:
        judge = ScriptedJudge({"faithfulness": [0.9]})

        with pytest.raises(ValueError, match="Unknown criterion"):
            await JudgeCalibrator(judge).calibrate([make_sample()], criteria=["vibes"])  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_accepts_calibration_set_and_serializes(self, tmp_path) -> None:
        calset = CalibrationSet(
            name="medical-v1",
            samples=[make_sample(human_pass=True)] * 12 + [make_sample(human_pass=False)] * 8,
        )
        path = tmp_path / "calset.json"
        calset.save(path)
        loaded = CalibrationSet.load(path)
        assert len(loaded) == 20

        judge = ScriptedJudge({"faithfulness": [0.9] * 12 + [0.2] * 8})
        report = await JudgeCalibrator(judge).calibrate(loaded)  # type: ignore[arg-type]

        data = report.to_dict()
        assert data["n_samples"] == 20
        assert data["insufficient_data"] is False
        assert data["criteria"]["faithfulness"]["kappa"] == 1.0
        assert "kappa 1.00 (almost perfect)" in report.summary()
