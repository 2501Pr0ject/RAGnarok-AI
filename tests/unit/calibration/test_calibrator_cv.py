"""Calibrator: protocol path, policy threshold, out-of-sample kappa, bootstrap, baselines, merge."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragnarok_ai.calibration.calibrator import (
    DEFAULT_PASS_THRESHOLD,
    JudgeCalibrator,
    _best_threshold,
    analyze_criterion,
)
from ragnarok_ai.calibration.models import CalibrationSample
from ragnarok_ai.core.exceptions import LLMConnectionError
from ragnarok_ai.evaluators.judge import JudgeResult, LLMJudge
from ragnarok_ai.judges import JudgeConfig, OrdinalBands, OrdinalScore, ScoreThreshold


def sample(human_pass: bool, score: float, group: str | None = None) -> CalibrationSample:
    """The answer text carries the score the scripted judge will return."""
    return CalibrationSample(question="q", context="c", answer=f"s{score}", human_pass=human_pass, group_id=group)


class ProtocolJudge:
    """A JudgeProtocol judge returning the score encoded in the answer."""

    criteria = frozenset({"faithfulness", "relevance"})

    def __init__(self, model: str = "scripted", policy=None) -> None:
        self.model = model
        self.policy = policy or OrdinalBands(4, 2)

    def config(self, criterion: str) -> JudgeConfig:
        return JudgeConfig(
            judge_class="tests.ProtocolJudge",
            judge_kind="custom",
            model=self.model,
            criterion=criterion,
            verdict_policy=self.policy,
            ragnarok_ai_version="test",
        )

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:  # noqa: ARG002
        if answer == "backend":
            raise LLMConnectionError("Ollama API error: 500")
        score = float(answer[1:])
        raw = round(score * 4) + 1
        return JudgeResult(
            verdict="PASS" if score >= 0.75 else "FAIL",
            score=score,
            explanation="",
            criteria=criterion,
            raw_score=raw,
            output=OrdinalScore(max(1, min(5, raw)), 1, 5),
        )


def study_matrix() -> tuple[list[float], list[bool]]:
    """The study's relevance matrix for Prometheus on the test split: 34 TP, 24 FN, 2 FP, 2 TN."""
    scores = [1.0] * 34 + [0.0] * 24 + [1.0] * 2 + [0.0] * 2
    human = [True] * 58 + [False] * 4
    return scores, human


class TestStudyReference:
    def test_reference_numbers(self):
        scores, human = study_matrix()
        c = analyze_criterion("relevance", scores, human, threshold=0.75)

        assert c.n == 62
        assert c.accuracy == pytest.approx(36 / 62)  # 58.1%
        assert c.kappa == pytest.approx(0.024, abs=5e-4)
        assert c.majority_baseline_accuracy == pytest.approx(58 / 62)  # 93.5%
        assert c.human_pass_rate == pytest.approx(58 / 62)
        assert c.minority_count == 4
        assert c.kappa_interpretable is False
        assert c.false_accept_rate == pytest.approx(2 / 4)
        assert c.false_reject_rate == pytest.approx(24 / 58)

    def test_minority_smaller_than_k_falls_back_to_leave_one_out(self):
        scores, human = study_matrix()
        c = analyze_criterion("relevance", scores, human, threshold=0.75)

        assert c.cv_method == "leave-one-out (minority class smaller than 5 folds)"
        assert c.cv_folds == 62
        assert c.kappa_at_recommended_cv is not None

    def test_bootstrap_excludes_single_class_draws_and_stays_reliable_here(self):
        scores, human = study_matrix()
        c = analyze_criterion("relevance", scores, human, threshold=0.75, bootstrap_resamples=1000)

        # P(no FAIL in a draw of 62) = (58/62)^62 ~ 1.6%: a few exclusions, well under 10%
        assert 0 < c.bootstrap_excluded < 100
        assert c.bootstrap_reliable is True
        assert c.kappa_cv_ci is not None and c.kappa_cv_ci[0] <= c.kappa_cv_ci[1]
        assert c.bootstrap_resamples == 1000

    def test_summary_flags_kappa_as_not_interpretable(self):
        scores, human = study_matrix()
        c = analyze_criterion("relevance", scores, human, threshold=0.75)
        from ragnarok_ai.calibration.models import CalibrationReport

        text = CalibrationReport(criteria={"relevance": c}, n_samples=62, insufficient_data=False).summary()
        assert "kappa not interpretable, the minority class has only 4 samples" in text
        assert "majority-class judge would agree 93.5%" in text
        assert "human PASS prevalence 93.5%" in text


class TestCrossValidationAndBootstrap:
    def test_stratified_five_fold_when_both_classes_are_large_enough(self):
        scores = [1.0] * 25 + [0.0] * 5 + [0.0] * 8 + [1.0] * 2
        human = [True] * 30 + [False] * 10
        c = analyze_criterion("faithfulness", scores, human, threshold=0.75)
        assert c.cv_method == "stratified-5-fold"
        assert c.cv_folds == 5

    def test_leave_one_out_below_forty_samples(self):
        scores = [1.0] * 15 + [0.0] * 15
        human = [True] * 15 + [False] * 15
        c = analyze_criterion("faithfulness", scores, human, threshold=0.75)
        assert c.cv_method == "leave-one-out"
        assert c.cv_folds == 30

    def test_bootstrap_is_marked_unreliable_when_too_many_draws_are_excluded(self):
        # one FAIL among 20: P(no FAIL in a draw) = 0.95^20 ~ 36% of draws excluded
        scores = [1.0] * 19 + [0.0]
        human = [True] * 19 + [False]
        c = analyze_criterion("faithfulness", scores, human, threshold=0.75, bootstrap_resamples=500)
        assert c.bootstrap_excluded > 50
        assert c.bootstrap_reliable is False
        assert c.kappa_cv_ci is not None  # an interval is still computed on the usable draws

    def test_no_usable_draw_gives_no_interval(self):
        scores = [1.0, 0.0, 1.0]
        human = [True, True, True]
        c = analyze_criterion("faithfulness", scores, human, threshold=0.75, bootstrap_resamples=50)
        assert c.kappa_cv_ci is None
        assert c.bootstrap_excluded == 50
        assert c.bootstrap_reliable is False

    def test_seeds_make_reports_identical(self):
        scores = [1.0] * 25 + [0.0] * 5 + [0.0] * 8 + [1.0] * 2
        human = [True] * 30 + [False] * 10
        a = analyze_criterion("faithfulness", scores, human, threshold=0.75, cv_seed=3, bootstrap_seed=9)
        b = analyze_criterion("faithfulness", scores, human, threshold=0.75, cv_seed=3, bootstrap_seed=9)
        other = analyze_criterion("faithfulness", scores, human, threshold=0.75, cv_seed=4, bootstrap_seed=9)
        assert a == b
        assert (a.cv_seed, a.bootstrap_seed) == (3, 9)
        assert a.kappa == other.kappa  # the in-sample statistics do not depend on seeds

    def test_grouped_bootstrap_keeps_groups_together(self):
        # Two samples per group; a group-level draw cannot separate them
        scores = [1.0, 1.0, 0.0, 0.0] * 5
        human = [True, True, False, False] * 5
        groups = [f"g{i // 2}" for i in range(20)]
        c = analyze_criterion("faithfulness", scores, human, threshold=0.75, group_ids=groups, bootstrap_resamples=200)
        assert c.group_ids == tuple(groups)
        assert c.kappa_cv_ci is not None


class TestExistingFieldsUnchanged:
    def test_in_sample_fields_match_the_sweep(self):
        scores = [0.9, 0.8, 0.75, 0.1, 0.3, 0.6, 0.95, 0.2]
        human = [True, True, False, False, True, False, True, False]
        c = analyze_criterion("faithfulness", scores, human, threshold=0.7)
        recommended, kappa_in_sample = _best_threshold(scores, human, 0.7)

        assert (c.recommended_threshold, c.kappa_at_recommended) == (recommended, kappa_in_sample)
        assert c.threshold == 0.7
        judge_pass = [s >= 0.7 for s in scores]
        assert c.accuracy == sum(1 for j, h in zip(judge_pass, human, strict=True) if j == h) / len(scores)

    @pytest.mark.parametrize("scores", [[0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.25, 1.0, 0.5, 0.0]])
    def test_seventy_and_seventy_five_give_identical_verdicts_on_discrete_scores(self, scores):
        """The historical 0.7 fallback and the policy's 0.75 agree on every Prometheus score."""
        human = [False, False, True, True, True, False, False, True, False, True]
        at_70 = analyze_criterion("faithfulness", scores, human, threshold=0.7)
        at_75 = analyze_criterion("faithfulness", scores, human, threshold=0.75)
        assert [s >= 0.7 for s in scores] == [s >= 0.75 for s in scores]
        assert (at_70.accuracy, at_70.kappa, at_70.disagreements) == (at_75.accuracy, at_75.kappa, at_75.disagreements)


class TestProtocolPath:
    @pytest.mark.asyncio
    async def test_threshold_comes_from_the_policy(self):
        samples = [sample(True, 0.9), sample(False, 0.6), sample(True, 0.75), sample(False, 0.2)]
        ordinal = JudgeCalibrator(ProtocolJudge(), min_samples=1)
        binary = JudgeCalibrator(ProtocolJudge(policy=ScoreThreshold(0.5)), min_samples=1)

        assert ordinal.uses_protocol is True
        assert (await ordinal.calibrate(samples)).criteria["faithfulness"].threshold == 0.75
        assert (await binary.calibrate(samples)).criteria["faithfulness"].threshold == 0.5
        assert JudgeCalibrator(ProtocolJudge(), pass_threshold=0.6, min_samples=1).threshold_for("faithfulness") == 0.6

    @pytest.mark.asyncio
    async def test_llmjudge_gets_seventy_five(self):
        llm = MagicMock(spec=["generate", "embed", "model"])
        llm.model = "mistral:7b"
        llm.generate = AsyncMock(return_value="Score: 4\nExplanation: ok")
        calibrator = JudgeCalibrator(LLMJudge(llm=llm), min_samples=1)

        report = await calibrator.calibrate([sample(True, 0.0), sample(False, 0.0)])

        c = report.criteria["faithfulness"]
        assert c.threshold == 0.75
        assert report.configs["faithfulness"].model == "mistral:7b"

    def test_non_protocol_judges_keep_the_fallback(self):
        class Legacy:
            async def evaluate_faithfulness(self, context, question, answer):  # noqa: ARG002
                return JudgeResult("PASS", 1.0, "", "faithfulness", 5)

        calibrator = JudgeCalibrator(Legacy())  # type: ignore[arg-type]
        assert calibrator.uses_protocol is False
        assert calibrator.threshold_for("faithfulness") == DEFAULT_PASS_THRESHOLD == 0.7

    @pytest.mark.asyncio
    async def test_backend_failure_is_counted_apart_and_excluded(self):
        samples = [
            sample(True, 0.9),
            CalibrationSample(question="q", context="c", answer="backend", human_pass=True),
            sample(False, 0.1),
        ]
        report = await JudgeCalibrator(ProtocolJudge(), min_samples=1).calibrate(samples)
        c = report.criteria["faithfulness"]

        assert c.n == 2 and c.backend_failures == 1 and c.parse_failures == 0
        assert c.failed_indices == [1]
        assert c.n_labeled == 3

    @pytest.mark.asyncio
    async def test_two_identical_calibrations_give_identical_reports(self):
        samples = [sample(True, 0.9), sample(False, 0.6), sample(True, 0.75), sample(False, 0.2), sample(True, 1.0)]
        a = await JudgeCalibrator(ProtocolJudge(), min_samples=1).calibrate(samples)
        b = await JudgeCalibrator(ProtocolJudge(), min_samples=1).calibrate(samples)
        da, db = a.to_dict(), b.to_dict()
        da.pop("timestamp"), db.pop("timestamp")
        assert da == db


class TestMerge:
    @pytest.mark.asyncio
    async def test_merge_recomputes_on_concatenated_samples(self):
        first = await JudgeCalibrator(ProtocolJudge(), min_samples=1).calibrate([sample(True, 0.9), sample(False, 0.2)])
        second = await JudgeCalibrator(ProtocolJudge(), min_samples=1).calibrate(
            [sample(True, 0.8), sample(False, 0.9)]
        )

        assert first.comparable_with(second)
        merged = first.merge(second)
        assert merged.n_samples == 4
        assert merged.criteria["faithfulness"].n == 4
        assert merged.criteria["faithfulness"].disagreements == [3]

    @pytest.mark.asyncio
    async def test_merge_refuses_different_configurations(self):
        first = await JudgeCalibrator(ProtocolJudge("model-a"), min_samples=1).calibrate(
            [sample(True, 0.9), sample(False, 0.2)]
        )
        second = await JudgeCalibrator(ProtocolJudge("model-b"), min_samples=1).calibrate(
            [sample(True, 0.9), sample(False, 0.2)]
        )

        assert not first.comparable_with(second)
        with pytest.raises(ValueError, match="not comparable"):
            first.merge(second)

    def test_reports_without_configuration_are_never_comparable(self):
        from ragnarok_ai.calibration.models import CalibrationReport

        scores, human = study_matrix()
        c = analyze_criterion("relevance", scores, human, threshold=0.7)
        a = CalibrationReport(criteria={"relevance": c}, n_samples=62, insufficient_data=False)
        assert not a.comparable_with(a)
