"""Calibrate an LLM judge against human labels.

The number-one objection to LLM-as-judge — especially with local models —
is "can I trust the judge?". Calibration answers it with data: label a
small set of (question, context, answer) triples yourself, run the judge
on them, and measure agreement.

    >>> from ragnarok_ai import JudgeCalibrator, LLMJudge
    >>> from ragnarok_ai.calibration import CalibrationSet
    >>>
    >>> samples = CalibrationSet.load("labeled-samples.json")
    >>> calibrator = JudgeCalibrator(LLMJudge(medical_mode=True))
    >>> report = await calibrator.calibrate(samples, criteria=["faithfulness"])
    >>> print(report.summary())

All statistics are dependency-free: Cohen's kappa for chance-corrected
agreement, a Wilson interval on accuracy, and a threshold sweep that
recommends the pass cutoff maximizing kappa on your labels.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ragnarok_ai.calibration.models import (
    CalibrationReport,
    CalibrationSample,
    CalibrationSet,
    CriterionCalibration,
    interpret_kappa,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from ragnarok_ai.evaluators.judge import JudgeResult, LLMJudge

# Default pass threshold, matching the documented judge verdict rule (PASS >= 0.7)
DEFAULT_PASS_THRESHOLD = 0.7

# Below this many labels, statistics are flagged as indicative only
DEFAULT_MIN_SAMPLES = 20

CRITERIA = ("faithfulness", "relevance", "hallucination", "completeness")


def _cohens_kappa(judge: Sequence[bool], human: Sequence[bool]) -> float:
    """Cohen's kappa for two binary raters."""
    n = len(judge)
    po = sum(1 for j, h in zip(judge, human, strict=True) if j == h) / n
    p_yes = (sum(judge) / n) * (sum(human) / n)
    p_no = (1 - sum(judge) / n) * (1 - sum(human) / n)
    pe = p_yes + p_no
    if pe >= 1.0:  # both raters unanimous
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


class JudgeCalibrator:
    """Measure judge-vs-human agreement on a labeled sample set.

    Attributes:
        judge: The LLM judge to calibrate.
        pass_threshold: Score at or above which the judge's answer counts
            as PASS (default 0.7, the documented verdict rule).
        min_samples: Below this many labels the report is flagged
            ``insufficient_data``.

    Example:
        >>> calibrator = JudgeCalibrator(judge)
        >>> report = await calibrator.calibrate(samples)
        >>> report.criteria["faithfulness"].kappa
        0.72
    """

    def __init__(
        self,
        judge: LLMJudge,
        *,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ) -> None:
        """Initialize the calibrator.

        Args:
            judge: The judge to calibrate.
            pass_threshold: Pass cutoff applied to judge scores.
            min_samples: Sample-size floor for trustworthy statistics.
        """
        self.judge = judge
        self.pass_threshold = pass_threshold
        self.min_samples = min_samples

    async def calibrate(
        self,
        samples: CalibrationSet | Sequence[CalibrationSample],
        *,
        criteria: Sequence[str] = ("faithfulness",),
    ) -> CalibrationReport:
        """Run the judge on labeled samples and measure agreement.

        Args:
            samples: The human-labeled examples (a ``CalibrationSet`` or
                a plain sequence of ``CalibrationSample``).
            criteria: Judge criteria to calibrate — any of "faithfulness",
                "relevance", "hallucination", "completeness". Each costs
                one judge call per sample.

        Returns:
            A ``CalibrationReport`` with per-criterion agreement.

        Raises:
            ValueError: If *samples* is empty or a criterion is unknown.
        """
        sample_list = list(samples.samples) if isinstance(samples, CalibrationSet) else list(samples)
        if not sample_list:
            msg = "Cannot calibrate on zero labeled samples"
            raise ValueError(msg)
        for criterion in criteria:
            if criterion not in CRITERIA:
                msg = f"Unknown criterion {criterion!r}; expected one of {CRITERIA}"
                raise ValueError(msg)

        results: dict[str, CriterionCalibration] = {}
        for criterion in criteria:
            scores = [(await self._judge_call(criterion, s)).score for s in sample_list]
            human = [s.label_for(criterion) for s in sample_list]
            results[criterion] = self._analyze(criterion, scores, human)

        return CalibrationReport(
            criteria=results,
            n_samples=len(sample_list),
            insufficient_data=len(sample_list) < self.min_samples,
        )

    # ── Judge dispatch ───────────────────────────────────────────────────

    def _judge_call(self, criterion: str, s: CalibrationSample) -> Awaitable[JudgeResult]:
        """One judge evaluation of *s* for *criterion*."""
        calls: dict[str, Callable[[], Awaitable[JudgeResult]]] = {
            "faithfulness": lambda: self.judge.evaluate_faithfulness(
                context=s.context, question=s.question, answer=s.answer
            ),
            "relevance": lambda: self.judge.evaluate_relevance(question=s.question, answer=s.answer),
            "hallucination": lambda: self.judge.detect_hallucination(context=s.context, answer=s.answer),
            "completeness": lambda: self.judge.evaluate_completeness(
                question=s.question, answer=s.answer, context=s.context
            ),
        }
        return calls[criterion]()

    # ── Statistics ───────────────────────────────────────────────────────

    def _analyze(self, criterion: str, scores: list[float], human: list[bool]) -> CriterionCalibration:
        """Agreement statistics for one criterion."""
        judge_pass = [score >= self.pass_threshold for score in scores]
        n = len(scores)

        agreements = sum(1 for j, h in zip(judge_pass, human, strict=True) if j == h)
        human_fails = sum(1 for h in human if not h)
        human_passes = n - human_fails
        false_accepts = sum(1 for j, h in zip(judge_pass, human, strict=True) if j and not h)
        false_rejects = sum(1 for j, h in zip(judge_pass, human, strict=True) if not j and h)

        recommended, kappa_at_recommended = self._best_threshold(scores, human)

        return CriterionCalibration(
            criterion=criterion,
            n=n,
            accuracy=agreements / n,
            accuracy_ci=_wilson_interval(agreements, n),
            kappa=_cohens_kappa(judge_pass, human),
            kappa_interpretation=interpret_kappa(_cohens_kappa(judge_pass, human)),
            false_accept_rate=false_accepts / human_fails if human_fails else 0.0,
            false_reject_rate=false_rejects / human_passes if human_passes else 0.0,
            threshold=self.pass_threshold,
            recommended_threshold=recommended,
            kappa_at_recommended=kappa_at_recommended,
            disagreements=[i for i, (j, h) in enumerate(zip(judge_pass, human, strict=True)) if j != h],
        )

    def _best_threshold(self, scores: list[float], human: list[bool]) -> tuple[float, float]:
        """Sweep pass thresholds and return the one maximizing kappa.

        Ties prefer the threshold closest to the current one, so the
        recommendation only moves when the data supports it.
        """
        candidates = sorted({round(t / 20, 2) for t in range(1, 20)} | {self.pass_threshold})
        best = (self.pass_threshold, _cohens_kappa([s >= self.pass_threshold for s in scores], human))
        for threshold in candidates:
            kappa = _cohens_kappa([s >= threshold for s in scores], human)
            better = kappa > best[1] + 1e-9
            as_good_but_closer = abs(kappa - best[1]) <= 1e-9 and abs(threshold - self.pass_threshold) < abs(
                best[0] - self.pass_threshold
            )
            if better or as_good_but_closer:
                best = (threshold, kappa)
        return best
