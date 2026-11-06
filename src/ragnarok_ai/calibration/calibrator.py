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

Two estimates of the recommended threshold's kappa are reported:

- ``kappa_at_recommended``: in-sample, measured on the very labels that
  selected the threshold, hence optimistic;
- ``kappa_at_recommended_cv``: out-of-sample, from stratified 5-fold
  cross-validation (leave-one-out below 40 samples, or when a class is
  smaller than the number of folds; by group when samples carry a
  ``group_id``), with a bootstrap confidence interval. The interval
  resamples the out-of-fold predictions without repeating the
  cross-validation, so it slightly underestimates the uncertainty. Both
  use fixed seeds, recorded in the report: two identical calibrations
  give identical reports.

Judges implementing ``JudgeProtocol`` run through ``judge_many``; the pass
threshold then comes from the judge's verdict policy (0.75 for the 1-5
scale of ``LLMJudge``, where ``raw >= 4`` is a PASS). Objects exposing only
the four named methods keep the historical path and ``DEFAULT_PASS_THRESHOLD``.

A judge response that cannot be parsed does not abort the calibration: the
sample is counted in ``parse_failures`` and excluded from the statistics,
which are computed on the scored samples only.
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING, Any

from ragnarok_ai.calibration.models import (
    CalibrationReport,
    CalibrationSample,
    CalibrationSet,
    CriterionCalibration,
    interpret_kappa,
)
from ragnarok_ai.core.exceptions import EvaluationError
from ragnarok_ai.core.protocols import JudgeProtocol
from ragnarok_ai.judges.runner import judge_many

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from ragnarok_ai.evaluators.judge import JudgeResult, LLMJudge
    from ragnarok_ai.judges.contract import JudgeConfig

# Fallback pass threshold for judges without a verdict policy. The documented
# rule of LLMJudge is raw >= 4 on the 1-5 scale, i.e. 0.75 once normalized;
# on those discrete scores (0, 0.25, 0.5, 0.75, 1) a cutoff of 0.7 gives
# exactly the same verdicts, which is why this value never changed results.
DEFAULT_PASS_THRESHOLD = 0.7

logger = logging.getLogger(__name__)

# Below this many labels, statistics are flagged as indicative only
DEFAULT_MIN_SAMPLES = 20

# Cross-validation: stratified K-fold above this many scored samples, leave-one-out below
CV_FOLDS = 5
CV_MIN_SAMPLES_FOR_KFOLD = 40

# Bootstrap of the out-of-sample kappa
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_MAX_EXCLUDED_SHARE = 0.10

# Below this many samples in the minority class, kappa is not interpretable
MIN_MINORITY_FOR_KAPPA = 10

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


def _best_threshold(scores: Sequence[float], human: Sequence[bool], current: float) -> tuple[float, float]:
    """Sweep pass thresholds and return the one maximizing kappa, with that kappa.

    In-sample: the returned kappa is measured on the very labels that
    selected the threshold, so it is optimistic. Ties prefer the threshold
    closest to *current*, so the recommendation only moves when the data
    supports it.
    """
    candidates = sorted({round(t / 20, 2) for t in range(1, 20)} | {current})
    best = (current, _cohens_kappa([s >= current for s in scores], human))
    for threshold in candidates:
        kappa = _cohens_kappa([s >= threshold for s in scores], human)
        better = kappa > best[1] + 1e-9
        as_good_but_closer = abs(kappa - best[1]) <= 1e-9 and abs(threshold - current) < abs(best[0] - current)
        if better or as_good_but_closer:
            best = (threshold, kappa)
    return best


def _units(n: int, group_ids: Sequence[str | None] | None) -> list[list[int]]:
    """Resampling units: one per group when group ids are given, one per sample otherwise."""
    groups: dict[str, list[int]] = {}
    for i in range(n):
        g = group_ids[i] if group_ids is not None else None
        groups.setdefault(g if g is not None else f"__sample_{i}", []).append(i)
    return list(groups.values())


def _folds(
    human: Sequence[bool], seed: int, group_ids: Sequence[str | None] | None = None
) -> tuple[list[list[int]], str]:
    """Fold assignment for cross-validation, and the method's name.

    Units are groups when ``group_ids`` are given (samples of one group
    always share a fold), samples otherwise. Stratified ``CV_FOLDS``-fold
    on the unit's majority label when there are at least
    ``CV_MIN_SAMPLES_FOR_KFOLD`` samples and each class has at least
    ``CV_FOLDS`` units; leave-one-unit-out otherwise, with the reason in
    the method name. The shuffle inside each class uses *seed*.
    """
    n = len(human)
    units = _units(n, group_ids)
    grouped = group_ids is not None and len(units) < n
    suffix = " (by group)" if grouped else ""
    one_out = "leave-one-group-out" if grouped else "leave-one-out"

    def majority(unit: list[int]) -> bool:
        passes = sum(1 for i in unit if human[i])
        return passes * 2 >= len(unit)

    positives = [u for u in units if majority(u)]
    negatives = [u for u in units if not majority(u)]
    if n < CV_MIN_SAMPLES_FOR_KFOLD:
        return units, one_out
    if min(len(positives), len(negatives)) < CV_FOLDS:
        return units, f"{one_out} (minority class smaller than {CV_FOLDS} folds)"
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(CV_FOLDS)]
    for members in (positives, negatives):
        shuffled = list(members)
        rng.shuffle(shuffled)
        for k, unit in enumerate(shuffled):
            folds[k % CV_FOLDS].extend(unit)
    return [sorted(f) for f in folds], f"stratified-{CV_FOLDS}-fold{suffix}"


def _cross_validated_predictions(
    scores: Sequence[float],
    human: Sequence[bool],
    current: float,
    seed: int,
    group_ids: Sequence[str | None] | None = None,
) -> tuple[list[bool], str, int]:
    """Out-of-fold PASS predictions: the threshold is chosen on the other folds only."""
    folds, method = _folds(human, seed, group_ids)
    predictions: list[bool | None] = [None] * len(scores)
    for fold in folds:
        held = set(fold)
        train_scores = [s for i, s in enumerate(scores) if i not in held]
        train_human = [h for i, h in enumerate(human) if i not in held]
        threshold, _ = _best_threshold(train_scores, train_human, current)
        for i in fold:
            predictions[i] = scores[i] >= threshold
    return [bool(p) for p in predictions], method, len(folds)


def _bootstrap_kappa(
    judge_pass: Sequence[bool],
    human: Sequence[bool],
    group_ids: Sequence[str | None],
    seed: int,
    resamples: int,
) -> tuple[tuple[float, float] | None, int, bool | None]:
    """Percentile 95% interval of kappa over *resamples* bootstrap draws.

    The draws resample the out-of-fold predictions as they are; the
    cross-validation is not repeated inside each draw, so the interval
    reflects the sampling of the labels but not the variability of the
    threshold choice, and slightly underestimates the uncertainty.

    Draws are made by group when ``group_ids`` are present (several samples
    from the same question stay together), by sample otherwise. A draw that
    contains a single human class leaves kappa undefined: it is excluded and
    counted. The interval is marked unreliable when more than
    ``BOOTSTRAP_MAX_EXCLUDED_SHARE`` of the draws were excluded, and ``None``
    when no draw could be used.
    """
    n = len(human)
    if n < 2 or resamples < 1:
        return None, 0, None
    units = _units(n, group_ids)
    rng = random.Random(seed)
    kappas: list[float] = []
    excluded = 0
    for _ in range(resamples):
        drawn = [i for _ in range(len(units)) for i in units[rng.randrange(len(units))]]
        drawn_human = [human[i] for i in drawn]
        if all(drawn_human) or not any(drawn_human):
            excluded += 1
            continue
        kappas.append(_cohens_kappa([judge_pass[i] for i in drawn], drawn_human))
    if not kappas:
        return None, excluded, False
    kappas.sort()
    lo = kappas[max(0, math.floor(0.025 * (len(kappas) - 1)))]
    hi = kappas[min(len(kappas) - 1, math.ceil(0.975 * (len(kappas) - 1)))]
    return (lo, hi), excluded, excluded <= BOOTSTRAP_MAX_EXCLUDED_SHARE * resamples


def analyze_criterion(
    criterion: str,
    all_scores: Sequence[float | None],
    all_human: Sequence[bool],
    *,
    threshold: float,
    group_ids: Sequence[str | None] | None = None,
    cv_seed: int = 0,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    backend_failures: int = 0,
) -> CriterionCalibration:
    """Agreement statistics for one criterion, on the scored samples only.

    ``None`` scores are failed samples: counted, excluded from every
    statistic, and listed in ``failed_indices``. Indices in
    ``disagreements`` and ``failed_indices`` refer to the input samples.
    """
    groups = list(group_ids) if group_ids is not None else [None] * len(all_scores)
    failed_indices = [i for i, score in enumerate(all_scores) if score is None]
    scored = [
        (i, score, h) for i, (score, h) in enumerate(zip(all_scores, all_human, strict=True)) if score is not None
    ]
    indices = [i for i, _, _ in scored]
    scores = [score for _, score, _ in scored]
    human = [h for _, _, h in scored]
    n = len(scores)
    parse_failures = len(failed_indices) - backend_failures

    if n == 0:
        return CriterionCalibration(
            criterion=criterion,
            n=0,
            accuracy=0.0,
            accuracy_ci=(0.0, 1.0),
            kappa=0.0,
            kappa_interpretation="no scored samples",
            false_accept_rate=0.0,
            false_reject_rate=0.0,
            threshold=threshold,
            recommended_threshold=threshold,
            kappa_at_recommended=0.0,
            disagreements=[],
            parse_failures=parse_failures,
            failed_indices=failed_indices,
            backend_failures=backend_failures,
            kappa_interpretable=False,
            cv_seed=cv_seed,
            bootstrap_seed=bootstrap_seed,
            judge_scores=tuple(all_scores),
            human_labels=tuple(all_human),
            group_ids=tuple(groups),
        )

    judge_pass = [score >= threshold for score in scores]

    agreements = sum(1 for j, h in zip(judge_pass, human, strict=True) if j == h)
    human_fails = sum(1 for h in human if not h)
    human_passes = n - human_fails
    false_accepts = sum(1 for j, h in zip(judge_pass, human, strict=True) if j and not h)
    false_rejects = sum(1 for j, h in zip(judge_pass, human, strict=True) if not j and h)

    recommended, kappa_at_recommended = _best_threshold(scores, human, threshold)

    minority = min(human_passes, human_fails)
    kappa_value = _cohens_kappa(judge_pass, human)

    scored_groups = [groups[i] for i in indices]
    has_groups = any(g is not None for g in scored_groups)
    oof_pass, cv_method, cv_folds = _cross_validated_predictions(
        scores, human, threshold, cv_seed, scored_groups if has_groups else None
    )
    kappa_cv = _cohens_kappa(oof_pass, human)
    ci, excluded, reliable = _bootstrap_kappa(oof_pass, human, scored_groups, bootstrap_seed, bootstrap_resamples)

    return CriterionCalibration(
        criterion=criterion,
        n=n,
        accuracy=agreements / n,
        accuracy_ci=_wilson_interval(agreements, n),
        kappa=kappa_value,
        kappa_interpretation=interpret_kappa(kappa_value),
        false_accept_rate=false_accepts / human_fails if human_fails else 0.0,
        false_reject_rate=false_rejects / human_passes if human_passes else 0.0,
        threshold=threshold,
        recommended_threshold=recommended,
        kappa_at_recommended=kappa_at_recommended,
        disagreements=[indices[k] for k, (j, h) in enumerate(zip(judge_pass, human, strict=True)) if j != h],
        parse_failures=parse_failures,
        failed_indices=failed_indices,
        backend_failures=backend_failures,
        human_pass_rate=human_passes / n,
        majority_baseline_accuracy=max(human_passes, human_fails) / n,
        minority_count=minority,
        kappa_interpretable=minority >= MIN_MINORITY_FOR_KAPPA,
        kappa_at_recommended_cv=kappa_cv,
        cv_method=cv_method,
        cv_folds=cv_folds,
        kappa_cv_ci=ci,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_excluded=excluded,
        bootstrap_reliable=reliable,
        cv_seed=cv_seed,
        bootstrap_seed=bootstrap_seed,
        judge_scores=tuple(all_scores),
        human_labels=tuple(all_human),
        group_ids=tuple(groups),
    )


class JudgeCalibrator:
    """Measure judge-vs-human agreement on a labeled sample set.

    Attributes:
        judge: The LLM judge to calibrate.
        pass_threshold: Score at or above which the judge's answer counts
            as PASS. ``None`` (default) takes it from the judge's verdict
            policy when the judge implements ``JudgeProtocol`` (0.75 for
            ``LLMJudge``), and ``DEFAULT_PASS_THRESHOLD`` otherwise.
        min_samples: Below this many labels the report is flagged
            ``insufficient_data``.
        cv_seed: Seed of the cross-validation fold assignment.
        bootstrap_seed: Seed of the bootstrap draws.
        bootstrap_resamples: Number of bootstrap draws.

    Example:
        >>> calibrator = JudgeCalibrator(judge)
        >>> report = await calibrator.calibrate(samples)
        >>> report.criteria["faithfulness"].kappa
        0.72
    """

    def __init__(
        self,
        judge: LLMJudge | JudgeProtocol,
        *,
        pass_threshold: float | None = None,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        cv_seed: int = 0,
        bootstrap_seed: int = 0,
        bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    ) -> None:
        """Initialize the calibrator.

        Args:
            judge: The judge to calibrate.
            pass_threshold: Pass cutoff applied to judge scores; ``None``
                derives it from the judge's verdict policy, with
                ``DEFAULT_PASS_THRESHOLD`` as fallback.
            min_samples: Sample-size floor for trustworthy statistics.
            cv_seed: Seed of the cross-validation fold assignment.
            bootstrap_seed: Seed of the bootstrap draws.
            bootstrap_resamples: Number of bootstrap draws (0 disables the interval).
        """
        self.judge = judge
        self.pass_threshold = pass_threshold
        self.min_samples = min_samples
        self.cv_seed = cv_seed
        self.bootstrap_seed = bootstrap_seed
        self.bootstrap_resamples = bootstrap_resamples

    @property
    def uses_protocol(self) -> bool:
        """True when the judge implements ``JudgeProtocol`` and runs through ``judge_many``."""
        return isinstance(self.judge, JudgeProtocol)

    def threshold_for(self, criterion: str) -> float:
        """The pass threshold applied to *criterion*."""
        if self.pass_threshold is not None:
            return self.pass_threshold
        if self.uses_protocol:
            return float(self.judge.config(criterion).verdict_policy.pass_threshold_on_score())
        return DEFAULT_PASS_THRESHOLD

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

        group_ids = [s.group_id for s in sample_list]
        results: dict[str, CriterionCalibration] = {}
        configs: dict[str, JudgeConfig] = {}
        for criterion in criteria:
            threshold = self.threshold_for(criterion)
            backend_failures = 0
            scores: list[float | None] = []
            if self.uses_protocol:
                attempts = await judge_many(self.judge, criterion, sample_list)
                configs[criterion] = attempts[0].config if attempts else self.judge.config(criterion)
                for index, attempt in enumerate(attempts):
                    if attempt.result is not None:
                        scores.append(attempt.result.score)
                    else:
                        assert attempt.error is not None
                        if attempt.error.kind != "parse":
                            backend_failures += 1
                        logger.warning(
                            f"{criterion}: sample {index} failed ({attempt.error.kind}: {attempt.error.message})"
                        )
                        scores.append(None)
            else:
                for index, sample in enumerate(sample_list):
                    try:
                        scores.append((await self._judge_call(criterion, sample)).score)
                    except EvaluationError as exc:
                        logger.warning(f"{criterion}: judge response for sample {index} could not be parsed ({exc})")
                        scores.append(None)
            human = [s.label_for(criterion) for s in sample_list]
            results[criterion] = analyze_criterion(
                criterion,
                scores,
                human,
                threshold=threshold,
                group_ids=group_ids,
                cv_seed=self.cv_seed,
                bootstrap_seed=self.bootstrap_seed,
                bootstrap_resamples=self.bootstrap_resamples,
                backend_failures=backend_failures,
            )

        return CalibrationReport(
            criteria=results,
            n_samples=len(sample_list),
            insufficient_data=len(sample_list) < self.min_samples
            or any(c.n < self.min_samples for c in results.values()),
            configs=configs,
        )

    # ── Judge dispatch (judges that do not implement JudgeProtocol) ──────

    def _judge_call(self, criterion: str, s: CalibrationSample) -> Awaitable[JudgeResult]:
        """One judge evaluation of *s* for *criterion* through the four named methods."""
        judge: Any = self.judge
        calls: dict[str, Callable[[], Awaitable[JudgeResult]]] = {
            "faithfulness": lambda: judge.evaluate_faithfulness(
                context=s.context, question=s.question, answer=s.answer
            ),
            "relevance": lambda: judge.evaluate_relevance(question=s.question, answer=s.answer),
            "hallucination": lambda: judge.detect_hallucination(context=s.context, answer=s.answer),
            "completeness": lambda: judge.evaluate_completeness(
                question=s.question, answer=s.answer, context=s.context
            ),
        }
        return calls[criterion]()

    # ── Statistics (kept as a method for callers that used it) ───────────

    def _analyze(
        self,
        criterion: str,
        all_scores: list[float | None],
        all_human: list[bool],
        threshold: float | None = None,
    ) -> CriterionCalibration:
        """Agreement statistics for one criterion; see :func:`analyze_criterion`."""
        return analyze_criterion(
            criterion,
            all_scores,
            all_human,
            threshold=threshold if threshold is not None else self.threshold_for(criterion),
            cv_seed=self.cv_seed,
            bootstrap_seed=self.bootstrap_seed,
            bootstrap_resamples=self.bootstrap_resamples,
        )

    def _best_threshold(self, scores: list[float], human: list[bool]) -> tuple[float, float]:
        """Threshold sweep at the calibrator's threshold; see :func:`_best_threshold`."""
        return _best_threshold(
            scores, human, self.pass_threshold if self.pass_threshold is not None else DEFAULT_PASS_THRESHOLD
        )
