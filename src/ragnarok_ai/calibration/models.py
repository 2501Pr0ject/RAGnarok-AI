"""Data models for judge calibration.

LLM-as-judge scores are only useful if you can trust them. Calibration
measures that trust: label a small set of examples yourself, run the
judge on them, and get agreement statistics (Cohen's kappa, error rates,
a recommended pass threshold) per criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CalibrationSample(BaseModel):
    """One human-labeled example.

    Attributes:
        question: The user question.
        context: The retrieved context the answer was generated from.
        answer: The generated answer being judged.
        human_pass: Your verdict — is the answer acceptable? Used for
            every criterion unless overridden in ``human_labels``.
        human_labels: Optional per-criterion overrides, e.g.
            ``{"faithfulness": True, "relevance": False}`` for an answer
            that is grounded but off-topic.
        note: Optional free-text rationale for the label.
        sample_id: Optional stable identifier, reported in ``JudgeAttempt``;
            the sample's position is used when absent.
        group_id: Optional group (the source question, for instance). The
            bootstrap resamples by group so duplicated questions do not count
            as independent samples.
    """

    model_config = {"frozen": True}

    question: str
    context: str
    answer: str
    human_pass: bool
    human_labels: dict[str, bool] = Field(default_factory=dict)
    note: str | None = None
    sample_id: str | None = None
    group_id: str | None = None

    def label_for(self, criterion: str) -> bool:
        """The human verdict for *criterion* (override or global)."""
        return self.human_labels.get(criterion, self.human_pass)


class CalibrationSet(BaseModel):
    """A named, versionable collection of labeled samples.

    Attributes:
        name: Identifier for this labeled set.
        samples: The labeled examples.
        created_at: When the set was created.
        metadata: Optional metadata (labeler, guidelines version...).
    """

    name: str = "calibration"
    samples: list[CalibrationSample] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def save(self, path: str | Path) -> None:
        """Write the set to a JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CalibrationSet:
        """Read a set from a JSON file."""
        return cls.model_validate_json(Path(path).read_text())


# Landis & Koch (1977) interpretation bands for Cohen's kappa
_KAPPA_BANDS: list[tuple[float, str]] = [
    (0.8, "almost perfect"),
    (0.6, "substantial"),
    (0.4, "moderate"),
    (0.2, "fair"),
    (0.0, "slight"),
]


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch interpretation of a kappa value."""
    if kappa < 0:
        return "poor"
    for floor, label in _KAPPA_BANDS:
        if kappa >= floor:
            return label
    return "poor"  # pragma: no cover


@dataclass(frozen=True)
class CriterionCalibration:
    """Judge-vs-human agreement for one criterion.

    Attributes:
        criterion: The judged criterion (e.g. "faithfulness").
        n: Number of samples with a usable judge score. Every statistic
            below is computed on these samples only.
        accuracy: Fraction of samples where judge and human agree.
        accuracy_ci: 95% Wilson confidence interval on accuracy.
        kappa: Cohen's kappa (chance-corrected agreement).
        kappa_interpretation: Landis & Koch band for the kappa value.
        false_accept_rate: Fraction of human-rejected answers the judge
            passed — the dangerous direction (bad answers slip through).
        false_reject_rate: Fraction of human-accepted answers the judge
            failed — the noisy direction (good answers flagged).
        threshold: The pass threshold these statistics were computed at.
        recommended_threshold: Threshold that maximizes kappa on this set.
        kappa_at_recommended: Kappa at the recommended threshold, measured
            on the same samples that selected it (in-sample, optimistic).
        disagreements: Indices of samples where judge and human disagree
            (at ``threshold``), for review.
        parse_failures: Samples whose judge response could not be parsed;
            excluded from the statistics.
        failed_indices: Indices of every failed sample (parse or backend), for review.
        backend_failures: Samples the backend failed on (timeout, server error);
            excluded like parse failures.
        human_pass_rate: Share of human PASS labels among the scored samples.
        majority_baseline_accuracy: Agreement a constant judge answering the
            majority class would reach; compare ``accuracy`` with it.
        minority_count: Size of the smaller human class.
        kappa_interpretable: False when the minority class has fewer than
            10 samples: kappa is then dominated by a handful of labels.
        kappa_at_recommended_cv: Out-of-sample kappa of the recommended
            threshold: the threshold is chosen on the other folds, the
            verdicts are scored on the held-out fold (``cv_method``).
        cv_method: ``stratified-5-fold``, ``leave-one-out``, or leave-one-out
            with the reason when a class is smaller than the number of folds.
        cv_folds: Number of folds actually used.
        kappa_cv_ci: 95% bootstrap interval of ``kappa_at_recommended_cv``,
            ``None`` when no draw could be used.
        bootstrap_resamples: Number of bootstrap draws requested.
        bootstrap_excluded: Draws with a single human class, where kappa is
            undefined; excluded from the interval.
        bootstrap_reliable: False when more than 10% of the draws were excluded.
        cv_seed: Seed of the fold assignment.
        bootstrap_seed: Seed of the bootstrap draws.
        judge_scores: The judge scores per submitted sample (``None`` for a
            failure), kept so that comparable reports can be merged.
        human_labels: The human labels per submitted sample.
        group_ids: The group of each submitted sample, ``None`` when absent.
    """

    criterion: str
    n: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    kappa: float
    kappa_interpretation: str
    false_accept_rate: float
    false_reject_rate: float
    threshold: float
    recommended_threshold: float
    kappa_at_recommended: float
    disagreements: list[int]
    parse_failures: int = 0
    failed_indices: list[int] = field(default_factory=list)
    backend_failures: int = 0
    human_pass_rate: float = 0.0
    majority_baseline_accuracy: float = 0.0
    minority_count: int = 0
    kappa_interpretable: bool = True
    kappa_at_recommended_cv: float | None = None
    cv_method: str | None = None
    cv_folds: int = 0
    kappa_cv_ci: tuple[float, float] | None = None
    bootstrap_resamples: int = 0
    bootstrap_excluded: int = 0
    bootstrap_reliable: bool | None = None
    cv_seed: int | None = None
    bootstrap_seed: int | None = None
    judge_scores: tuple[float | None, ...] = ()
    human_labels: tuple[bool, ...] = ()
    group_ids: tuple[str | None, ...] = ()

    @property
    def n_labeled(self) -> int:
        """Labeled samples submitted to the judge, scored or not."""
        return self.n + self.parse_failures + self.backend_failures


@dataclass
class CalibrationReport:
    """Result of calibrating a judge against human labels.

    Attributes:
        criteria: Per-criterion agreement statistics.
        n_samples: Number of labeled samples.
        insufficient_data: True when fewer than ``min_samples`` labels
            were provided — statistics are reported but not trustworthy.
        timestamp: When calibration ran.
    """

    criteria: dict[str, CriterionCalibration]
    n_samples: int
    insufficient_data: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    configs: dict[str, Any] = field(default_factory=dict)

    def comparable_with(self, other: CalibrationReport) -> bool:
        """True when both reports carry the same judge configuration for every shared criterion.

        Reports without configurations (judges outside ``JudgeProtocol``)
        are never comparable: nothing proves they come from the same judge.
        """
        shared = set(self.criteria) & set(other.criteria)
        if not shared or not self.configs or not other.configs:
            return False
        return all(
            c in self.configs and c in other.configs and self.configs[c].comparable_with(other.configs[c])
            for c in shared
        )

    def merge(self, other: CalibrationReport, **analysis: Any) -> CalibrationReport:
        """Combine two runs of the same judge into one report.

        Statistics are recomputed on the concatenated samples with
        :func:`ragnarok_ai.calibration.calibrator.analyze_criterion`
        (extra keyword arguments go to it: seeds, bootstrap size).

        Raises:
            ValueError: If the two runs are not comparable (different judge
                configuration, missing configuration, or no shared criterion).
        """
        if not self.comparable_with(other):
            msg = "Cannot merge calibration reports: the judge configurations are not comparable"
            raise ValueError(msg)
        from ragnarok_ai.calibration.calibrator import analyze_criterion

        merged: dict[str, CriterionCalibration] = {}
        for criterion in self.criteria:
            if criterion not in other.criteria:
                continue
            a, b = self.criteria[criterion], other.criteria[criterion]
            if a.threshold != b.threshold:
                msg = f"Cannot merge {criterion}: thresholds differ ({a.threshold} and {b.threshold})"
                raise ValueError(msg)
            merged[criterion] = analyze_criterion(
                criterion,
                [*a.judge_scores, *b.judge_scores],
                [*a.human_labels, *b.human_labels],
                threshold=a.threshold,
                group_ids=[*a.group_ids, *b.group_ids],
                backend_failures=a.backend_failures + b.backend_failures,
                **analysis,
            )
        return CalibrationReport(
            criteria=merged,
            n_samples=self.n_samples + other.n_samples,
            insufficient_data=self.insufficient_data and other.insufficient_data,
            configs={c: self.configs[c] for c in merged},
        )

    def summary(self) -> str:
        """Human-readable summary for console output."""
        lines = [f"Judge calibration on {self.n_samples} labeled samples"]
        if self.insufficient_data:
            lines.append("WARNING: small sample — treat these numbers as indicative only")
        for c in self.criteria.values():
            lines.append("")
            lines.append(f"{c.criterion}:")
            lines.append(
                f"  agreement {c.accuracy:.0%} "
                f"(95% CI {c.accuracy_ci[0]:.0%}-{c.accuracy_ci[1]:.0%}), "
                f"kappa {c.kappa:.2f} ({c.kappa_interpretation})"
            )
            lines.append(f"  false accepts {c.false_accept_rate:.0%}, false rejects {c.false_reject_rate:.0%}")
            if c.parse_failures:
                lines.append(
                    f"  parse failures {c.parse_failures}/{c.n_labeled} "
                    f"(excluded from the statistics above; statistics on {c.n} scored samples)"
                )
            if c.backend_failures:
                lines.append(f"  backend failures {c.backend_failures}/{c.n_labeled} (excluded as well)")
            if c.n:
                lines.append(
                    f"  human PASS prevalence {c.human_pass_rate:.1%}; a constant majority-class judge "
                    f"would agree {c.majority_baseline_accuracy:.1%}"
                )
                if not c.kappa_interpretable:
                    lines.append(
                        f"  WARNING: kappa not interpretable, the minority class has only {c.minority_count} samples"
                    )
            if abs(c.recommended_threshold - c.threshold) > 1e-9:
                lines.append(
                    f"  recommended threshold {c.recommended_threshold:.2f} "
                    f"(kappa {c.kappa_at_recommended:.2f}, current {c.threshold:.2f})"
                )
                lines.append(
                    f"  WARNING: that kappa is measured on the same {c.n} samples that selected the threshold "
                    "(in-sample, optimistic); confirm it on labels the sweep has not seen"
                )
            if c.kappa_at_recommended_cv is not None:
                ci = (
                    f"95% CI {c.kappa_cv_ci[0]:.2f} to {c.kappa_cv_ci[1]:.2f}"
                    if c.kappa_cv_ci is not None
                    else "no usable bootstrap draw"
                )
                reliability = "" if c.bootstrap_reliable is not False else "; interval NOT reliable"
                lines.append(
                    f"  out-of-sample kappa of the recommended threshold {c.kappa_at_recommended_cv:.2f} "
                    f"({c.cv_method}, {ci}; {c.bootstrap_excluded}/{c.bootstrap_resamples} single-class draws "
                    f"excluded{reliability}; seeds cv={c.cv_seed} bootstrap={c.bootstrap_seed})"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "n_samples": self.n_samples,
            "insufficient_data": self.insufficient_data,
            "timestamp": self.timestamp.isoformat(),
            "criteria": {
                name: {
                    "n": c.n,
                    "accuracy": c.accuracy,
                    "accuracy_ci": list(c.accuracy_ci),
                    "kappa": c.kappa,
                    "kappa_interpretation": c.kappa_interpretation,
                    "false_accept_rate": c.false_accept_rate,
                    "false_reject_rate": c.false_reject_rate,
                    "threshold": c.threshold,
                    "recommended_threshold": c.recommended_threshold,
                    "kappa_at_recommended": c.kappa_at_recommended,
                    "disagreements": list(c.disagreements),
                    "parse_failures": c.parse_failures,
                    "failed_indices": list(c.failed_indices),
                    "n_labeled": c.n_labeled,
                    "backend_failures": c.backend_failures,
                    "human_pass_rate": c.human_pass_rate,
                    "majority_baseline_accuracy": c.majority_baseline_accuracy,
                    "minority_count": c.minority_count,
                    "kappa_interpretable": c.kappa_interpretable,
                    "kappa_at_recommended_cv": c.kappa_at_recommended_cv,
                    "cv_method": c.cv_method,
                    "cv_folds": c.cv_folds,
                    "kappa_cv_ci": list(c.kappa_cv_ci) if c.kappa_cv_ci is not None else None,
                    "bootstrap_resamples": c.bootstrap_resamples,
                    "bootstrap_excluded": c.bootstrap_excluded,
                    "bootstrap_reliable": c.bootstrap_reliable,
                    "cv_seed": c.cv_seed,
                    "bootstrap_seed": c.bootstrap_seed,
                }
                for name, c in self.criteria.items()
            },
            "configs": {name: cfg.to_dict() for name, cfg in self.configs.items()},
        }
