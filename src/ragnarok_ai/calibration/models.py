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
    """

    model_config = {"frozen": True}

    question: str
    context: str
    answer: str
    human_pass: bool
    human_labels: dict[str, bool] = Field(default_factory=dict)
    note: str | None = None

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
        n: Number of labeled samples used.
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
        kappa_at_recommended: Kappa at the recommended threshold.
        disagreements: Indices of samples where judge and human disagree
            (at ``threshold``), for review.
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
            if abs(c.recommended_threshold - c.threshold) > 1e-9:
                lines.append(
                    f"  recommended threshold {c.recommended_threshold:.2f} "
                    f"(kappa {c.kappa_at_recommended:.2f}, current {c.threshold:.2f})"
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
                }
                for name, c in self.criteria.items()
            },
        }
