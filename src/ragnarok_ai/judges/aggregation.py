"""Aggregate unit judgments (sentences, claims) into an answer-level score and verdict.

The rules are declared by an ``AggregationPolicy`` and recorded in the
``JudgeConfig``, so two results are only compared when they were aggregated
the same way. Skipped units never count. When nothing could be judged, the
sample is a ``no_units`` failure, never a score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragnarok_ai.core.exceptions import NoUnitsError
from ragnarok_ai.judges.contract import (
    AggregationPolicy,
    LabelMap,
    OrdinalBands,
    ScoreThreshold,
    UnitJudgment,
    Verdict,
    VerdictPolicy,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def verdict_from_score(policy: VerdictPolicy, score: float) -> Verdict:
    """The verdict a policy gives to an answer-level *score* in ``[0, 1]``.

    ``ScoreThreshold`` applies its own thresholds. ``OrdinalBands`` projects
    its bands on the score: PASS at or above the pass band's lower edge,
    FAIL at or below the fail band's upper edge, PARTIAL between.
    ``LabelMap`` has no score rule of its own: PASS at or above 0.5, FAIL
    below (see ``LabelMap.pass_threshold_on_score``).
    """
    if isinstance(policy, ScoreThreshold):
        if score >= policy.pass_threshold:
            return "PASS"
        if policy.fail_threshold is None or score <= policy.fail_threshold:
            return "FAIL"
        return "PARTIAL"
    if isinstance(policy, OrdinalBands):
        span = policy.scale_max - policy.scale_min
        if score >= policy.pass_threshold_on_score():
            return "PASS"
        if score <= (policy.fail_at - policy.scale_min) / span:
            return "FAIL"
        return "PARTIAL"
    if isinstance(policy, LabelMap):
        return "PASS" if score >= policy.pass_threshold_on_score() else "FAIL"
    msg = f"Unsupported verdict policy {type(policy).__name__}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def worst_verdict(verdicts: Sequence[Verdict]) -> Verdict:
    """FAIL if any unit is FAIL, else PARTIAL if any is PARTIAL, else PASS."""
    if "FAIL" in verdicts:
        return "FAIL"
    if "PARTIAL" in verdicts:
        return "PARTIAL"
    return "PASS"


def aggregate_units(
    units: Sequence[UnitJudgment],
    aggregation: AggregationPolicy,
    verdict_policy: VerdictPolicy,
) -> tuple[float, Verdict]:
    """Answer-level ``(score, verdict)`` from unit judgments.

    Skipped units are excluded. ``score_rule``: ``fraction_pass`` is the share
    of judged units whose verdict is PASS, ``mean`` and ``min`` apply to the
    unit scores. ``verdict_rule``: ``worst`` takes the worst unit verdict,
    ``from_score`` applies *verdict_policy* to the aggregated score.

    Raises:
        NoUnitsError: If no unit was judged (all skipped, or none at all).
        ValueError: If a judged unit lacks a score or a verdict.
    """
    judged = [u for u in units if u.skipped is None]
    if not judged:
        reasons = sorted({u.skipped for u in units if u.skipped is not None})
        detail = f"; skipped: {', '.join(reasons)}" if reasons else ""
        msg = f"No unit could be judged ({len(units)} unit(s){detail})"
        raise NoUnitsError(msg)
    for u in judged:
        if u.score is None or u.verdict is None:
            msg = f"Judged unit {u.span} has no score or no verdict"
            raise ValueError(msg)
    scores = [u.score for u in judged if u.score is not None]
    verdicts: list[Verdict] = [u.verdict for u in judged if u.verdict is not None]

    if aggregation.score_rule == "fraction_pass":
        score = sum(1 for v in verdicts if v == "PASS") / len(verdicts)
    elif aggregation.score_rule == "mean":
        score = sum(scores) / len(scores)
    else:
        score = min(scores)

    verdict = (
        worst_verdict(verdicts) if aggregation.verdict_rule == "worst" else verdict_from_score(verdict_policy, score)
    )
    return score, verdict


__all__ = ["aggregate_units", "verdict_from_score", "worst_verdict"]
