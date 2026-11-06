"""Tests for the aggregation of unit judgments (§1.5 of the judge contract)."""

from __future__ import annotations

import re

import pytest

from ragnarok_ai.core.exceptions import EvaluationError, NoUnitsError
from ragnarok_ai.judges import (
    AggregationPolicy,
    BinaryProbability,
    LabelMap,
    OrdinalBands,
    OrdinalScore,
    ScoreThreshold,
    UnitJudgment,
    aggregate_units,
    verdict_from_score,
    worst_verdict,
)

BANDS = OrdinalBands(4, 2)


def unit(score: float, verdict: str, start: int = 0) -> UnitJudgment:
    return UnitJudgment(text="x", span=(start, start + 1), score=score, verdict=verdict)  # type: ignore[arg-type]


def skipped(reason: str = "no alphanumeric content") -> UnitJudgment:
    return UnitJudgment(text="...", span=(0, 3), skipped=reason)


UNITS = [unit(1.0, "PASS"), unit(0.75, "PASS"), unit(0.25, "FAIL"), unit(0.5, "PARTIAL")]


class TestScoreRules:
    def test_fraction_pass(self):
        score, _ = aggregate_units(UNITS, AggregationPolicy("fraction_pass", "worst"), BANDS)
        assert score == 0.5

    def test_mean(self):
        score, _ = aggregate_units(UNITS, AggregationPolicy("mean", "worst"), BANDS)
        assert score == pytest.approx(0.625)

    def test_min(self):
        score, _ = aggregate_units(UNITS, AggregationPolicy("min", "worst"), BANDS)
        assert score == 0.25

    def test_skipped_units_do_not_count(self):
        units = [*UNITS, skipped(), skipped("citation marker alone")]
        for rule, expected in (("fraction_pass", 0.5), ("mean", 0.625), ("min", 0.25)):
            score, _ = aggregate_units(units, AggregationPolicy(rule, "worst"), BANDS)  # type: ignore[arg-type]
            assert score == pytest.approx(expected), rule


class TestVerdictRules:
    def test_worst(self):
        assert worst_verdict(["PASS", "PASS"]) == "PASS"
        assert worst_verdict(["PASS", "PARTIAL"]) == "PARTIAL"
        assert worst_verdict(["PARTIAL", "FAIL", "PASS"]) == "FAIL"
        _, verdict = aggregate_units(
            [unit(1.0, "PASS"), unit(0.5, "PARTIAL")], AggregationPolicy("mean", "worst"), BANDS
        )
        assert verdict == "PARTIAL"

    @pytest.mark.parametrize(
        ("policy", "score", "expected"),
        [
            (BANDS, 0.75, "PASS"),
            (BANDS, 0.5, "PARTIAL"),
            (BANDS, 0.25, "FAIL"),
            (BANDS, 0.3, "PARTIAL"),
            (ScoreThreshold(0.5), 0.5, "PASS"),
            (ScoreThreshold(0.5), 0.49, "FAIL"),
            (ScoreThreshold(0.7, 0.3), 0.5, "PARTIAL"),
            (LabelMap((("yes", "PASS"), ("no", "FAIL"))), 0.5, "PASS"),
            (LabelMap((("yes", "PASS"), ("no", "FAIL"))), 0.4, "FAIL"),
        ],
    )
    def test_from_score(self, policy, score, expected):
        assert verdict_from_score(policy, score) == expected

    def test_from_score_applies_to_the_aggregated_score(self):
        units = [unit(1.0, "PASS"), unit(1.0, "PASS"), unit(0.0, "FAIL"), unit(1.0, "PASS")]
        score, verdict = aggregate_units(units, AggregationPolicy("fraction_pass", "from_score"), BANDS)
        assert (score, verdict) == (0.75, "PASS")  # worst would say FAIL
        _, worst = aggregate_units(units, AggregationPolicy("fraction_pass", "worst"), BANDS)
        assert worst == "FAIL"


class TestFailures:
    def test_no_judged_unit_is_a_no_units_failure(self):
        with pytest.raises(
            NoUnitsError,
            match=re.escape(
                "No unit could be judged (2 unit(s); skipped: citation marker alone, no alphanumeric content)"
            ),
        ):
            aggregate_units([skipped(), skipped("citation marker alone")], AggregationPolicy("mean", "worst"), BANDS)
        with pytest.raises(NoUnitsError):
            aggregate_units([], AggregationPolicy("mean", "worst"), BANDS)

    def test_no_units_error_is_an_evaluation_error(self):
        assert issubclass(NoUnitsError, EvaluationError)

    def test_judged_unit_without_verdict_is_rejected(self):
        unit_without_verdict = UnitJudgment(text="x", span=(0, 1), output=BinaryProbability(0.9), score=0.9)
        with pytest.raises(ValueError, match="no score or no verdict"):
            aggregate_units([unit_without_verdict], AggregationPolicy("mean", "worst"), BANDS)

    def test_ordinal_units_round_trip(self):
        units = [UnitJudgment("a", (0, 1), output=OrdinalScore(5, 1, 5), score=1.0, verdict="PASS")]
        assert aggregate_units(units, AggregationPolicy("min", "from_score"), BANDS) == (1.0, "PASS")
