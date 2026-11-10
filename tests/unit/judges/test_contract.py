"""Tests for the common judge contract (pure data, nothing wired in)."""

from __future__ import annotations

import json
import math

import pytest

from ragnarok_ai.evaluators.judge import JudgeResult
from ragnarok_ai.judges import (
    AggregationPolicy,
    BinaryProbability,
    CallTrace,
    ClassProbabilities,
    JudgeAttempt,
    JudgeConfig,
    JudgeFailure,
    LabelMap,
    OrdinalBands,
    OrdinalScore,
    ScoreThreshold,
    UnitJudgment,
    apply_verdict_policy,
    canonical_json,
)


def make_config(**overrides: object) -> JudgeConfig:
    """A valid Prometheus-like configuration."""
    defaults: dict = {
        "judge_class": "ragnarok_ai.evaluators.judge.LLMJudge",
        "judge_kind": "llm",
        "model": "prometheus2:latest",
        "criterion": "faithfulness",
        "verdict_policy": OrdinalBands(pass_at=4, fail_at=2),
        "ragnarok_ai_version": "1.13.0",
        "backend": "ollama",
        "options": (("num_ctx", 8192), ("seed", 42), ("temperature", 0.0)),
        "prompt_format": "prometheus",
    }
    defaults.update(overrides)
    return JudgeConfig(**defaults)


# =============================================================================
# Native outputs
# =============================================================================


class TestOrdinalScore:
    @pytest.mark.parametrize(("value", "expected"), [(1, 0.0), (2, 0.25), (3, 0.5), (4, 0.75), (5, 1.0)])
    def test_projection_on_1_5(self, value, expected):
        assert OrdinalScore(value, 1, 5).score == expected

    def test_projection_on_other_scale(self):
        assert OrdinalScore(0, 0, 10).score == 0.0
        assert OrdinalScore(10, 0, 10).score == 1.0

    @pytest.mark.parametrize(("value", "lo", "hi"), [(6, 1, 5), (0, 1, 5), (3, 5, 1), (3, 3, 3)])
    def test_invalid(self, value, lo, hi):
        with pytest.raises(ValueError):
            OrdinalScore(value, lo, hi)

    def test_is_frozen_and_discriminated(self):
        out = OrdinalScore(4, 1, 5)
        assert out.kind == "ordinal"
        with pytest.raises(AttributeError):
            out.value = 5  # type: ignore[misc]


class TestBinaryProbability:
    def test_projection(self):
        assert BinaryProbability(0.37).score == 0.37
        assert BinaryProbability(0.0).kind == "binary"

    @pytest.mark.parametrize("p", [-0.01, 1.01])
    def test_invalid(self, p):
        with pytest.raises(ValueError):
            BinaryProbability(p)


class TestClassProbabilities:
    def test_score_is_positive_label_probability(self):
        out = ClassProbabilities(("entailment", "neutral", "contradiction"), (0.2, 0.5, 0.3), "entailment")
        assert out.score == 0.2
        assert out.predicted == "neutral"
        assert out.kind == "classes"

    def test_tolerance_on_sum(self):
        ClassProbabilities(("a", "b"), (0.5000004, 0.4999999), "a")

    def test_tie_goes_to_first_label(self):
        assert ClassProbabilities(("a", "b"), (0.5, 0.5), "b").predicted == "a"

    @pytest.mark.parametrize(
        ("labels", "probs", "positive"),
        [
            (("a", "b"), (0.5,), "a"),  # length mismatch
            (("a", "a"), (0.5, 0.5), "a"),  # duplicate labels
            (("a", "b"), (0.5, 0.5), "c"),  # unknown positive label
            (("a", "b"), (0.7, 0.4), "a"),  # does not sum to 1
            (("a", "b"), (1.2, -0.2), "a"),  # out of range
            ((), (), "a"),  # empty
        ],
    )
    def test_invalid(self, labels, probs, positive):
        with pytest.raises(ValueError):
            ClassProbabilities(labels, probs, positive)


# =============================================================================
# Verdict policies
# =============================================================================


class TestOrdinalBands:
    @pytest.mark.parametrize(
        ("value", "expected"), [(5, "PASS"), (4, "PASS"), (3, "PARTIAL"), (2, "FAIL"), (1, "FAIL")]
    )
    def test_prometheus_bands_match_verdict_from_raw_score(self, value, expected):
        """The 1.12.3 rule: 4-5 PASS, 3 PARTIAL, 1-2 FAIL."""
        from ragnarok_ai.evaluators.judge import verdict_from_raw_score

        policy = OrdinalBands(pass_at=4, fail_at=2)
        assert policy.verdict(OrdinalScore(value, 1, 5)) == expected == verdict_from_raw_score(value)

    def test_pass_threshold_on_score(self):
        assert OrdinalBands(4, 2).pass_threshold_on_score() == 0.75
        assert OrdinalBands(7, 3, scale_min=0, scale_max=10).pass_threshold_on_score() == 0.7

    def test_rejects_other_outputs_and_scales(self):
        policy = OrdinalBands(4, 2)
        with pytest.raises(TypeError):
            policy.verdict(BinaryProbability(0.9))
        with pytest.raises(ValueError, match="defined on"):
            policy.verdict(OrdinalScore(7, 0, 10))

    def test_other_scale_is_an_explicit_error_not_a_silent_verdict(self):
        """Bands defined on 1-5 applied to a 0-10 score: error naming both scales, whatever the value."""
        policy = OrdinalBands(pass_at=4, fail_at=2)
        for value in (0, 4, 10):
            with pytest.raises(ValueError, match=r"defined on \[1, 5\] but the output is on \[0, 10\]"):
                policy.verdict(OrdinalScore(value, 0, 10))
        with pytest.raises(ValueError, match=r"defined on \[1, 5\] but the output is on \[1, 4\]"):
            policy.verdict(OrdinalScore(4, 1, 4))

    @pytest.mark.parametrize(("pass_at", "fail_at"), [(2, 4), (4, 4), (6, 2), (4, 0)])
    def test_invalid_bands(self, pass_at, fail_at):
        with pytest.raises(ValueError):
            OrdinalBands(pass_at, fail_at)


class TestScoreThreshold:
    def test_binary_without_fail_threshold(self):
        policy = ScoreThreshold(0.5)
        assert policy.verdict(BinaryProbability(0.5)) == "PASS"
        assert policy.verdict(BinaryProbability(0.49)) == "FAIL"
        assert policy.pass_threshold_on_score() == 0.5

    def test_three_bands_with_fail_threshold(self):
        policy = ScoreThreshold(0.7, fail_threshold=0.3)
        assert policy.verdict(BinaryProbability(0.7)) == "PASS"
        assert policy.verdict(BinaryProbability(0.5)) == "PARTIAL"
        assert policy.verdict(BinaryProbability(0.3)) == "FAIL"

    def test_applies_to_any_output_through_its_score(self):
        assert ScoreThreshold(0.75).verdict(OrdinalScore(4, 1, 5)) == "PASS"
        assert ScoreThreshold(0.75).verdict(ClassProbabilities(("s", "u"), (0.8, 0.2), "s")) == "PASS"

    @pytest.mark.parametrize(("pass_t", "fail_t"), [(1.5, None), (0.5, 0.5), (0.5, 0.6), (0.5, -0.1)])
    def test_invalid(self, pass_t, fail_t):
        with pytest.raises(ValueError):
            ScoreThreshold(pass_t, fail_t)


class TestLabelMap:
    GROUNDING = LabelMap((("SUPPORTED", "PASS"), ("UNSUPPORTED", "FAIL"), ("CONTRADICTED", "FAIL")))

    @pytest.mark.parametrize(
        ("probs", "expected"),
        [((0.7, 0.2, 0.1), "PASS"), ((0.2, 0.7, 0.1), "FAIL"), ((0.1, 0.2, 0.7), "FAIL")],
    )
    def test_unsupported_is_a_failure_never_partial(self, probs, expected):
        out = ClassProbabilities(("SUPPORTED", "UNSUPPORTED", "CONTRADICTED"), probs, "SUPPORTED")
        assert self.GROUNDING.verdict(out) == expected
        assert apply_verdict_policy(self.GROUNDING, out) == expected

    def test_threshold_on_score_is_half_and_differs_from_argmax(self):
        """Documented limit: with three classes the argmax can win below 0.5."""
        out = ClassProbabilities(("SUPPORTED", "UNSUPPORTED", "CONTRADICTED"), (0.4, 0.3, 0.3), "SUPPORTED")
        assert self.GROUNDING.pass_threshold_on_score() == 0.5
        assert self.GROUNDING.verdict(out) == "PASS"  # argmax
        assert out.score < 0.5  # a 0.5 threshold on the score would say FAIL

    def test_unknown_predicted_label_raises(self):
        out = ClassProbabilities(("yes", "no"), (0.9, 0.1), "yes")
        with pytest.raises(ValueError, match="no verdict for predicted label 'yes'"):
            self.GROUNDING.verdict(out)

    def test_rejects_other_outputs(self):
        with pytest.raises(TypeError):
            self.GROUNDING.verdict(BinaryProbability(0.9))

    @pytest.mark.parametrize(
        "mapping",
        [(), (("a", "PASS"), ("a", "FAIL")), (("a", "MAYBE"),)],
    )
    def test_invalid(self, mapping):
        with pytest.raises(ValueError):
            LabelMap(mapping)  # type: ignore[arg-type]


# =============================================================================
# Units and aggregation policy
# =============================================================================


class TestUnitJudgment:
    def test_judged_unit(self):
        unit = UnitJudgment("The sky is blue.", (0, 16), output=BinaryProbability(0.9), score=0.9, verdict="PASS")
        assert unit.skipped is None

    def test_skipped_unit_carries_a_reason_and_nothing_else(self):
        unit = UnitJudgment("Sure!", (0, 5), skipped="fragment of fewer than 3 words")
        assert unit.score is None and unit.verdict is None
        with pytest.raises(ValueError, match="no score and no verdict"):
            UnitJudgment("Sure!", (0, 5), score=1.0, skipped="fragment")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"span": (5, 2), "score": 0.5},  # end before start
            {"span": (-1, 2), "score": 0.5},
            {"span": (0, 2)},  # neither score nor skipped
            {"span": (0, 2), "score": 1.5},
            {"span": (0, 2), "score": 0.5, "verdict": "MAYBE"},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            UnitJudgment("text", **kwargs)


class TestAggregationPolicy:
    def test_valid(self):
        policy = AggregationPolicy("fraction_pass", "worst")
        assert (policy.score_rule, policy.verdict_rule) == ("fraction_pass", "worst")

    @pytest.mark.parametrize(("score_rule", "verdict_rule"), [("median", "worst"), ("mean", "best")])
    def test_invalid(self, score_rule, verdict_rule):
        with pytest.raises(ValueError):
            AggregationPolicy(score_rule, verdict_rule)  # type: ignore[arg-type]


# =============================================================================
# JudgeConfig
# =============================================================================


class TestJudgeConfig:
    def test_hash_is_stable_and_order_independent_of_field_definition(self):
        a = make_config()
        b = make_config()
        assert a.config_hash == b.config_hash
        assert a.comparable_with(b)
        assert hash(a) == hash(b)
        assert len(a.config_hash) == 64

    def test_any_field_changes_the_hash(self):
        base = make_config()
        for change in (
            {"model": "mistral:7b"},
            {"options": (("num_ctx", 4096), ("seed", 42), ("temperature", 0.0))},
            {"prompt_format": "score-n"},
            {"criterion": "relevance"},
            {"verdict_policy": OrdinalBands(pass_at=3, fail_at=2)},
            {"input_transforms": ("medical_normalizer:1",)},
            {"ragnarok_ai_version": "1.12.3"},
            {"judge_specific": (("training_set", "v3"),)},
        ):
            assert not base.comparable_with(make_config(**change)), change

    def test_policies_with_same_fields_but_different_types_differ(self):
        a = make_config(verdict_policy=ScoreThreshold(0.5))
        b = make_config(verdict_policy=LabelMap((("yes", "PASS"), ("no", "FAIL"))))
        assert not a.comparable_with(b)
        assert a.to_dict()["verdict_policy"]["type"] == "ScoreThreshold"

    def test_numbers_are_normalized(self):
        """0 and 0.0, 8192 and 8192.0 give the same hash."""
        a = make_config(options=(("num_ctx", 8192), ("seed", 42), ("temperature", 0)))
        b = make_config(options=(("num_ctx", 8192.0), ("seed", 42), ("temperature", 0.0)))
        assert a.comparable_with(b)
        c = make_config(options=(("num_ctx", 8192), ("seed", 42), ("temperature", 0.1)))
        assert not a.comparable_with(c)

    def test_booleans_are_not_numbers(self):
        a = make_config(judge_specific=(("flag", True),))
        b = make_config(judge_specific=(("flag", 1),))
        assert not a.comparable_with(b)

    def test_values_must_be_json_serializable(self):
        with pytest.raises(ValueError, match="not JSON-serializable"):
            make_config(options=(("callback", object()),))

    @pytest.mark.parametrize(
        "value",
        [math.nan, math.inf, -math.inf, [1.0, math.nan], {"nested": {"deep": math.inf}}, (0.5, -math.inf)],
        ids=["nan", "inf", "-inf", "nan in list", "inf in nested dict", "-inf in tuple"],
    )
    @pytest.mark.parametrize("field_name", ["options", "judge_specific"])
    def test_nan_and_inf_are_refused_at_construction(self, field_name, value):
        """A hash must never depend on a value JSON cannot carry."""
        with pytest.raises(ValueError, match=f"JudgeConfig {field_name}\\['x'\\] is not JSON-serializable"):
            make_config(**{field_name: (("x", value),)})

    def test_pairs_must_be_sorted_and_unique(self):
        with pytest.raises(ValueError, match="sorted"):
            make_config(options=(("seed", 42), ("num_ctx", 8192)))
        with pytest.raises(ValueError, match="unique"):
            make_config(options=(("seed", 1), ("seed", 2)))

    def test_unit_and_aggregation_consistency(self):
        with pytest.raises(ValueError, match="aggregation must be None"):
            make_config(aggregation=AggregationPolicy("mean", "worst"))
        with pytest.raises(ValueError, match="requires an aggregation"):
            make_config(unit="sentence", segmenter="pysbd:0.3.4:rules1")
        with pytest.raises(ValueError, match="requires a segmenter"):
            make_config(unit="sentence", aggregation=AggregationPolicy("min", "worst"))
        ok = make_config(unit="sentence", aggregation=AggregationPolicy("min", "worst"), segmenter="pysbd:0.3.4:rules1")
        assert ok.unit == "sentence"

    def test_invalid_enums(self):
        with pytest.raises(ValueError):
            make_config(judge_kind="oracle")
        with pytest.raises(ValueError):
            make_config(unit="paragraph")

    def test_to_dict_is_json_and_canonical_json_is_sorted_and_compact(self):
        payload = make_config().to_dict()
        text = canonical_json(payload)
        assert json.loads(text) == json.loads(json.dumps(payload))
        assert " " not in text.replace("ragnarok_ai.evaluators.judge.LLMJudge", "")
        assert list(json.loads(text).keys()) == sorted(json.loads(text).keys())


# =============================================================================
# CallTrace, JudgeFailure, JudgeAttempt
# =============================================================================


class TestCallTrace:
    def test_defaults_are_unknown(self):
        trace = CallTrace()
        assert trace.prompt_tokens is None and trace.context_truncated is None and trace.raw_output is None

    @pytest.mark.parametrize("kwargs", [{"prompt_tokens": -1}, {"completion_tokens": -3}, {"latency_ms": -0.5}])
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            CallTrace(**kwargs)


class TestJudgeFailure:
    def test_valid(self):
        failure = JudgeFailure("parse", "no [RESULT] marker", raw_output="…")
        assert failure.kind == "parse"

    @pytest.mark.parametrize(("kind", "message"), [("oom", "x"), ("parse", "")])
    def test_invalid(self, kind, message):
        with pytest.raises(ValueError):
            JudgeFailure(kind, message)  # type: ignore[arg-type]


class TestJudgeAttempt:
    RESULT = JudgeResult(verdict="PASS", score=0.75, explanation="ok", criteria="faithfulness", raw_score=4)

    def test_result_xor_error(self):
        config = make_config()
        ok = JudgeAttempt(
            "p001", "faithfulness", config, result=self.RESULT, error=None, trace=CallTrace(latency_ms=12.0)
        )
        ko = JudgeAttempt("p002", "faithfulness", config, result=None, error=JudgeFailure("timeout", "300 s"))
        assert ok.ok is True and ko.ok is False
        with pytest.raises(ValueError, match="exactly one"):
            JudgeAttempt("p003", "faithfulness", config, result=self.RESULT, error=JudgeFailure("parse", "x"))
        with pytest.raises(ValueError, match="exactly one"):
            JudgeAttempt("p004", "faithfulness", config, result=None, error=None)

    def test_criterion_must_match_config(self):
        with pytest.raises(ValueError, match="differs from its config"):
            JudgeAttempt("p001", "relevance", make_config(), result=self.RESULT, error=None)
