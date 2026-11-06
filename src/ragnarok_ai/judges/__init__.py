"""Common judge contract: native outputs, verdict policies, configuration, traces, attempts.

See :mod:`ragnarok_ai.judges.contract`. The existing judge types
(``JudgeResult``, ``JudgeResults``, ``LLMJudge``) stay in
:mod:`ragnarok_ai.evaluators.judge`.
"""

from __future__ import annotations

from ragnarok_ai.judges.contract import (
    FAILURE_KINDS,
    JUDGE_KINDS,
    SCORE_RULES,
    UNITS,
    VERDICT_RULES,
    VERDICTS,
    AggregationPolicy,
    BinaryProbability,
    CallTrace,
    ClassProbabilities,
    FailureKind,
    JudgeAttempt,
    JudgeConfig,
    JudgeFailure,
    JudgeKind,
    JudgeOutput,
    LabelMap,
    OrdinalBands,
    OrdinalScore,
    ScoreRule,
    ScoreThreshold,
    Unit,
    UnitJudgment,
    Verdict,
    VerdictPolicy,
    VerdictRule,
    apply_verdict_policy,
    canonical_json,
)

__all__ = [
    "FAILURE_KINDS",
    "JUDGE_KINDS",
    "SCORE_RULES",
    "UNITS",
    "VERDICTS",
    "VERDICT_RULES",
    "AggregationPolicy",
    "BinaryProbability",
    "CallTrace",
    "ClassProbabilities",
    "FailureKind",
    "JudgeAttempt",
    "JudgeConfig",
    "JudgeFailure",
    "JudgeKind",
    "JudgeOutput",
    "LabelMap",
    "OrdinalBands",
    "OrdinalScore",
    "ScoreRule",
    "ScoreThreshold",
    "Unit",
    "UnitJudgment",
    "Verdict",
    "VerdictPolicy",
    "VerdictRule",
    "apply_verdict_policy",
    "canonical_json",
]
