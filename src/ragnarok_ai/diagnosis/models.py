"""Data models for root-cause diagnosis of RAG evaluation failures.

An evaluation score tells you *that* quality is low; diagnosis tells you
*why*. Each failing query is classified into a failure cause (retrieval
miss, ranking problem, hallucination...), and the report aggregates causes
into an actionable picture: which stage of the pipeline to fix first, and
which kinds of questions fail most.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class FailureCause(str, Enum):
    """Root cause of a failing query.

    Values:
        PIPELINE_ERROR: The pipeline raised an error; fix this first.
        RETRIEVAL_MISS: No (or too few) ground-truth documents retrieved —
            the generator never saw the right information.
        RETRIEVAL_RANKING: Relevant documents retrieved but ranked low.
        CONTEXT_INSUFFICIENT: The right documents were retrieved but their
            chunks do not contain enough information to answer — typically
            a chunking problem. (LLM-assisted diagnosis only.)
        GENERATION_HALLUCINATION: The context was sufficient but the answer
            makes unsupported claims. (LLM-assisted diagnosis only.)
        GENERATION_INCOMPLETE: The answer is grounded but does not fully
            address the question (or is empty).
        UNDETERMINED: Failing, but the available signals cannot decide —
            provide an LLM (and documents) to refine.
    """

    PIPELINE_ERROR = "pipeline_error"
    RETRIEVAL_MISS = "retrieval_miss"
    RETRIEVAL_RANKING = "retrieval_ranking"
    CONTEXT_INSUFFICIENT = "context_insufficient"
    GENERATION_HALLUCINATION = "generation_hallucination"
    GENERATION_INCOMPLETE = "generation_incomplete"
    UNDETERMINED = "undetermined"


# Actionable advice per cause, used to order the report's recommendations
RECOMMENDATIONS: dict[FailureCause, str] = {
    FailureCause.PIPELINE_ERROR: ("Fix pipeline errors first — they mask every other quality signal."),
    FailureCause.RETRIEVAL_MISS: (
        "Relevant documents are not being retrieved: revisit chunk size/overlap, "
        "try different embeddings, or increase k."
    ),
    FailureCause.RETRIEVAL_RANKING: (
        "Relevant documents are retrieved but ranked low: add a reranker or tune hybrid search weights."
    ),
    FailureCause.CONTEXT_INSUFFICIENT: (
        "The right documents come back but their chunks lack the needed "
        "information: increase chunk size or overlap so answers are not split "
        "across boundaries."
    ),
    FailureCause.GENERATION_HALLUCINATION: (
        "The context is sufficient but answers stray from it: tighten the "
        'prompt ("answer only from the context"), lower temperature, or use '
        "a stronger generator model."
    ),
    FailureCause.GENERATION_INCOMPLETE: (
        "Answers are grounded but partial: pass more retrieved documents to the generator or raise its context window."
    ),
    FailureCause.UNDETERMINED: (
        "Some failures could not be classified: provide an LLM and the document corpus to diagnose the generation side."
    ),
}


@dataclass
class DiagnosisThresholds:
    """Thresholds separating passing from failing queries.

    Attributes:
        recall_pass: Queries with recall below this fail on retrieval.
        mrr_pass: Retrieval-failing queries with recall > 0 but MRR below
            this are classified as ranking problems (the information was
            found, just buried).
    """

    recall_pass: float = 0.5
    mrr_pass: float = 0.5


@dataclass(frozen=True)
class QueryDiagnosis:
    """Diagnosis of one failing query.

    Attributes:
        index: Position of the query in the test set.
        question: The query text.
        cause: The classified failure cause.
        tier: "heuristic" (metrics only) or "llm" (LLM-assisted).
        evidence: Supporting signals — metric values, missing document
            IDs, LLM verdicts — for drill-down and audit.
    """

    index: int
    question: str
    cause: FailureCause
    tier: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosisReport:
    """Aggregated root-cause analysis of an evaluation run.

    Attributes:
        total_queries: Number of queries in the evaluation.
        failing_queries: Number of queries diagnosed as failing.
        diagnoses: Per-query diagnoses (failing queries only).
        breakdown: Count of failing queries per cause.
        patterns: Failure rate per query-metadata value, keyed as
            ``{metadata_key: {value: failure_rate}}`` — surfaces e.g.
            "multi_hop questions fail three times more than simple ones".
        recommendations: Actionable advice, ordered by dominant cause.
        timestamp: When the diagnosis ran.
    """

    total_queries: int
    failing_queries: int
    diagnoses: list[QueryDiagnosis]
    breakdown: dict[FailureCause, int]
    patterns: dict[str, dict[str, float]]
    recommendations: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def failure_rate(self) -> float:
        """Fraction of queries diagnosed as failing."""
        if self.total_queries == 0:
            return 0.0
        return self.failing_queries / self.total_queries

    def summary(self) -> str:
        """Human-readable summary for console output."""
        lines = [
            f"Diagnosed {self.failing_queries}/{self.total_queries} failing queries "
            f"({self.failure_rate:.0%} failure rate)",
        ]
        if self.breakdown:
            lines.append("")
            lines.append("Failure causes:")
            for cause, count in sorted(self.breakdown.items(), key=lambda kv: -kv[1]):
                share = count / self.failing_queries
                lines.append(f"  {cause.value:26s} {count:4d}  ({share:.0%})")
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            lines.extend(f"  {i}. {rec}" for i, rec in enumerate(self.recommendations, start=1))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "total_queries": self.total_queries,
            "failing_queries": self.failing_queries,
            "failure_rate": self.failure_rate,
            "timestamp": self.timestamp.isoformat(),
            "breakdown": {cause.value: count for cause, count in self.breakdown.items()},
            "patterns": self.patterns,
            "recommendations": list(self.recommendations),
            "diagnoses": [
                {
                    "index": d.index,
                    "question": d.question,
                    "cause": d.cause.value,
                    "tier": d.tier,
                    "evidence": d.evidence,
                }
                for d in self.diagnoses
            ],
        }
