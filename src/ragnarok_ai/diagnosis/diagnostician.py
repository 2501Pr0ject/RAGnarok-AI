"""Root-cause diagnosis of RAG evaluation failures.

Two-tier design, consistent with the rest of the framework:

- **Heuristic tier (free)**: classifies everything decidable from the
  metrics an evaluation already computed — pipeline errors, retrieval
  misses (ground truth never retrieved), ranking problems (retrieved but
  buried), empty answers.
- **LLM tier (opt-in)**: when an ``llm`` and the document corpus are
  provided, queries that pass retrieval are additionally checked on the
  generation side with closed YES/NO questions: is the context sufficient
  (chunking), is the answer supported (hallucination), is it complete?

Example:
    >>> from ragnarok_ai import RAGDiagnostician, evaluate
    >>>
    >>> result = await evaluate(rag, testset, metrics=["retrieval"])
    >>> diagnostician = RAGDiagnostician()
    >>> report = await diagnostician.diagnose(result)
    >>> print(report.summary())
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from ragnarok_ai.diagnosis.models import (
    RECOMMENDATIONS,
    DiagnosisReport,
    DiagnosisThresholds,
    FailureCause,
    QueryDiagnosis,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ragnarok_ai.core.evaluate import EvaluationResult, QueryResult
    from ragnarok_ai.core.protocols import LLMProtocol
    from ragnarok_ai.core.types import Document

logger = logging.getLogger(__name__)

ANSWERABILITY_PROMPT = """Can the following question be fully answered using ONLY the context below?

Question: {question}

Context:
{context}

Answer with YES or NO only."""

SUPPORT_PROMPT = """Is every factual claim in the answer below supported by the context?

Context:
{context}

Answer: {answer}

Reply with YES or NO only."""

COMPLETENESS_PROMPT = """Does the answer below fully address the question?

Question: {question}

Answer: {answer}

Reply with YES or NO only."""


def _parse_yes_no(response: str) -> bool | None:
    """Parse a YES/NO verdict from an LLM response; None if unclear."""
    token = response.strip().upper()
    has_yes = re.search(r"\bYES\b", token) is not None
    has_no = re.search(r"\bNO\b", token) is not None
    if has_yes and has_no:
        return None  # ambiguous
    if has_yes or token.startswith("YES"):
        return True
    if has_no or token.startswith("NO"):
        return False
    return None


class RAGDiagnostician:
    """Classify evaluation failures by root cause.

    Attributes:
        llm: Optional LLM for generation-side diagnosis (any small local
            model works; the checks are closed YES/NO questions).
        thresholds: Pass/fail thresholds.

    Example:
        >>> diagnostician = RAGDiagnostician(llm=OllamaLLM(model="mistral"))
        >>> report = await diagnostician.diagnose(result, documents=corpus)
        >>> report.breakdown
        {<FailureCause.RETRIEVAL_MISS>: 14, <FailureCause.GENERATION_HALLUCINATION>: 4}
    """

    def __init__(
        self,
        llm: LLMProtocol | None = None,
        thresholds: DiagnosisThresholds | None = None,
    ) -> None:
        """Initialize the diagnostician.

        Args:
            llm: Optional LLM enabling generation-side checks.
            thresholds: Optional custom pass/fail thresholds.
        """
        self.llm = llm
        self.thresholds = thresholds or DiagnosisThresholds()

    async def diagnose(
        self,
        result: EvaluationResult,
        *,
        documents: Mapping[str, Document | str] | None = None,
    ) -> DiagnosisReport:
        """Diagnose the failures of an evaluation run.

        Args:
            result: An ``EvaluationResult`` from ``evaluate()``. Queries
                need ``ground_truth_docs`` for retrieval-side diagnosis
                (generated test sets have them).
            documents: The document corpus as an id → Document (or id →
                text) mapping. Required for the LLM tier, which needs the
                retrieved chunks' content to judge the generation side.

        Returns:
            A ``DiagnosisReport`` aggregating per-query diagnoses,
            cause breakdown, metadata patterns, and recommendations.
        """
        diagnoses: list[QueryDiagnosis] = []
        failing_indices: set[int] = set()

        for index, qr in enumerate(result.query_results):
            diagnosis = self._diagnose_heuristic(index, qr)
            if diagnosis is None and self.llm is not None and documents is not None:
                diagnosis = await self._diagnose_generation(index, qr, documents)
            if diagnosis is not None:
                diagnoses.append(diagnosis)
                failing_indices.add(index)

        breakdown = dict(Counter(d.cause for d in diagnoses))
        return DiagnosisReport(
            total_queries=len(result.query_results),
            failing_queries=len(diagnoses),
            diagnoses=diagnoses,
            breakdown=breakdown,
            patterns=self._patterns(result.query_results, failing_indices),
            recommendations=self._recommendations(breakdown),
        )

    # ── Heuristic tier ───────────────────────────────────────────────────

    def _diagnose_heuristic(self, index: int, qr: QueryResult) -> QueryDiagnosis | None:
        """Classify what the existing metrics can decide; None if passing
        (or only decidable by the LLM tier)."""
        if qr.error is not None:
            return QueryDiagnosis(
                index=index,
                question=qr.query.text,
                cause=FailureCause.PIPELINE_ERROR,
                tier="heuristic",
                evidence={"error": str(qr.error), "error_type": type(qr.error).__name__},
            )

        if not qr.answer.strip():
            return QueryDiagnosis(
                index=index,
                question=qr.query.text,
                cause=FailureCause.GENERATION_INCOMPLETE,
                tier="heuristic",
                evidence={"empty_answer": True},
            )

        # Retrieval-side diagnosis needs ground truth
        if not qr.query.ground_truth_docs:
            return None

        if qr.metric.recall < self.thresholds.recall_pass:
            missing = [d for d in qr.query.ground_truth_docs if d not in qr.retrieved_doc_ids]
            evidence: dict[str, Any] = {
                "recall": qr.metric.recall,
                "mrr": qr.metric.mrr,
                "missing_doc_ids": missing,
                "retrieved_doc_ids": list(qr.retrieved_doc_ids),
            }
            if qr.metric.recall > 0 and qr.metric.mrr < self.thresholds.mrr_pass:
                return QueryDiagnosis(
                    index=index,
                    question=qr.query.text,
                    cause=FailureCause.RETRIEVAL_RANKING,
                    tier="heuristic",
                    evidence=evidence,
                )
            return QueryDiagnosis(
                index=index,
                question=qr.query.text,
                cause=FailureCause.RETRIEVAL_MISS,
                tier="heuristic",
                evidence=evidence,
            )

        return None  # retrieval passes; generation side needs the LLM tier

    # ── LLM tier ─────────────────────────────────────────────────────────

    async def _diagnose_generation(
        self,
        index: int,
        qr: QueryResult,
        documents: Mapping[str, Document | str],
    ) -> QueryDiagnosis | None:
        """Closed YES/NO checks on the generation side; None if passing
        or inconclusive."""
        context = self._build_context(qr.retrieved_doc_ids, documents)
        if not context:
            return None
        assert self.llm is not None  # guarded by caller

        answerable = await self._ask(ANSWERABILITY_PROMPT.format(question=qr.query.text, context=context))
        if answerable is False:
            return QueryDiagnosis(
                index=index,
                question=qr.query.text,
                cause=FailureCause.CONTEXT_INSUFFICIENT,
                tier="llm",
                evidence={"answerable": False, "retrieved_doc_ids": list(qr.retrieved_doc_ids)},
            )

        supported = await self._ask(SUPPORT_PROMPT.format(context=context, answer=qr.answer))
        if supported is False:
            return QueryDiagnosis(
                index=index,
                question=qr.query.text,
                cause=FailureCause.GENERATION_HALLUCINATION,
                tier="llm",
                evidence={"answer_supported": False},
            )

        complete = await self._ask(COMPLETENESS_PROMPT.format(question=qr.query.text, answer=qr.answer))
        if complete is False:
            return QueryDiagnosis(
                index=index,
                question=qr.query.text,
                cause=FailureCause.GENERATION_INCOMPLETE,
                tier="llm",
                evidence={"answer_complete": False},
            )

        return None  # every check passed (or was inconclusive)

    async def _ask(self, prompt: str) -> bool | None:
        """Run one closed question against the LLM; None on failure."""
        assert self.llm is not None  # guarded by callers
        try:
            response = await self.llm.generate(prompt)
        except Exception:
            logger.warning("Diagnosis LLM call failed; skipping check.", exc_info=True)
            return None
        return _parse_yes_no(response)

    @staticmethod
    def _build_context(doc_ids: Sequence[str], documents: Mapping[str, Document | str]) -> str:
        """Concatenate retrieved documents' content, in rank order."""
        parts = []
        for doc_id in doc_ids:
            doc = documents.get(doc_id)
            if doc is None:
                continue
            parts.append(doc if isinstance(doc, str) else doc.content)
        return "\n\n".join(parts)

    # ── Aggregation ──────────────────────────────────────────────────────

    @staticmethod
    def _patterns(
        query_results: Sequence[QueryResult],
        failing_indices: set[int],
    ) -> dict[str, dict[str, float]]:
        """Failure rate per query-metadata value (string-valued keys only)."""
        totals: dict[str, Counter[str]] = {}
        failures: dict[str, Counter[str]] = {}
        for index, qr in enumerate(query_results):
            for key, value in qr.query.metadata.items():
                if not isinstance(value, str):
                    continue
                totals.setdefault(key, Counter())[value] += 1
                if index in failing_indices:
                    failures.setdefault(key, Counter())[value] += 1

        return {
            key: {value: failures.get(key, Counter())[value] / count for value, count in value_counts.items()}
            for key, value_counts in totals.items()
        }

    @staticmethod
    def _recommendations(breakdown: dict[FailureCause, int]) -> list[str]:
        """Recommendations ordered by how many failures each cause explains."""
        ordered = sorted(breakdown.items(), key=lambda kv: -kv[1])
        return [RECOMMENDATIONS[cause] for cause, _count in ordered]
