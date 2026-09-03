"""Tests for root-cause diagnosis of RAG evaluation failures."""

from __future__ import annotations

import pytest

from ragnarok_ai.core.evaluate import EvaluationResult, QueryResult
from ragnarok_ai.core.types import Query, TestSet
from ragnarok_ai.diagnosis.diagnostician import RAGDiagnostician, _parse_yes_no
from ragnarok_ai.diagnosis.models import DiagnosisThresholds, FailureCause
from ragnarok_ai.evaluators.retrieval import RetrievalMetrics


def make_query_result(
    text: str = "What is CHF?",
    *,
    ground_truth: list[str] | None = None,
    retrieved: list[str] | None = None,
    recall: float = 1.0,
    mrr: float = 1.0,
    answer: str = "An answer.",
    error: Exception | None = None,
    metadata: dict[str, str] | None = None,
) -> QueryResult:
    """Build a QueryResult with controllable diagnosis signals."""
    return QueryResult(
        query=Query(
            text=text,
            ground_truth_docs=ground_truth if ground_truth is not None else ["doc1"],
            metadata=metadata or {},
        ),
        metric=RetrievalMetrics(precision=0.5, recall=recall, mrr=mrr, ndcg=0.5, k=10),
        answer=answer,
        latency_ms=100.0,
        error=error,
        retrieved_doc_ids=retrieved if retrieved is not None else ["doc1"],
    )


def make_result(query_results: list[QueryResult]) -> EvaluationResult:
    """Wrap QueryResults into an EvaluationResult."""
    return EvaluationResult(
        testset=TestSet(queries=[qr.query for qr in query_results]),
        metrics=[qr.metric for qr in query_results],
        responses=[qr.answer for qr in query_results],
        query_results=query_results,
    )


class FakeLLM:
    """Scripted LLMProtocol implementation for tests."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]

    async def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0]


class TestParseYesNo:
    """Test suite for YES/NO parsing."""

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("YES", True),
            ("yes.", True),
            ("No", False),
            ("NO, it is not.", False),
            ("The answer is YES", True),
            ("I cannot tell", None),
            ("YES and NO", None),
        ],
    )
    def test_parsing(self, response: str, expected: bool | None) -> None:
        assert _parse_yes_no(response) is expected


class TestHeuristicTier:
    """Test suite for metric-based diagnosis (no LLM)."""

    @pytest.mark.asyncio
    async def test_pipeline_error_is_first_priority(self) -> None:
        qr = make_query_result(error=RuntimeError("boom"), recall=0.0, answer="")
        report = await RAGDiagnostician().diagnose(make_result([qr]))

        assert report.failing_queries == 1
        diagnosis = report.diagnoses[0]
        assert diagnosis.cause == FailureCause.PIPELINE_ERROR
        assert diagnosis.tier == "heuristic"
        assert diagnosis.evidence["error_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_retrieval_miss_when_nothing_relevant_retrieved(self) -> None:
        qr = make_query_result(
            ground_truth=["doc1", "doc2"],
            retrieved=["other1", "other2"],
            recall=0.0,
            mrr=0.0,
        )
        report = await RAGDiagnostician().diagnose(make_result([qr]))

        diagnosis = report.diagnoses[0]
        assert diagnosis.cause == FailureCause.RETRIEVAL_MISS
        assert diagnosis.evidence["missing_doc_ids"] == ["doc1", "doc2"]

    @pytest.mark.asyncio
    async def test_ranking_problem_when_found_but_buried(self) -> None:
        # Partial recall with low MRR: the relevant doc is there, ranked low
        qr = make_query_result(
            ground_truth=["doc1", "doc2"],
            retrieved=["other1", "other2", "doc1"],
            recall=0.4,
            mrr=0.33,
        )
        report = await RAGDiagnostician().diagnose(make_result([qr]))

        assert report.diagnoses[0].cause == FailureCause.RETRIEVAL_RANKING

    @pytest.mark.asyncio
    async def test_empty_answer_is_generation_incomplete(self) -> None:
        qr = make_query_result(answer="   ")
        report = await RAGDiagnostician().diagnose(make_result([qr]))

        diagnosis = report.diagnoses[0]
        assert diagnosis.cause == FailureCause.GENERATION_INCOMPLETE
        assert diagnosis.evidence["empty_answer"] is True

    @pytest.mark.asyncio
    async def test_passing_query_is_not_diagnosed(self) -> None:
        report = await RAGDiagnostician().diagnose(make_result([make_query_result()]))

        assert report.failing_queries == 0
        assert report.diagnoses == []
        assert report.failure_rate == 0.0

    @pytest.mark.asyncio
    async def test_no_ground_truth_skips_retrieval_diagnosis(self) -> None:
        qr = make_query_result(ground_truth=[], recall=0.0, mrr=0.0)
        report = await RAGDiagnostician().diagnose(make_result([qr]))

        assert report.failing_queries == 0  # undecidable without LLM tier

    @pytest.mark.asyncio
    async def test_thresholds_are_configurable(self) -> None:
        qr = make_query_result(recall=0.7, mrr=1.0)  # passes defaults

        strict = RAGDiagnostician(thresholds=DiagnosisThresholds(recall_pass=0.9))
        report = await strict.diagnose(make_result([qr]))

        assert report.diagnoses[0].cause == FailureCause.RETRIEVAL_MISS


class TestLLMTier:
    """Test suite for LLM-assisted generation diagnosis."""

    @pytest.fixture
    def documents(self) -> dict[str, str]:
        return {"doc1": "CHF is congestive heart failure.", "other1": "Unrelated."}

    @pytest.mark.asyncio
    async def test_context_insufficient_when_not_answerable(self, documents: dict[str, str]) -> None:
        llm = FakeLLM(["NO"])  # answerability check fails
        qr = make_query_result()
        report = await RAGDiagnostician(llm=llm).diagnose(make_result([qr]), documents=documents)

        diagnosis = report.diagnoses[0]
        assert diagnosis.cause == FailureCause.CONTEXT_INSUFFICIENT
        assert diagnosis.tier == "llm"
        assert len(llm.calls) == 1  # stops at the first failing check

    @pytest.mark.asyncio
    async def test_hallucination_when_answer_unsupported(self, documents: dict[str, str]) -> None:
        llm = FakeLLM(["YES", "NO"])  # answerable, but unsupported
        qr = make_query_result()
        report = await RAGDiagnostician(llm=llm).diagnose(make_result([qr]), documents=documents)

        assert report.diagnoses[0].cause == FailureCause.GENERATION_HALLUCINATION

    @pytest.mark.asyncio
    async def test_incomplete_when_answer_partial(self, documents: dict[str, str]) -> None:
        llm = FakeLLM(["YES", "YES", "NO"])  # answerable, supported, incomplete
        qr = make_query_result()
        report = await RAGDiagnostician(llm=llm).diagnose(make_result([qr]), documents=documents)

        assert report.diagnoses[0].cause == FailureCause.GENERATION_INCOMPLETE

    @pytest.mark.asyncio
    async def test_all_checks_passing_means_no_failure(self, documents: dict[str, str]) -> None:
        llm = FakeLLM(["YES", "YES", "YES"])
        qr = make_query_result()
        report = await RAGDiagnostician(llm=llm).diagnose(make_result([qr]), documents=documents)

        assert report.failing_queries == 0

    @pytest.mark.asyncio
    async def test_retrieval_failures_never_reach_the_llm(self, documents: dict[str, str]) -> None:
        llm = FakeLLM(["NO"])
        qr = make_query_result(retrieved=["other1"], recall=0.0, mrr=0.0)
        report = await RAGDiagnostician(llm=llm).diagnose(make_result([qr]), documents=documents)

        assert report.diagnoses[0].cause == FailureCause.RETRIEVAL_MISS
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_llm_error_is_inconclusive_not_failing(self, documents: dict[str, str]) -> None:
        class FailingLLM:
            async def generate(self, prompt: str) -> str:  # noqa: ARG002
                msg = "connection refused"
                raise RuntimeError(msg)

            async def embed(self, text: str) -> list[float]:  # noqa: ARG002
                return [0.0]

        qr = make_query_result()
        report = await RAGDiagnostician(llm=FailingLLM()).diagnose(make_result([qr]), documents=documents)

        assert report.failing_queries == 0

    @pytest.mark.asyncio
    async def test_unknown_documents_skip_llm_checks(self) -> None:
        llm = FakeLLM(["NO"])
        qr = make_query_result(retrieved=["not-in-corpus"])
        report = await RAGDiagnostician(llm=llm).diagnose(make_result([qr]), documents={})

        assert report.failing_queries == 0
        assert llm.calls == []


class TestAggregation:
    """Test suite for report aggregation."""

    @pytest.mark.asyncio
    async def test_breakdown_and_recommendations_are_ordered(self) -> None:
        query_results = [
            *[
                make_query_result(retrieved=["other"], recall=0.0, mrr=0.0)
                for _ in range(3)  # 3 retrieval misses
            ],
            make_query_result(error=RuntimeError("boom")),  # 1 pipeline error
            make_query_result(),  # 1 passing
        ]
        report = await RAGDiagnostician().diagnose(make_result(query_results))

        assert report.total_queries == 5
        assert report.failing_queries == 4
        assert report.breakdown[FailureCause.RETRIEVAL_MISS] == 3
        assert report.breakdown[FailureCause.PIPELINE_ERROR] == 1
        # Dominant cause first
        assert "not being retrieved" in report.recommendations[0]

    @pytest.mark.asyncio
    async def test_patterns_surface_failing_question_types(self) -> None:
        query_results = [
            make_query_result(metadata={"type": "simple"}),
            make_query_result(metadata={"type": "simple"}),
            make_query_result(retrieved=["other"], recall=0.0, mrr=0.0, metadata={"type": "multi_hop"}),
            make_query_result(retrieved=["other"], recall=0.0, mrr=0.0, metadata={"type": "multi_hop"}),
        ]
        report = await RAGDiagnostician().diagnose(make_result(query_results))

        assert report.patterns["type"]["simple"] == 0.0
        assert report.patterns["type"]["multi_hop"] == 1.0

    @pytest.mark.asyncio
    async def test_summary_and_to_dict(self) -> None:
        query_results = [
            make_query_result(retrieved=["other"], recall=0.0, mrr=0.0),
            make_query_result(),
        ]
        report = await RAGDiagnostician().diagnose(make_result(query_results))

        text = report.summary()
        assert "1/2 failing" in text
        assert "retrieval_miss" in text
        assert "Recommendations:" in text

        data = report.to_dict()
        assert data["failure_rate"] == 0.5
        assert data["breakdown"] == {"retrieval_miss": 1}
        assert data["diagnoses"][0]["cause"] == "retrieval_miss"
