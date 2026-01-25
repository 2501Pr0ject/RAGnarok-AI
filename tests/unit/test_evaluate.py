import pytest

from ragnarok_ai.core.evaluate import EvaluationResult, evaluate
from ragnarok_ai.core.exceptions import EvaluationError
from ragnarok_ai.core.types import Document, Query, RAGResponse, TestSet


class MockRAG:
    """Mock RAG pipeline for testing."""

    async def query(self, question: str) -> RAGResponse:
        if "fail" in question:
            raise ValueError("Simulated failure")

        return RAGResponse(
            answer=f"Answer to {question}",
            retrieved_docs=[
                Document(id="doc_1", content="Content 1"),
                Document(id="doc_2", content="Content 2"),
            ],
        )


@pytest.mark.asyncio
async def test_evaluate_success():
    rag = MockRAG()
    testset = TestSet(
        queries=[
            Query(text="q1", ground_truth_docs=["doc_1"]),
            Query(text="q2", ground_truth_docs=["doc_3"]),
        ]
    )

    result = await evaluate(rag, testset)

    assert isinstance(result, EvaluationResult)
    assert len(result.responses) == 2
    assert len(result.metrics) == 2
    summary = result.summary()
    assert summary["num_queries"] == 2
    assert "precision" in summary
    assert "recall" in summary


@pytest.mark.asyncio
async def test_evaluate_failure():
    rag = MockRAG()
    testset = TestSet(
        queries=[
            Query(text="fail this query", ground_truth_docs=[]),
        ]
    )

    with pytest.raises(EvaluationError) as excinfo:
        await evaluate(rag, testset)

    assert "Failed to query pipeline" in str(excinfo.value)
    assert "fail this query" in str(excinfo.value)


@pytest.mark.asyncio
async def test_evaluate_empty_testset():
    rag = MockRAG()
    testset = TestSet(queries=[])

    result = await evaluate(rag, testset)

    assert len(result.responses) == 0
    assert result.summary() == {}
