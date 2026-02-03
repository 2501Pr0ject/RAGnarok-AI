"""Unit tests for the DSPy adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ragnarok_ai.adapters.frameworks.dspy import (
    DSPyModuleAdapter,
    DSPyRAGAdapter,
    DSPyRetrieverAdapter,
    _convert_passage_to_document,
    _convert_passages_to_documents,
    create_dspy_adapter,
)
from ragnarok_ai.core.types import Document, RAGResponse

# ============================================================================
# Mock DSPy Classes
# ============================================================================


class MockPassage:
    """Mock DSPy passage object."""

    def __init__(
        self,
        text: str,
        pid: str | None = None,
        title: str | None = None,
        score: float | None = None,
    ) -> None:
        self.text = text
        self.pid = pid
        self.title = title
        self.score = score


class MockRetrievalResult:
    """Mock DSPy retrieval result."""

    def __init__(self, passages: list[Any]) -> None:
        self.passages = passages


class MockRetriever:
    """Mock DSPy Retrieve module."""

    def __init__(self, passages: list[Any]) -> None:
        self._passages = passages

    def __call__(self, query: str) -> MockRetrievalResult:  # noqa: ARG002
        return MockRetrievalResult(self._passages)


class MockGeneratorResult:
    """Mock DSPy generator result."""

    def __init__(self, answer: str, rationale: str | None = None) -> None:
        self.answer = answer
        self.rationale = rationale


class MockGenerator:
    """Mock DSPy generator module (Predict, ChainOfThought, etc.)."""

    def __init__(self, answer: str, rationale: str | None = None) -> None:
        self._answer = answer
        self._rationale = rationale

    def __call__(self, **kwargs: Any) -> MockGeneratorResult:  # noqa: ARG002
        return MockGeneratorResult(self._answer, self._rationale)


class MockRAGModuleResult:
    """Mock DSPy RAG module result."""

    def __init__(
        self,
        answer: str,
        passages: list[Any] | None = None,
        rationale: str | None = None,
    ) -> None:
        self.answer = answer
        self.passages = passages or []
        self.rationale = rationale


class MockRAGModule:
    """Mock DSPy RAG module."""

    def __init__(self, answer: str, passages: list[Any] | None = None) -> None:
        self._answer = answer
        self._passages = passages or []

    def __call__(self, **kwargs: Any) -> MockRAGModuleResult:  # noqa: ARG002
        return MockRAGModuleResult(self._answer, self._passages)


# ============================================================================
# Passage Conversion Tests
# ============================================================================


class TestPassageConversion:
    """Tests for DSPy passage conversion."""

    def test_convert_string_passage(self) -> None:
        """Test converting a string passage."""
        doc = _convert_passage_to_document("This is a passage", 0)

        assert doc.content == "This is a passage"
        assert doc.metadata["index"] == 0

    def test_convert_dict_passage(self) -> None:
        """Test converting a dict passage."""
        passage = {
            "id": "doc123",
            "text": "Passage content",
            "title": "Document Title",
        }
        doc = _convert_passage_to_document(passage, 0)

        assert doc.id == "doc123"
        assert doc.content == "Passage content"
        assert doc.metadata["title"] == "Document Title"

    def test_convert_dict_passage_with_pid(self) -> None:
        """Test converting dict with pid instead of id."""
        passage = {"pid": "passage456", "content": "Content"}
        doc = _convert_passage_to_document(passage, 0)

        assert doc.id == "passage456"
        assert doc.content == "Content"

    def test_convert_object_passage(self) -> None:
        """Test converting an object passage."""
        passage = MockPassage(
            text="Object passage content",
            pid="obj123",
            title="Title",
            score=0.95,
        )
        doc = _convert_passage_to_document(passage, 0)

        assert doc.id == "obj123"
        assert doc.content == "Object passage content"
        assert doc.metadata["title"] == "Title"
        assert doc.metadata["score"] == 0.95

    def test_convert_passages_list(self) -> None:
        """Test converting a list of passages."""
        passages = [
            "String passage",
            {"id": "dict1", "text": "Dict passage"},
            MockPassage("Object passage", "obj1"),
        ]
        docs = _convert_passages_to_documents(passages)

        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].content == "String passage"
        assert docs[1].id == "dict1"
        assert docs[2].id == "obj1"


# ============================================================================
# DSPyRetrieverAdapter Tests
# ============================================================================


class TestDSPyRetrieverAdapter:
    """Tests for DSPyRetrieverAdapter."""

    @pytest.mark.asyncio
    async def test_query_basic(self) -> None:
        """Test basic query with retriever."""
        passages = [
            MockPassage("First passage", "p1", score=0.9),
            MockPassage("Second passage", "p2", score=0.8),
        ]
        retriever = MockRetriever(passages)
        adapter = DSPyRetrieverAdapter(retriever)

        response = await adapter.query("test query")

        assert isinstance(response, RAGResponse)
        assert len(response.retrieved_docs) == 2
        assert response.retrieved_docs[0].id == "p1"
        assert response.retrieved_docs[1].id == "p2"
        assert "dspy_retriever" in response.metadata["adapter"]

    @pytest.mark.asyncio
    async def test_query_with_string_passages(self) -> None:
        """Test query returning string passages."""
        passages = ["Passage 1", "Passage 2", "Passage 3"]
        retriever = MockRetriever(passages)
        adapter = DSPyRetrieverAdapter(retriever)

        response = await adapter.query("test query")

        assert len(response.retrieved_docs) == 3
        assert response.retrieved_docs[0].content == "Passage 1"

    @pytest.mark.asyncio
    async def test_query_with_answer_generator(self) -> None:
        """Test query with custom answer generator."""
        passages = [MockPassage("Content", "p1")]
        retriever = MockRetriever(passages)

        def custom_generator(question: str, docs: list[Document]) -> str:
            return f"Answer for {question} with {len(docs)} docs"

        adapter = DSPyRetrieverAdapter(retriever, answer_generator=custom_generator)
        response = await adapter.query("what is RAG?")

        assert response.answer == "Answer for what is RAG? with 1 docs"

    @pytest.mark.asyncio
    async def test_query_empty_results(self) -> None:
        """Test query with no results."""
        retriever = MockRetriever([])
        adapter = DSPyRetrieverAdapter(retriever)

        response = await adapter.query("obscure query")

        assert len(response.retrieved_docs) == 0
        assert "0 passages" in response.answer

    def test_retriever_property(self) -> None:
        """Test retriever property access."""
        retriever = MockRetriever([])
        adapter = DSPyRetrieverAdapter(retriever)

        assert adapter.retriever is retriever


# ============================================================================
# DSPyModuleAdapter Tests
# ============================================================================


class TestDSPyModuleAdapter:
    """Tests for DSPyModuleAdapter."""

    @pytest.mark.asyncio
    async def test_query_basic(self) -> None:
        """Test basic query with RAG module."""
        passages = [MockPassage("Source content", "src1")]
        module = MockRAGModule("The answer is 42", passages)
        adapter = DSPyModuleAdapter(module)

        response = await adapter.query("What is the answer?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "The answer is 42"
        assert len(response.retrieved_docs) == 1
        assert "dspy_module" in response.metadata["adapter"]

    @pytest.mark.asyncio
    async def test_query_custom_fields(self) -> None:
        """Test query with custom field names."""

        class CustomResult:
            def __init__(self) -> None:
                self.response = "Custom response"
                self.retrieved_docs = [{"id": "d1", "text": "Doc content"}]

        class CustomModule:
            def __call__(self, **kwargs: Any) -> CustomResult:  # noqa: ARG002
                return CustomResult()

        adapter = DSPyModuleAdapter(
            CustomModule(),
            answer_field="response",
            passages_field="retrieved_docs",
        )
        response = await adapter.query("test")

        assert response.answer == "Custom response"
        assert len(response.retrieved_docs) == 1

    @pytest.mark.asyncio
    async def test_query_no_passages(self) -> None:
        """Test query with no passages in result."""
        module = MockRAGModule("Answer without context")
        adapter = DSPyModuleAdapter(module)

        response = await adapter.query("Simple question")

        assert response.answer == "Answer without context"
        assert len(response.retrieved_docs) == 0

    @pytest.mark.asyncio
    async def test_query_custom_input_field(self) -> None:
        """Test query with custom input field name."""

        class QueryModule:
            def __call__(self, **kwargs: Any) -> MockGeneratorResult:
                # Verify correct input field is used
                assert "query" in kwargs
                return MockGeneratorResult(f"Answer to: {kwargs['query']}")

        adapter = DSPyModuleAdapter(QueryModule(), input_field="query")
        response = await adapter.query("test question")

        assert "test question" in response.answer

    def test_module_property(self) -> None:
        """Test module property access."""
        module = MockRAGModule("test")
        adapter = DSPyModuleAdapter(module)

        assert adapter.module is module


# ============================================================================
# DSPyRAGAdapter Tests
# ============================================================================


class TestDSPyRAGAdapter:
    """Tests for DSPyRAGAdapter."""

    @pytest.mark.asyncio
    async def test_query_basic(self) -> None:
        """Test basic RAG query."""
        passages = [
            MockPassage("Context about France", "p1"),
            MockPassage("More context", "p2"),
        ]
        retriever = MockRetriever(passages)
        generator = MockGenerator("Paris is the capital")
        adapter = DSPyRAGAdapter(retriever, generator)

        response = await adapter.query("What is the capital of France?")

        assert response.answer == "Paris is the capital"
        assert len(response.retrieved_docs) == 2
        assert "dspy_rag" in response.metadata["adapter"]

    @pytest.mark.asyncio
    async def test_query_string_passages(self) -> None:
        """Test RAG with string passages."""
        retriever = MockRetriever(["Passage 1", "Passage 2"])
        generator = MockGenerator("Generated answer")
        adapter = DSPyRAGAdapter(retriever, generator)

        response = await adapter.query("test")

        assert response.answer == "Generated answer"
        assert len(response.retrieved_docs) == 2

    @pytest.mark.asyncio
    async def test_query_empty_context(self) -> None:
        """Test RAG with no retrieved passages."""
        retriever = MockRetriever([])
        generator = MockGenerator("No context answer")
        adapter = DSPyRAGAdapter(retriever, generator)

        response = await adapter.query("obscure question")

        assert response.answer == "No context answer"
        assert len(response.retrieved_docs) == 0

    def test_properties(self) -> None:
        """Test retriever and generator properties."""
        retriever = MockRetriever([])
        generator = MockGenerator("test")
        adapter = DSPyRAGAdapter(retriever, generator)

        assert adapter.retriever is retriever
        assert adapter.generator is generator


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateDSPyAdapter:
    """Tests for create_dspy_adapter factory function."""

    def test_create_requires_dspy(self) -> None:
        """Test that factory requires dspy when not installed."""
        from unittest.mock import patch

        # Mock dspy import to simulate it not being installed
        with patch.dict("sys.modules", {"dspy": None}):
            # The factory function checks for dspy types, which will fail
            # when dspy isn't available, raising TypeError for unknown types
            with pytest.raises((ImportError, TypeError)):
                create_dspy_adapter(MagicMock())


# ============================================================================
# Integration Tests
# ============================================================================


class TestDSPyAdapterIntegration:
    """Integration tests for DSPy adapters."""

    @pytest.mark.asyncio
    async def test_full_rag_pipeline(self) -> None:
        """Test full RAG pipeline simulation."""
        # Simulate a complete DSPy RAG flow
        passages = [
            MockPassage("Paris is the capital of France.", "wiki_france", score=0.98),
            MockPassage("France is located in Europe.", "wiki_europe", score=0.92),
        ]
        retriever = MockRetriever(passages)
        generator = MockGenerator("Paris is the capital of France, located in Europe.")
        adapter = DSPyRAGAdapter(retriever, generator)

        result = await adapter.query("What is the capital of France?")

        assert "Paris" in result.answer
        assert len(result.retrieved_docs) == 2
        assert result.retrieved_docs[0].metadata["score"] == 0.98

    @pytest.mark.asyncio
    async def test_module_with_rationale(self) -> None:
        """Test module that includes rationale."""

        class RationaleResult:
            def __init__(self) -> None:
                self.answer = "42"
                self.rationale = "Because it's the answer to everything"
                self.context = ["Hitchhiker's Guide reference"]

        class RationaleModule:
            def __call__(self, **kwargs: Any) -> RationaleResult:  # noqa: ARG002
                return RationaleResult()

        adapter = DSPyModuleAdapter(RationaleModule())
        result = await adapter.query("What is the meaning of life?")

        assert result.answer == "42"
        assert len(result.retrieved_docs) == 1
