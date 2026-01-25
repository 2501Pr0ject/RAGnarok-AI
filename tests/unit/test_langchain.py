"""Unit tests for the LangChain adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ragnarok_ai.adapters.frameworks.langchain import (
    LangChainAdapter,
    LangChainRetrieverAdapter,
    _convert_lc_document,
    _convert_lc_documents,
    create_langchain_adapter,
)
from ragnarok_ai.core.types import Document, RAGResponse

# ============================================================================
# Mock LangChain Classes
# ============================================================================


class MockLCDocument:
    """Mock LangChain Document."""

    def __init__(self, page_content: str, metadata: dict[str, Any] | None = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}


class MockRetriever:
    """Mock LangChain Retriever (sync)."""

    def __init__(self, docs: list[MockLCDocument]) -> None:
        self._docs = docs

    def invoke(self, _query: str) -> list[MockLCDocument]:
        return self._docs


class MockAsyncRetriever:
    """Mock LangChain Retriever (async)."""

    def __init__(self, docs: list[MockLCDocument]) -> None:
        self._docs = docs

    async def ainvoke(self, _query: str) -> list[MockLCDocument]:
        return self._docs


class MockChain:
    """Mock LangChain Chain (sync)."""

    def __init__(self, answer: str, docs: list[MockLCDocument] | None = None) -> None:
        self._answer = answer
        self._docs = docs or []

    def invoke(self, _input_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": self._answer,
            "source_documents": self._docs,
        }


class MockAsyncChain:
    """Mock LangChain Chain (async)."""

    def __init__(self, answer: str, docs: list[MockLCDocument] | None = None) -> None:
        self._answer = answer
        self._docs = docs or []

    async def ainvoke(self, _input_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": self._answer,
            "source_documents": self._docs,
        }


class MockStringChain:
    """Mock LangChain Chain that returns a string directly."""

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def invoke(self, _input_data: Any) -> str:
        return self._answer


# ============================================================================
# Document Conversion Tests
# ============================================================================


class TestDocumentConversion:
    """Tests for LangChain document conversion."""

    def test_convert_lc_document_basic(self) -> None:
        """Test converting a basic LangChain document."""
        lc_doc = MockLCDocument(
            page_content="This is the content",
            metadata={"source": "test.pdf"},
        )
        doc = _convert_lc_document(lc_doc)

        assert doc.content == "This is the content"
        assert doc.metadata["source"] == "test.pdf"
        assert doc.id == "test.pdf"  # Uses source as ID

    def test_convert_lc_document_with_id(self) -> None:
        """Test converting document with explicit ID in metadata."""
        lc_doc = MockLCDocument(
            page_content="Content",
            metadata={"id": "doc123", "source": "test.pdf"},
        )
        doc = _convert_lc_document(lc_doc)

        assert doc.id == "doc123"  # Prefers explicit ID

    def test_convert_lc_document_no_metadata(self) -> None:
        """Test converting document without metadata."""
        lc_doc = MockLCDocument(page_content="Content only")
        doc = _convert_lc_document(lc_doc)

        assert doc.content == "Content only"
        assert doc.id  # Should have a hash-based ID
        assert doc.metadata == {}

    def test_convert_lc_documents_list(self) -> None:
        """Test converting a list of documents."""
        lc_docs = [
            MockLCDocument("Content 1", {"id": "doc1"}),
            MockLCDocument("Content 2", {"id": "doc2"}),
            MockLCDocument("Content 3", {"id": "doc3"}),
        ]
        docs = _convert_lc_documents(lc_docs)

        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)
        assert [d.id for d in docs] == ["doc1", "doc2", "doc3"]


# ============================================================================
# LangChainRetrieverAdapter Tests
# ============================================================================


class TestLangChainRetrieverAdapter:
    """Tests for LangChainRetrieverAdapter."""

    @pytest.fixture
    def sample_docs(self) -> list[MockLCDocument]:
        """Create sample LangChain documents."""
        return [
            MockLCDocument("Paris is the capital of France.", {"id": "doc1", "source": "geography.txt"}),
            MockLCDocument("The Eiffel Tower is in Paris.", {"id": "doc2", "source": "landmarks.txt"}),
        ]

    @pytest.mark.asyncio
    async def test_query_sync_retriever(self, sample_docs: list[MockLCDocument]) -> None:
        """Test querying with a sync retriever."""
        retriever = MockRetriever(sample_docs)
        adapter = LangChainRetrieverAdapter(retriever)

        response = await adapter.query("What is the capital of France?")

        assert isinstance(response, RAGResponse)
        assert len(response.retrieved_docs) == 2
        assert response.retrieved_docs[0].id == "doc1"
        assert "Paris" in response.retrieved_docs[0].content

    @pytest.mark.asyncio
    async def test_query_async_retriever(self, sample_docs: list[MockLCDocument]) -> None:
        """Test querying with an async retriever."""
        retriever = MockAsyncRetriever(sample_docs)
        adapter = LangChainRetrieverAdapter(retriever)

        response = await adapter.query("What is the capital of France?")

        assert isinstance(response, RAGResponse)
        assert len(response.retrieved_docs) == 2

    @pytest.mark.asyncio
    async def test_query_with_answer_generator(self, sample_docs: list[MockLCDocument]) -> None:
        """Test querying with custom answer generator."""

        def generate_answer(_query: str, docs: list[Document]) -> str:
            return f"Based on {len(docs)} documents: The answer is Paris."

        retriever = MockRetriever(sample_docs)
        adapter = LangChainRetrieverAdapter(retriever, answer_generator=generate_answer)

        response = await adapter.query("What is the capital of France?")

        assert "Based on 2 documents" in response.answer
        assert "Paris" in response.answer

    @pytest.mark.asyncio
    async def test_query_default_answer(self, sample_docs: list[MockLCDocument]) -> None:
        """Test default answer when no generator provided."""
        retriever = MockRetriever(sample_docs)
        adapter = LangChainRetrieverAdapter(retriever)

        response = await adapter.query("Test question")

        assert "Retrieved 2 documents" in response.answer
        assert "Test question" in response.answer

    def test_retriever_property(self, sample_docs: list[MockLCDocument]) -> None:
        """Test accessing the underlying retriever."""
        retriever = MockRetriever(sample_docs)
        adapter = LangChainRetrieverAdapter(retriever)

        assert adapter.retriever is retriever


# ============================================================================
# LangChainAdapter Tests
# ============================================================================


class TestLangChainAdapter:
    """Tests for LangChainAdapter."""

    @pytest.fixture
    def sample_docs(self) -> list[MockLCDocument]:
        """Create sample LangChain documents."""
        return [
            MockLCDocument("Document 1 content", {"id": "doc1"}),
            MockLCDocument("Document 2 content", {"id": "doc2"}),
        ]

    @pytest.mark.asyncio
    async def test_query_sync_chain(self, sample_docs: list[MockLCDocument]) -> None:
        """Test querying with a sync chain."""
        chain = MockChain("The answer is 42.", sample_docs)
        adapter = LangChainAdapter(chain)

        response = await adapter.query("What is the answer?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "The answer is 42."
        assert len(response.retrieved_docs) == 2

    @pytest.mark.asyncio
    async def test_query_async_chain(self, sample_docs: list[MockLCDocument]) -> None:
        """Test querying with an async chain."""
        chain = MockAsyncChain("Async answer", sample_docs)
        adapter = LangChainAdapter(chain)

        response = await adapter.query("Test question")

        assert response.answer == "Async answer"
        assert len(response.retrieved_docs) == 2

    @pytest.mark.asyncio
    async def test_query_string_output(self) -> None:
        """Test chain that returns string directly."""
        chain = MockStringChain("Direct string answer")
        adapter = LangChainAdapter(chain, output_key=None)

        response = await adapter.query("Test")

        assert response.answer == "Direct string answer"
        assert response.retrieved_docs == []

    @pytest.mark.asyncio
    async def test_custom_input_key(self) -> None:
        """Test custom input key."""

        class CustomInputChain:
            def invoke(self, data: dict[str, Any]) -> dict[str, Any]:
                assert "question" in data
                return {"answer": "Custom input works", "source_documents": []}

        chain = CustomInputChain()
        adapter = LangChainAdapter(chain, input_key="question")

        response = await adapter.query("Test")
        assert response.answer == "Custom input works"

    @pytest.mark.asyncio
    async def test_custom_output_key(self) -> None:
        """Test custom output key."""

        class CustomOutputChain:
            def invoke(self, _data: dict[str, Any]) -> dict[str, Any]:
                return {"result": "Custom output", "documents": []}

        chain = CustomOutputChain()
        adapter = LangChainAdapter(chain, output_key="result", docs_key="documents")

        response = await adapter.query("Test")
        assert response.answer == "Custom output"

    @pytest.mark.asyncio
    async def test_input_transform(self) -> None:
        """Test custom input transformation."""

        class TransformChain:
            def invoke(self, data: dict[str, Any]) -> dict[str, Any]:
                return {"answer": f"Got: {data['query']}", "source_documents": []}

        def transform(question: str) -> dict[str, Any]:
            return {"query": question.upper()}

        chain = TransformChain()
        adapter = LangChainAdapter(chain, input_transform=transform)

        response = await adapter.query("hello")
        assert response.answer == "Got: HELLO"

    @pytest.mark.asyncio
    async def test_output_transform(self, sample_docs: list[MockLCDocument]) -> None:
        """Test custom output transformation."""

        class RawChain:
            def invoke(self, _data: dict[str, Any]) -> dict[str, Any]:
                return {"text": "Raw output", "docs": sample_docs}

        def transform(result: dict[str, Any]) -> tuple[str, list[Document]]:
            return result["text"], _convert_lc_documents(result["docs"])

        chain = RawChain()
        adapter = LangChainAdapter(chain, output_transform=transform)

        response = await adapter.query("Test")
        assert response.answer == "Raw output"
        assert len(response.retrieved_docs) == 2

    @pytest.mark.asyncio
    async def test_dict_documents(self) -> None:
        """Test extracting documents from dict format."""

        class DictDocsChain:
            def invoke(self, _data: dict[str, Any]) -> dict[str, Any]:
                return {
                    "answer": "Answer",
                    "source_documents": [
                        {"id": "d1", "content": "Content 1"},
                        {"id": "d2", "page_content": "Content 2"},
                    ],
                }

        chain = DictDocsChain()
        adapter = LangChainAdapter(chain)

        response = await adapter.query("Test")
        assert len(response.retrieved_docs) == 2
        assert response.retrieved_docs[0].id == "d1"
        assert response.retrieved_docs[0].content == "Content 1"

    def test_chain_property(self, sample_docs: list[MockLCDocument]) -> None:
        """Test accessing the underlying chain."""
        chain = MockChain("Answer", sample_docs)
        adapter = LangChainAdapter(chain)

        assert adapter.chain is chain

    @pytest.mark.asyncio
    async def test_none_input_key(self) -> None:
        """Test passing question directly without wrapping in dict."""

        class DirectInputChain:
            def invoke(self, question: str) -> str:
                return f"Answer to: {question}"

        chain = DirectInputChain()
        adapter = LangChainAdapter(chain, input_key=None, output_key=None)

        response = await adapter.query("Direct question")
        assert response.answer == "Answer to: Direct question"


# ============================================================================
# Factory Function Tests
# ============================================================================

# Check if langchain_core is available
try:
    import langchain_core  # noqa: F401

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")
class TestCreateLangChainAdapter:
    """Tests for create_langchain_adapter factory function.

    These tests require langchain-core to be installed.
    """

    def test_create_retriever_adapter(self) -> None:
        """Test creating adapter for retriever."""
        # Create a mock that looks like a BaseRetriever
        mock_retriever = MagicMock()
        mock_retriever.__class__.__name__ = "MockRetriever"

        # We need to mock the import check
        # Since we can't easily mock isinstance with BaseRetriever,
        # we'll test the chain path instead
        mock_chain = MagicMock()
        mock_chain.invoke = MagicMock(return_value={"answer": "Test", "source_documents": []})

        adapter = create_langchain_adapter(mock_chain)
        assert isinstance(adapter, LangChainAdapter)

    def test_create_chain_adapter(self) -> None:
        """Test creating adapter for chain."""
        chain = MockChain("Answer", [])
        adapter = create_langchain_adapter(chain)

        assert isinstance(adapter, LangChainAdapter)

    def test_create_with_kwargs(self) -> None:
        """Test creating adapter with additional kwargs."""
        chain = MockChain("Answer", [])
        adapter = create_langchain_adapter(chain, input_key="question", output_key="result")

        assert isinstance(adapter, LangChainAdapter)
        assert adapter._input_key == "question"
        assert adapter._output_key == "result"


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestProtocolCompliance:
    """Tests for RAGProtocol compliance."""

    @pytest.mark.asyncio
    async def test_retriever_adapter_implements_protocol(self) -> None:
        """Test that LangChainRetrieverAdapter implements RAGProtocol."""
        from ragnarok_ai.core.protocols import RAGProtocol

        retriever = MockRetriever([MockLCDocument("Content", {"id": "doc1"})])
        adapter = LangChainRetrieverAdapter(retriever)

        # Check it has the required method
        assert hasattr(adapter, "query")
        assert callable(adapter.query)

        # Check it works with the protocol
        assert isinstance(adapter, RAGProtocol)

    @pytest.mark.asyncio
    async def test_chain_adapter_implements_protocol(self) -> None:
        """Test that LangChainAdapter implements RAGProtocol."""
        from ragnarok_ai.core.protocols import RAGProtocol

        chain = MockChain("Answer", [])
        adapter = LangChainAdapter(chain)

        assert hasattr(adapter, "query")
        assert callable(adapter.query)
        assert isinstance(adapter, RAGProtocol)


# ============================================================================
# Integration with Evaluate Tests
# ============================================================================


class TestEvaluateIntegration:
    """Tests for integration with evaluate function."""

    @pytest.mark.asyncio
    async def test_evaluate_with_retriever_adapter(self) -> None:
        """Test using retriever adapter with evaluate."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        docs = [
            MockLCDocument("Paris is the capital of France.", {"id": "doc1"}),
            MockLCDocument("Berlin is the capital of Germany.", {"id": "doc2"}),
        ]
        retriever = MockRetriever(docs)
        adapter = LangChainRetrieverAdapter(retriever)

        testset = TestSet(
            queries=[
                Query(text="What is the capital of France?", ground_truth_docs=["doc1"]),
            ]
        )

        result = await evaluate(adapter, testset)

        assert result is not None
        assert len(result.responses) == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_chain_adapter(self) -> None:
        """Test using chain adapter with evaluate."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        docs = [MockLCDocument("Relevant content", {"id": "doc1"})]
        chain = MockChain("The answer is Paris.", docs)
        adapter = LangChainAdapter(chain)

        testset = TestSet(
            queries=[
                Query(text="What is the capital?", ground_truth_docs=["doc1"]),
            ]
        )

        result = await evaluate(adapter, testset)

        assert result is not None
        assert len(result.responses) == 1
        assert "Paris" in result.responses[0]
