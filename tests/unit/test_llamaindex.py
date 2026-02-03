"""Unit tests for the LlamaIndex adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ragnarok_ai.adapters.frameworks.llamaindex import (
    LlamaIndexAdapter,
    LlamaIndexQueryEngineAdapter,
    LlamaIndexRetrieverAdapter,
    _convert_node_to_document,
    _convert_nodes_to_documents,
    create_llamaindex_adapter,
)
from ragnarok_ai.core.types import Document, RAGResponse

# ============================================================================
# Mock LlamaIndex Classes
# ============================================================================


class MockNode:
    """Mock LlamaIndex Node."""

    def __init__(
        self,
        content: str,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.node_id = node_id
        self.id_ = node_id
        self.metadata = metadata or {}
        self._content = content

    def get_content(self) -> str:
        return self._content


class MockNodeWithScore:
    """Mock LlamaIndex NodeWithScore."""

    def __init__(self, node: MockNode, score: float = 1.0) -> None:
        self.node = node
        self.score = score


class MockResponse:
    """Mock LlamaIndex Response."""

    def __init__(
        self,
        response: str,
        source_nodes: list[MockNodeWithScore] | None = None,
    ) -> None:
        self._response = response
        self.source_nodes = source_nodes or []

    def __str__(self) -> str:
        return self._response


class MockRetriever:
    """Mock LlamaIndex Retriever (sync)."""

    def __init__(self, nodes: list[MockNodeWithScore]) -> None:
        self._nodes = nodes

    def retrieve(self, _query: str) -> list[MockNodeWithScore]:
        return self._nodes


class MockAsyncRetriever:
    """Mock LlamaIndex Retriever (async)."""

    def __init__(self, nodes: list[MockNodeWithScore]) -> None:
        self._nodes = nodes

    async def aretrieve(self, _query: str) -> list[MockNodeWithScore]:
        return self._nodes


class MockQueryEngine:
    """Mock LlamaIndex QueryEngine (sync)."""

    def __init__(self, response: MockResponse) -> None:
        self._response = response

    def query(self, _query: str) -> MockResponse:
        return self._response


class MockAsyncQueryEngine:
    """Mock LlamaIndex QueryEngine (async)."""

    def __init__(self, response: MockResponse) -> None:
        self._response = response

    async def aquery(self, _query: str) -> MockResponse:
        return self._response


class MockIndex:
    """Mock LlamaIndex Index."""

    def __init__(self, query_engine: MockQueryEngine | MockAsyncQueryEngine) -> None:
        self._query_engine = query_engine

    def as_query_engine(self, **_kwargs: Any) -> MockQueryEngine | MockAsyncQueryEngine:
        return self._query_engine


# ============================================================================
# Node Conversion Tests
# ============================================================================


class TestNodeConversion:
    """Tests for LlamaIndex node conversion."""

    def test_convert_node_basic(self) -> None:
        """Test converting a basic LlamaIndex node."""
        node = MockNode(content="This is the content", node_id="node123")
        node_with_score = MockNodeWithScore(node, score=0.95)
        doc = _convert_node_to_document(node_with_score)

        assert doc.content == "This is the content"
        assert doc.id == "node123"
        assert doc.metadata["score"] == 0.95

    def test_convert_node_with_metadata(self) -> None:
        """Test converting node with metadata."""
        node = MockNode(
            content="Content",
            node_id="doc1",
            metadata={"source": "test.pdf", "page": 5},
        )
        node_with_score = MockNodeWithScore(node, score=0.8)
        doc = _convert_node_to_document(node_with_score)

        assert doc.metadata["source"] == "test.pdf"
        assert doc.metadata["page"] == 5
        assert doc.metadata["score"] == 0.8

    def test_convert_node_no_id(self) -> None:
        """Test converting node without ID."""
        node = MockNode(content="Content only")
        node_with_score = MockNodeWithScore(node, score=0.9)
        doc = _convert_node_to_document(node_with_score)

        assert doc.content == "Content only"
        assert doc.id  # Should have a hash-based ID

    def test_convert_nodes_list(self) -> None:
        """Test converting a list of nodes."""
        nodes = [
            MockNodeWithScore(MockNode("Content 1", "node1"), 0.9),
            MockNodeWithScore(MockNode("Content 2", "node2"), 0.8),
            MockNodeWithScore(MockNode("Content 3", "node3"), 0.7),
        ]
        docs = _convert_nodes_to_documents(nodes)

        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)
        assert [d.id for d in docs] == ["node1", "node2", "node3"]
        assert [d.metadata["score"] for d in docs] == [0.9, 0.8, 0.7]


# ============================================================================
# LlamaIndexRetrieverAdapter Tests
# ============================================================================


class TestLlamaIndexRetrieverAdapter:
    """Tests for LlamaIndexRetrieverAdapter."""

    @pytest.mark.asyncio
    async def test_query_sync_retriever(self) -> None:
        """Test query with sync retriever."""
        nodes = [
            MockNodeWithScore(MockNode("Doc 1 content", "doc1"), 0.95),
            MockNodeWithScore(MockNode("Doc 2 content", "doc2"), 0.85),
        ]
        retriever = MockRetriever(nodes)
        adapter = LlamaIndexRetrieverAdapter(retriever)

        response = await adapter.query("test query")

        assert isinstance(response, RAGResponse)
        assert len(response.retrieved_docs) == 2
        assert response.retrieved_docs[0].id == "doc1"
        assert response.retrieved_docs[1].id == "doc2"
        assert "llamaindex_retriever" in response.metadata["adapter"]

    @pytest.mark.asyncio
    async def test_query_async_retriever(self) -> None:
        """Test query with async retriever."""
        nodes = [
            MockNodeWithScore(MockNode("Async content", "async_doc"), 0.9),
        ]
        retriever = MockAsyncRetriever(nodes)
        adapter = LlamaIndexRetrieverAdapter(retriever)

        response = await adapter.query("test query")

        assert len(response.retrieved_docs) == 1
        assert response.retrieved_docs[0].id == "async_doc"

    @pytest.mark.asyncio
    async def test_query_with_answer_generator(self) -> None:
        """Test query with custom answer generator."""
        nodes = [MockNodeWithScore(MockNode("Content", "doc1"), 0.9)]
        retriever = MockRetriever(nodes)

        def custom_generator(question: str, docs: list[Document]) -> str:
            return f"Answer for {question} using {len(docs)} docs"

        adapter = LlamaIndexRetrieverAdapter(retriever, answer_generator=custom_generator)
        response = await adapter.query("what is RAG?")

        assert response.answer == "Answer for what is RAG? using 1 docs"

    @pytest.mark.asyncio
    async def test_query_empty_results(self) -> None:
        """Test query with no results."""
        retriever = MockRetriever([])
        adapter = LlamaIndexRetrieverAdapter(retriever)

        response = await adapter.query("obscure query")

        assert len(response.retrieved_docs) == 0
        assert "0 documents" in response.answer

    def test_retriever_property(self) -> None:
        """Test retriever property access."""
        retriever = MockRetriever([])
        adapter = LlamaIndexRetrieverAdapter(retriever)

        assert adapter.retriever is retriever


# ============================================================================
# LlamaIndexQueryEngineAdapter Tests
# ============================================================================


class TestLlamaIndexQueryEngineAdapter:
    """Tests for LlamaIndexQueryEngineAdapter."""

    @pytest.mark.asyncio
    async def test_query_sync_engine(self) -> None:
        """Test query with sync query engine."""
        nodes = [
            MockNodeWithScore(MockNode("Source content", "src1"), 0.9),
        ]
        response = MockResponse("The answer is 42", source_nodes=nodes)
        engine = MockQueryEngine(response)
        adapter = LlamaIndexQueryEngineAdapter(engine)

        result = await adapter.query("What is the answer?")

        assert isinstance(result, RAGResponse)
        assert result.answer == "The answer is 42"
        assert len(result.retrieved_docs) == 1
        assert result.retrieved_docs[0].id == "src1"
        assert "llamaindex_query_engine" in result.metadata["adapter"]

    @pytest.mark.asyncio
    async def test_query_async_engine(self) -> None:
        """Test query with async query engine."""
        nodes = [MockNodeWithScore(MockNode("Async source", "async1"), 0.85)]
        response = MockResponse("Async answer", source_nodes=nodes)
        engine = MockAsyncQueryEngine(response)
        adapter = LlamaIndexQueryEngineAdapter(engine)

        result = await adapter.query("Async question?")

        assert result.answer == "Async answer"
        assert len(result.retrieved_docs) == 1

    @pytest.mark.asyncio
    async def test_query_no_source_nodes(self) -> None:
        """Test query with no source nodes."""
        response = MockResponse("Answer without sources")
        engine = MockQueryEngine(response)
        adapter = LlamaIndexQueryEngineAdapter(engine)

        result = await adapter.query("Simple question")

        assert result.answer == "Answer without sources"
        assert len(result.retrieved_docs) == 0

    def test_query_engine_property(self) -> None:
        """Test query_engine property access."""
        response = MockResponse("Test")
        engine = MockQueryEngine(response)
        adapter = LlamaIndexQueryEngineAdapter(engine)

        assert adapter.query_engine is engine


# ============================================================================
# LlamaIndexAdapter Tests
# ============================================================================


class TestLlamaIndexAdapter:
    """Tests for LlamaIndexAdapter."""

    @pytest.mark.asyncio
    async def test_query_from_index(self) -> None:
        """Test query using index adapter."""
        nodes = [MockNodeWithScore(MockNode("Index content", "idx1"), 0.92)]
        response = MockResponse("Index answer", source_nodes=nodes)
        engine = MockQueryEngine(response)
        index = MockIndex(engine)
        adapter = LlamaIndexAdapter(index)

        result = await adapter.query("Index question?")

        assert isinstance(result, RAGResponse)
        assert result.answer == "Index answer"
        assert len(result.retrieved_docs) == 1
        assert "llamaindex" in result.metadata["adapter"]

    @pytest.mark.asyncio
    async def test_query_with_async_engine(self) -> None:
        """Test query with index that returns async engine."""
        nodes = [MockNodeWithScore(MockNode("Async idx content", "aidx1"), 0.88)]
        response = MockResponse("Async idx answer", source_nodes=nodes)
        engine = MockAsyncQueryEngine(response)
        index = MockIndex(engine)
        adapter = LlamaIndexAdapter(index)

        result = await adapter.query("Async index question?")

        assert result.answer == "Async idx answer"

    def test_index_property(self) -> None:
        """Test index property access."""
        response = MockResponse("Test")
        engine = MockQueryEngine(response)
        index = MockIndex(engine)
        adapter = LlamaIndexAdapter(index)

        assert adapter.index is index

    def test_query_engine_property(self) -> None:
        """Test query_engine property access."""
        response = MockResponse("Test")
        engine = MockQueryEngine(response)
        index = MockIndex(engine)
        adapter = LlamaIndexAdapter(index)

        assert adapter.query_engine is engine


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateLlamaIndexAdapter:
    """Tests for create_llamaindex_adapter factory function."""

    def test_create_from_retriever(self) -> None:
        """Test factory raises TypeError for non-LlamaIndex types."""
        retriever = MockRetriever([])

        # MockRetriever is not a real LlamaIndex type, so TypeError is raised
        with pytest.raises(TypeError, match="Unsupported LlamaIndex object type"):
            create_llamaindex_adapter(retriever)

    def test_create_requires_llama_index(self) -> None:
        """Test that factory raises TypeError for unknown types."""
        # MagicMock is not a LlamaIndex type, so TypeError is raised
        with pytest.raises(TypeError, match="Unsupported LlamaIndex object type"):
            create_llamaindex_adapter(MagicMock())


# ============================================================================
# Integration Tests
# ============================================================================


class TestLlamaIndexAdapterIntegration:
    """Integration tests for LlamaIndex adapters."""

    @pytest.mark.asyncio
    async def test_retriever_to_query_pipeline(self) -> None:
        """Test using retriever adapter in a pipeline."""
        nodes = [
            MockNodeWithScore(MockNode("First document about RAG", "doc1"), 0.95),
            MockNodeWithScore(MockNode("Second document about LLMs", "doc2"), 0.85),
            MockNodeWithScore(MockNode("Third document about vectors", "doc3"), 0.75),
        ]
        retriever = MockRetriever(nodes)
        adapter = LlamaIndexRetrieverAdapter(retriever)

        response = await adapter.query("What is RAG?")

        assert len(response.retrieved_docs) == 3
        # Verify scores are preserved in metadata
        assert response.retrieved_docs[0].metadata["score"] == 0.95
        assert response.retrieved_docs[1].metadata["score"] == 0.85
        assert response.retrieved_docs[2].metadata["score"] == 0.75

    @pytest.mark.asyncio
    async def test_full_rag_pipeline(self) -> None:
        """Test full RAG pipeline simulation."""
        # Simulate a complete RAG flow
        source_nodes = [
            MockNodeWithScore(MockNode("Paris is the capital of France.", "wiki_france"), 0.98),
            MockNodeWithScore(MockNode("France is a country in Europe.", "wiki_europe"), 0.92),
        ]
        response = MockResponse(
            "Paris is the capital of France. It is located in Europe.",
            source_nodes=source_nodes,
        )
        engine = MockQueryEngine(response)
        index = MockIndex(engine)
        adapter = LlamaIndexAdapter(index)

        result = await adapter.query("What is the capital of France?")

        assert "Paris" in result.answer
        assert len(result.retrieved_docs) == 2
        assert result.retrieved_docs[0].content == "Paris is the capital of France."
