"""Unit tests for Haystack framework adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ragnarok_ai.adapters.frameworks.haystack import (
    HaystackAdapter,
    HaystackRetrieverAdapter,
)

# ============================================================================
# Initialization Tests
# ============================================================================


class TestHaystackAdapterInit:
    """Tests for HaystackAdapter initialization."""

    def test_init_with_pipeline(self) -> None:
        """Test initialization with a pipeline."""
        mock_pipeline = MagicMock()
        adapter = HaystackAdapter(mock_pipeline)
        assert adapter.pipeline is mock_pipeline
        assert adapter.is_local is True

    def test_init_with_custom_keys(self) -> None:
        """Test initialization with custom keys."""
        mock_pipeline = MagicMock()
        adapter = HaystackAdapter(
            mock_pipeline,
            query_key="input",
            answer_key="output",
            docs_key="results",
        )
        assert adapter._query_key == "input"
        assert adapter._answer_key == "output"
        assert adapter._docs_key == "results"

    def test_init_with_component_names(self) -> None:
        """Test initialization with specific component names."""
        mock_pipeline = MagicMock()
        adapter = HaystackAdapter(
            mock_pipeline,
            answer_component="generator",
            docs_component="retriever",
        )
        assert adapter._answer_component == "generator"
        assert adapter._docs_component == "retriever"


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestHaystackAdapterProtocol:
    """Tests for protocol compliance."""

    def test_is_local_true(self) -> None:
        """Test that is_local is True."""
        mock_pipeline = MagicMock()
        adapter = HaystackAdapter(mock_pipeline)
        assert adapter.is_local is True


# ============================================================================
# Query Tests
# ============================================================================


class TestHaystackAdapterQuery:
    """Tests for query execution."""

    @pytest.mark.asyncio
    async def test_query_success(self) -> None:
        """Test successful query execution."""
        mock_pipeline = MagicMock()

        # Mock Haystack document
        mock_doc = MagicMock()
        mock_doc.id = "doc1"
        mock_doc.content = "Test content"
        mock_doc.meta = {"source": "test"}

        mock_pipeline.run.return_value = {
            "generator": {"replies": ["This is the answer"]},
            "retriever": {"documents": [mock_doc]},
        }

        adapter = HaystackAdapter(mock_pipeline)

        with patch("asyncio.to_thread", return_value=mock_pipeline.run.return_value):
            response = await adapter.query("What is the answer?")

        assert response.answer == "This is the answer"
        assert len(response.retrieved_docs) == 1
        assert response.retrieved_docs[0].id == "doc1"
        assert response.retrieved_docs[0].content == "Test content"

    @pytest.mark.asyncio
    async def test_query_empty_results(self) -> None:
        """Test query with empty results."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {}

        adapter = HaystackAdapter(mock_pipeline)

        with patch("asyncio.to_thread", return_value=mock_pipeline.run.return_value):
            response = await adapter.query("What is the answer?")

        assert response.answer == ""
        assert response.retrieved_docs == []

    @pytest.mark.asyncio
    async def test_query_with_specific_components(self) -> None:
        """Test query with specific component names."""
        mock_pipeline = MagicMock()

        mock_doc = MagicMock()
        mock_doc.id = "doc1"
        mock_doc.content = "Content"
        mock_doc.meta = {}

        mock_pipeline.run.return_value = {
            "my_generator": {"replies": ["Answer"]},
            "my_retriever": {"documents": [mock_doc]},
        }

        adapter = HaystackAdapter(
            mock_pipeline,
            answer_component="my_generator",
            docs_component="my_retriever",
        )

        with patch("asyncio.to_thread", return_value=mock_pipeline.run.return_value):
            response = await adapter.query("Question?")

        assert response.answer == "Answer"
        assert len(response.retrieved_docs) == 1


# ============================================================================
# Document Conversion Tests
# ============================================================================


class TestHaystackDocumentConversion:
    """Tests for document conversion."""

    @pytest.mark.asyncio
    async def test_convert_haystack_documents(self) -> None:
        """Test conversion of Haystack documents."""
        mock_pipeline = MagicMock()

        mock_doc1 = MagicMock()
        mock_doc1.id = "doc1"
        mock_doc1.content = "First document"
        mock_doc1.meta = {"page": 1}

        mock_doc2 = MagicMock()
        mock_doc2.id = "doc2"
        mock_doc2.content = "Second document"
        mock_doc2.meta = {"page": 2}

        mock_pipeline.run.return_value = {
            "retriever": {"documents": [mock_doc1, mock_doc2]},
        }

        adapter = HaystackAdapter(mock_pipeline)

        with patch("asyncio.to_thread", return_value=mock_pipeline.run.return_value):
            response = await adapter.query("Test query")

        assert len(response.retrieved_docs) == 2
        assert response.retrieved_docs[0].metadata["page"] == 1
        assert response.retrieved_docs[1].metadata["page"] == 2

    @pytest.mark.asyncio
    async def test_convert_dict_documents(self) -> None:
        """Test conversion of dict-format documents."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "retriever": {
                "documents": [
                    {"id": "d1", "content": "Content 1", "meta": {"key": "value"}},
                    {"id": "d2", "content": "Content 2"},
                ]
            },
        }

        adapter = HaystackAdapter(mock_pipeline)

        with patch("asyncio.to_thread", return_value=mock_pipeline.run.return_value):
            response = await adapter.query("Test query")

        assert len(response.retrieved_docs) == 2
        assert response.retrieved_docs[0].id == "d1"


# ============================================================================
# Retriever Adapter Tests
# ============================================================================


class TestHaystackRetrieverAdapter:
    """Tests for HaystackRetrieverAdapter."""

    def test_init(self) -> None:
        """Test initialization."""
        mock_retriever = MagicMock()
        adapter = HaystackRetrieverAdapter(mock_retriever)
        assert adapter.retriever is mock_retriever
        assert adapter.is_local is True

    @pytest.mark.asyncio
    async def test_query_success(self) -> None:
        """Test successful retrieval."""
        mock_retriever = MagicMock()

        mock_doc = MagicMock()
        mock_doc.id = "doc1"
        mock_doc.content = "Test content"
        mock_doc.meta = {}

        mock_retriever.run.return_value = {"documents": [mock_doc]}

        adapter = HaystackRetrieverAdapter(mock_retriever)

        with patch("asyncio.to_thread", return_value=mock_retriever.run.return_value):
            response = await adapter.query("Test query")

        assert len(response.retrieved_docs) == 1
        assert "Retrieved 1 documents" in response.answer

    @pytest.mark.asyncio
    async def test_query_empty(self) -> None:
        """Test retrieval with no results."""
        mock_retriever = MagicMock()
        mock_retriever.run.return_value = {"documents": []}

        adapter = HaystackRetrieverAdapter(mock_retriever)

        with patch("asyncio.to_thread", return_value=mock_retriever.run.return_value):
            response = await adapter.query("Test query")

        assert response.retrieved_docs == []
