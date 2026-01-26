"""Unit tests for ChromaDB vector store adapter."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from ragnarok_ai.core.protocols import VectorStoreProtocol
from ragnarok_ai.core.types import Document

# ============================================================================
# Mock chromadb module
# ============================================================================


@pytest.fixture
def mock_chromadb():
    """Create a mock chromadb module."""
    mock_module = MagicMock()
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_module.PersistentClient.return_value = mock_client
    mock_module.Client.return_value = mock_client

    with patch.dict(sys.modules, {"chromadb": mock_module}):
        # Clear any cached import
        if "ragnarok_ai.adapters.vectorstore.chroma" in sys.modules:
            del sys.modules["ragnarok_ai.adapters.vectorstore.chroma"]

        from ragnarok_ai.adapters.vectorstore.chroma import ChromaVectorStore

        yield {
            "module": mock_module,
            "client": mock_client,
            "collection": mock_collection,
            "ChromaVectorStore": ChromaVectorStore,
        }


# ============================================================================
# Initialization Tests
# ============================================================================


class TestChromaVectorStoreInit:
    """Tests for ChromaVectorStore initialization."""

    def test_init_defaults(self, mock_chromadb) -> None:
        """Test initialization with default values."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        store = ChromaVectorStore()
        assert store.collection_name == "ragnarok_documents"
        assert store.persist_directory == ".chroma"
        assert store.is_local is True

    def test_init_custom_values(self, mock_chromadb) -> None:
        """Test initialization with custom values."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        store = ChromaVectorStore(
            collection_name="my_docs",
            persist_directory="/data/chroma",
        )
        assert store.collection_name == "my_docs"
        assert store.persist_directory == "/data/chroma"

    def test_init_in_memory(self, mock_chromadb) -> None:
        """Test initialization for in-memory mode."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        store = ChromaVectorStore(persist_directory=None)
        assert store.persist_directory is None


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestChromaVectorStoreProtocol:
    """Tests for VectorStoreProtocol compliance."""

    def test_implements_vectorstore_protocol(self, mock_chromadb) -> None:
        """Test that ChromaVectorStore implements VectorStoreProtocol."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        store = ChromaVectorStore()
        assert isinstance(store, VectorStoreProtocol)

    def test_is_local_true(self, mock_chromadb) -> None:
        """Test that is_local is True for local adapter."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        store = ChromaVectorStore()
        assert store.is_local is True


# ============================================================================
# Search Tests
# ============================================================================


class TestChromaVectorStoreSearch:
    """Tests for similarity search."""

    @pytest.mark.asyncio
    async def test_search_success(self, mock_chromadb) -> None:
        """Test successful search."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        mock_collection = mock_chromadb["collection"]

        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Content 1", "Content 2"]],
            "metadatas": [[{"key": "val1"}, {"key": "val2"}]],
            "distances": [[0.1, 0.3]],
        }

        store = ChromaVectorStore()
        results = await store.search([0.1, 0.2, 0.3], k=5)

        assert len(results) == 2
        assert results[0][0].id == "doc1"
        assert results[0][0].content == "Content 1"
        assert results[0][1] == pytest.approx(0.9)  # 1 - 0.1
        assert results[1][1] == pytest.approx(0.7)  # 1 - 0.3

    @pytest.mark.asyncio
    async def test_search_empty_results(self, mock_chromadb) -> None:
        """Test search with no results."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        mock_collection = mock_chromadb["collection"]

        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        store = ChromaVectorStore()
        results = await store.search([0.1, 0.2, 0.3], k=5)

        assert len(results) == 0


# ============================================================================
# Add Tests
# ============================================================================


class TestChromaVectorStoreAdd:
    """Tests for adding documents."""

    @pytest.mark.asyncio
    async def test_add_success(self, mock_chromadb) -> None:
        """Test successful document addition."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        mock_collection = mock_chromadb["collection"]

        store = ChromaVectorStore()
        docs = [
            Document(id="doc1", content="Content 1", metadata={"embedding": [0.1, 0.2]}),
            Document(id="doc2", content="Content 2", metadata={"embedding": [0.3, 0.4]}),
        ]
        await store.add(docs)

        mock_collection.upsert.assert_called_once()
        call_args = mock_collection.upsert.call_args
        assert call_args.kwargs["ids"] == ["doc1", "doc2"]
        assert call_args.kwargs["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_add_empty_list(self, mock_chromadb) -> None:
        """Test adding empty list does nothing."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        store = ChromaVectorStore()
        await store.add([])
        # Should not raise and client should not be initialized
        assert store._client is None

    @pytest.mark.asyncio
    async def test_add_missing_embedding(self, mock_chromadb) -> None:
        """Test adding document without embedding raises error."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        store = ChromaVectorStore()
        docs = [Document(id="doc1", content="Content", metadata={})]

        with pytest.raises(ValueError, match="missing 'embedding'"):
            await store.add(docs)


# ============================================================================
# Delete Tests
# ============================================================================


class TestChromaVectorStoreDelete:
    """Tests for deleting documents."""

    @pytest.mark.asyncio
    async def test_delete_success(self, mock_chromadb) -> None:
        """Test successful document deletion."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        mock_collection = mock_chromadb["collection"]

        store = ChromaVectorStore()
        await store.delete(["doc1", "doc2"])

        mock_collection.delete.assert_called_once_with(ids=["doc1", "doc2"])

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, mock_chromadb) -> None:
        """Test deleting empty list does nothing."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        store = ChromaVectorStore()
        await store.delete([])
        assert store._client is None


# ============================================================================
# Count Tests
# ============================================================================


class TestChromaVectorStoreCount:
    """Tests for counting documents."""

    @pytest.mark.asyncio
    async def test_count_success(self, mock_chromadb) -> None:
        """Test successful count."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        mock_collection = mock_chromadb["collection"]
        mock_collection.count.return_value = 42

        store = ChromaVectorStore()
        count = await store.count()

        assert count == 42


# ============================================================================
# is_available Tests
# ============================================================================


class TestChromaVectorStoreIsAvailable:
    """Tests for availability check."""

    @pytest.mark.asyncio
    async def test_is_available_true(self, mock_chromadb) -> None:
        """Test availability when ChromaDB is initialized."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]

        store = ChromaVectorStore()
        assert await store.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false_on_error(self, mock_chromadb) -> None:
        """Test availability when initialization fails."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]
        store = ChromaVectorStore()

        with patch.object(store, "_ensure_client", side_effect=Exception("Init failed")):
            assert await store.is_available() is False


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestChromaVectorStoreContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self, mock_chromadb) -> None:
        """Test context manager initializes and cleans up."""
        ChromaVectorStore = mock_chromadb["ChromaVectorStore"]

        async with ChromaVectorStore() as store:
            assert store._client is not None
            assert store._collection is not None

        # After exit, references should be cleared
        assert store._client is None
        assert store._collection is None
