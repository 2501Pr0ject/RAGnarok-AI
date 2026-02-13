"""Unit tests for Milvus vector store adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ragnarok_ai.adapters.vectorstore.milvus import MilvusVectorStore
from ragnarok_ai.core.exceptions import VectorStoreConnectionError
from ragnarok_ai.core.types import Document

# ============================================================================
# Initialization Tests
# ============================================================================


class TestMilvusVectorStoreInit:
    """Tests for MilvusVectorStore initialization."""

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        store = MilvusVectorStore()
        assert store.host == "localhost"
        assert store.port == 19530
        assert store.collection_name == "ragnarok_documents"
        assert store.vector_size == 768
        assert store.is_local is True

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        store = MilvusVectorStore(
            host="milvus.example.com",
            port=19531,
            collection_name="custom_collection",
            vector_size=1024,
            timeout=120.0,
        )
        assert store.host == "milvus.example.com"
        assert store.port == 19531
        assert store.collection_name == "custom_collection"
        assert store.vector_size == 1024
        assert store.timeout == 120.0


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestMilvusVectorStoreProtocol:
    """Tests for protocol compliance."""

    def test_is_local_true(self) -> None:
        """Test that is_local is True for self-hosted adapter."""
        store = MilvusVectorStore()
        assert store.is_local is True


# ============================================================================
# Search Tests
# ============================================================================


class TestMilvusVectorStoreSearch:
    """Tests for vector search."""

    @pytest.mark.asyncio
    async def test_search_success(self) -> None:
        """Test successful vector search."""
        store = MilvusVectorStore(vector_size=3)

        mock_collection = MagicMock()
        mock_hit1 = MagicMock()
        mock_hit1.entity.get = lambda x, default=None: {"id": "doc1", "content": "Test content 1", "metadata": {}}.get(
            x, default
        )
        mock_hit1.distance = 0.1

        mock_hit2 = MagicMock()
        mock_hit2.entity.get = lambda x, default=None: {"id": "doc2", "content": "Test content 2", "metadata": {}}.get(
            x, default
        )
        mock_hit2.distance = 0.2

        mock_collection.search.return_value = [[mock_hit1, mock_hit2]]

        with patch.object(store, "_ensure_connection", return_value=mock_collection):
            results = await store.search([0.1, 0.2, 0.3], k=2)

        assert len(results) == 2
        assert results[0][0].id == "doc1"
        assert results[0][0].content == "Test content 1"

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        """Test search with no results."""
        store = MilvusVectorStore(vector_size=3)

        mock_collection = MagicMock()
        mock_collection.search.return_value = [[]]

        with patch.object(store, "_ensure_connection", return_value=mock_collection):
            results = await store.search([0.1, 0.2, 0.3], k=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_wrong_dimension_raises(self) -> None:
        """Test search with wrong embedding dimension raises ValueError."""
        store = MilvusVectorStore(vector_size=768)

        with pytest.raises(ValueError, match="doesn't match vector_size"):
            await store.search([0.1, 0.2, 0.3], k=10)

    @pytest.mark.asyncio
    async def test_search_connection_error(self) -> None:
        """Test search with connection error."""
        store = MilvusVectorStore(vector_size=3)

        with (
            patch.object(store, "_ensure_connection", side_effect=VectorStoreConnectionError("Connection failed")),
            pytest.raises(VectorStoreConnectionError, match="Connection failed"),
        ):
            await store.search([0.1, 0.2, 0.3], k=10)


# ============================================================================
# Add Tests
# ============================================================================


class TestMilvusVectorStoreAdd:
    """Tests for adding documents."""

    @pytest.mark.asyncio
    async def test_add_success(self) -> None:
        """Test successful document addition."""
        store = MilvusVectorStore(vector_size=3)

        mock_collection = MagicMock()

        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"embedding": [0.1, 0.2, 0.3]},
        )

        with patch.object(store, "_ensure_connection", return_value=mock_collection):
            await store.add([doc])

        mock_collection.insert.assert_called_once()
        mock_collection.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_empty_list(self) -> None:
        """Test adding empty list does nothing."""
        store = MilvusVectorStore()

        mock_collection = MagicMock()

        with patch.object(store, "_ensure_connection", return_value=mock_collection):
            await store.add([])

        mock_collection.insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_missing_embedding_raises(self) -> None:
        """Test that adding document without embedding raises ValueError."""
        store = MilvusVectorStore()

        doc = Document(id="doc1", content="Test content", metadata={})

        with pytest.raises(ValueError, match="missing 'embedding'"):
            await store.add([doc])

    @pytest.mark.asyncio
    async def test_add_wrong_dimension_raises(self) -> None:
        """Test that adding document with wrong dimension raises ValueError."""
        store = MilvusVectorStore(vector_size=768)

        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"embedding": [0.1, 0.2, 0.3]},
        )

        with pytest.raises(ValueError, match="doesn't match vector_size"):
            await store.add([doc])


# ============================================================================
# Delete Tests
# ============================================================================


class TestMilvusVectorStoreDelete:
    """Tests for deleting documents."""

    @pytest.mark.asyncio
    async def test_delete_success(self) -> None:
        """Test successful document deletion."""
        store = MilvusVectorStore()

        mock_collection = MagicMock()

        with patch.object(store, "_ensure_connection", return_value=mock_collection):
            await store.delete(["doc1", "doc2"])

        mock_collection.delete.assert_called_once()
        mock_collection.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_empty_list(self) -> None:
        """Test deleting empty list does nothing."""
        store = MilvusVectorStore()

        mock_collection = MagicMock()

        with patch.object(store, "_ensure_connection", return_value=mock_collection):
            await store.delete([])

        mock_collection.delete.assert_not_called()


# ============================================================================
# Count Tests
# ============================================================================


class TestMilvusVectorStoreCount:
    """Tests for counting documents."""

    @pytest.mark.asyncio
    async def test_count_success(self) -> None:
        """Test successful count."""
        store = MilvusVectorStore()

        mock_collection = MagicMock()
        mock_collection.num_entities = 42

        with patch.object(store, "_ensure_connection", return_value=mock_collection):
            count = await store.count()

        assert count == 42


# ============================================================================
# is_available Tests
# ============================================================================


class TestMilvusVectorStoreIsAvailable:
    """Tests for availability check."""

    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        """Test availability when Milvus is reachable."""
        store = MilvusVectorStore()

        mock_collection = MagicMock()
        store._connected = True
        store._collection = mock_collection

        with patch.object(store, "_ensure_connection", return_value=mock_collection):
            result = await store.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_false_on_error(self) -> None:
        """Test availability when connection fails."""
        store = MilvusVectorStore()

        with patch.object(store, "_ensure_connection", side_effect=Exception("Connection failed")):
            result = await store.is_available()

        assert result is False


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestMilvusVectorStoreContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test that context manager properly initializes and cleans up."""
        store = MilvusVectorStore()

        mock_collection = MagicMock()
        mock_connections = MagicMock()

        async def mock_ensure():
            store._collection = mock_collection
            store._connected = True
            return mock_collection

        with (
            patch.object(store, "_ensure_connection", side_effect=mock_ensure),
            patch.dict("sys.modules", {"pymilvus": MagicMock(connections=mock_connections)}),
        ):
            async with store as s:
                assert s is not None
                assert s._connected is True

            # Should have disconnected
            mock_connections.disconnect.assert_called_once_with("default")
