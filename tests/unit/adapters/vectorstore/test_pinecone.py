"""Unit tests for Pinecone vector store adapter."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ragnarok_ai.adapters.vectorstore.pinecone import PineconeVectorStore
from ragnarok_ai.core.exceptions import VectorStoreConnectionError
from ragnarok_ai.core.types import Document

# ============================================================================
# Initialization Tests
# ============================================================================


class TestPineconeVectorStoreInit:
    """Tests for PineconeVectorStore initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        store = PineconeVectorStore(api_key="test_key", index_name="test-index")
        assert store.api_key == "test_key"
        assert store.index_name == "test-index"
        assert store.namespace == ""
        assert store.is_local is False

    def test_init_with_env_var(self) -> None:
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "env_key"}):
            store = PineconeVectorStore(index_name="test-index")
            assert store.api_key == "env_key"

    def test_init_without_api_key_raises(self) -> None:
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PINECONE_API_KEY", None)
            with pytest.raises(ValueError, match="API key required"):
                PineconeVectorStore(index_name="test-index")

    def test_init_without_index_name_raises(self) -> None:
        """Test that initialization without index_name raises ValueError."""
        with pytest.raises(ValueError, match="index_name is required"):
            PineconeVectorStore(api_key="test_key", index_name=None)

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        store = PineconeVectorStore(
            api_key="test_key",
            index_name="custom-index",
            namespace="custom-ns",
            timeout=120.0,
        )
        assert store.index_name == "custom-index"
        assert store.namespace == "custom-ns"
        assert store.timeout == 120.0


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestPineconeVectorStoreProtocol:
    """Tests for protocol compliance."""

    def test_is_local_false(self) -> None:
        """Test that is_local is False for cloud adapter."""
        store = PineconeVectorStore(api_key="test_key", index_name="test")
        assert store.is_local is False


# ============================================================================
# Search Tests
# ============================================================================


class TestPineconeVectorStoreSearch:
    """Tests for vector search."""

    @pytest.mark.asyncio
    async def test_search_success(self) -> None:
        """Test successful vector search."""
        store = PineconeVectorStore(api_key="test_key", index_name="test")

        mock_index = MagicMock()
        mock_index.query.return_value = {
            "matches": [
                {"id": "doc1", "score": 0.95, "metadata": {"content": "Test content 1"}},
                {"id": "doc2", "score": 0.85, "metadata": {"content": "Test content 2"}},
            ]
        }

        with patch.object(store, "_ensure_client", return_value=mock_index):
            results = await store.search([0.1, 0.2, 0.3], k=2)

        assert len(results) == 2
        assert results[0][0].id == "doc1"
        assert results[0][0].content == "Test content 1"
        assert results[0][1] == 0.95

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        """Test search with no results."""
        store = PineconeVectorStore(api_key="test_key", index_name="test")

        mock_index = MagicMock()
        mock_index.query.return_value = {"matches": []}

        with patch.object(store, "_ensure_client", return_value=mock_index):
            results = await store.search([0.1, 0.2, 0.3], k=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_connection_error(self) -> None:
        """Test search with connection error."""
        store = PineconeVectorStore(api_key="test_key", index_name="test")

        with (
            patch.object(store, "_ensure_client", side_effect=VectorStoreConnectionError("Connection failed")),
            pytest.raises(VectorStoreConnectionError, match="Connection failed"),
        ):
            await store.search([0.1, 0.2, 0.3], k=10)


# ============================================================================
# Add Tests
# ============================================================================


class TestPineconeVectorStoreAdd:
    """Tests for adding documents."""

    @pytest.mark.asyncio
    async def test_add_success(self) -> None:
        """Test successful document addition."""
        store = PineconeVectorStore(api_key="test_key", index_name="test")

        mock_index = MagicMock()

        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"embedding": [0.1, 0.2, 0.3]},
        )

        with patch.object(store, "_ensure_client", return_value=mock_index):
            await store.add([doc])

        mock_index.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_empty_list(self) -> None:
        """Test adding empty list does nothing."""
        store = PineconeVectorStore(api_key="test_key", index_name="test")

        mock_index = MagicMock()

        with patch.object(store, "_ensure_client", return_value=mock_index):
            await store.add([])

        mock_index.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_missing_embedding_raises(self) -> None:
        """Test that adding document without embedding raises ValueError."""
        store = PineconeVectorStore(api_key="test_key", index_name="test")

        doc = Document(id="doc1", content="Test content", metadata={})

        with pytest.raises(ValueError, match="missing 'embedding'"):
            await store.add([doc])


# ============================================================================
# Delete Tests
# ============================================================================


class TestPineconeVectorStoreDelete:
    """Tests for deleting documents."""

    @pytest.mark.asyncio
    async def test_delete_success(self) -> None:
        """Test successful document deletion."""
        store = PineconeVectorStore(api_key="test_key", index_name="test")

        mock_index = MagicMock()

        with patch.object(store, "_ensure_client", return_value=mock_index):
            await store.delete(["doc1", "doc2"])

        mock_index.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_empty_list(self) -> None:
        """Test deleting empty list does nothing."""
        store = PineconeVectorStore(api_key="test_key", index_name="test")

        mock_index = MagicMock()

        with patch.object(store, "_ensure_client", return_value=mock_index):
            await store.delete([])

        mock_index.delete.assert_not_called()


# ============================================================================
# is_available Tests
# ============================================================================


class TestPineconeVectorStoreIsAvailable:
    """Tests for availability check."""

    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        """Test availability when Pinecone is reachable."""
        store = PineconeVectorStore(api_key="test_key", index_name="test")

        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = {}

        with patch.object(store, "_ensure_client", return_value=mock_index):
            result = await store.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_false_on_error(self) -> None:
        """Test availability when connection fails."""
        store = PineconeVectorStore(api_key="test_key", index_name="test")

        with patch.object(store, "_ensure_client", side_effect=Exception("Connection failed")):
            result = await store.is_available()

        assert result is False


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestPineconeVectorStoreContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test that context manager properly initializes and cleans up."""
        store = PineconeVectorStore(api_key="test_key", index_name="test")

        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = {"namespaces": {}}

        async def mock_ensure():
            store._index = mock_index
            return mock_index

        with patch.object(store, "_ensure_client", side_effect=mock_ensure):
            async with store as s:
                assert s._index is not None

        # After exit, should be cleaned up
        assert store._index is None
        assert store._client is None
