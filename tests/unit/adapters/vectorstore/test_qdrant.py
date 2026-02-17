"""Tests for Qdrant vector store adapter."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock qdrant_client module before any imports that might use it
mock_qdrant_client = MagicMock()
mock_qdrant_client.AsyncQdrantClient = MagicMock()
mock_qdrant_client.http = MagicMock()
mock_qdrant_client.http.models = MagicMock()
mock_qdrant_client.http.models.Distance = MagicMock()
mock_qdrant_client.http.models.Distance.COSINE = "Cosine"
mock_qdrant_client.http.models.VectorParams = MagicMock()
mock_qdrant_client.http.models.PointStruct = MagicMock()
mock_qdrant_client.http.models.PointIdsList = MagicMock()
mock_qdrant_client.http.exceptions = MagicMock()
mock_qdrant_client.http.exceptions.UnexpectedResponse = Exception

sys.modules["qdrant_client"] = mock_qdrant_client
sys.modules["qdrant_client.http"] = mock_qdrant_client.http
sys.modules["qdrant_client.http.models"] = mock_qdrant_client.http.models
sys.modules["qdrant_client.http.exceptions"] = mock_qdrant_client.http.exceptions

from ragnarok_ai.adapters.vectorstore.qdrant import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    DEFAULT_TIMEOUT,
    DEFAULT_URL,
    DEFAULT_VECTOR_SIZE,
    QdrantVectorStore,
)
from ragnarok_ai.core.exceptions import VectorStoreConnectionError  # noqa: E402
from ragnarok_ai.core.types import Document  # noqa: E402


def _make_collection_mock(name: str) -> Any:
    """Create a mock collection with the given name attribute."""
    mock = MagicMock()
    mock.name = name
    return mock


def _make_collections_response(collection_names: list[str]) -> MagicMock:
    """Create a mock get_collections response."""
    collections = [_make_collection_mock(name) for name in collection_names]
    return MagicMock(collections=collections)


class TestQdrantVectorStoreInit:
    """Tests for QdrantVectorStore initialization."""

    def test_default_values(self) -> None:
        """Default values are set correctly."""
        store = QdrantVectorStore()

        assert store.url == DEFAULT_URL
        assert store.api_key is None
        assert store.collection_name == DEFAULT_COLLECTION_NAME
        assert store.vector_size == DEFAULT_VECTOR_SIZE
        assert store.timeout == DEFAULT_TIMEOUT

    def test_custom_values(self) -> None:
        """Custom values are set correctly."""
        store = QdrantVectorStore(
            url="http://custom:6333",
            api_key="secret-key",
            collection_name="my_collection",
            vector_size=1536,
            timeout=60.0,
        )

        assert store.url == "http://custom:6333"
        assert store.api_key == "secret-key"
        assert store.collection_name == "my_collection"
        assert store.vector_size == 1536
        assert store.timeout == 60.0

    def test_trailing_slash_removed(self) -> None:
        """Trailing slash is removed from url."""
        store = QdrantVectorStore(url="http://localhost:6333/")

        assert store.url == "http://localhost:6333"


class TestQdrantVectorStoreAdd:
    """Tests for QdrantVectorStore.add method."""

    @pytest.mark.asyncio
    async def test_add_success(self) -> None:
        """Successful add stores documents."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore(vector_size=4)

        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"embedding": [0.1, 0.2, 0.3, 0.4], "source": "test"},
        )
        await store.add([doc])

        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args.kwargs["collection_name"] == "ragnarok_documents"
        points = call_args.kwargs["points"]
        assert len(points) == 1

    @pytest.mark.asyncio
    async def test_add_empty_list(self) -> None:
        """Adding empty list does nothing."""
        store = QdrantVectorStore()
        # Should not raise and should not try to connect
        await store.add([])
        assert store._client is None

    @pytest.mark.asyncio
    async def test_add_missing_embedding(self) -> None:
        """Missing embedding raises ValueError."""
        store = QdrantVectorStore()

        doc = Document(id="doc1", content="Test content")

        with pytest.raises(ValueError, match="missing 'embedding'"):
            await store.add([doc])

    @pytest.mark.asyncio
    async def test_add_wrong_dimension(self) -> None:
        """Wrong embedding dimension raises ValueError."""
        store = QdrantVectorStore(vector_size=4)

        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"embedding": [0.1, 0.2]},  # Wrong dimension
        )

        with pytest.raises(ValueError, match="doesn't match vector_size"):
            await store.add([doc])

    @pytest.mark.asyncio
    async def test_add_connection_error(self) -> None:
        """Connection error raises VectorStoreConnectionError."""
        mock_qdrant_client.AsyncQdrantClient.side_effect = Exception("Connection refused")

        store = QdrantVectorStore(vector_size=4)

        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"embedding": [0.1, 0.2, 0.3, 0.4]},
        )

        with pytest.raises(VectorStoreConnectionError, match="Failed to connect"):
            await store.add([doc])

        # Reset for other tests
        mock_qdrant_client.AsyncQdrantClient.side_effect = None


class TestQdrantVectorStoreSearch:
    """Tests for QdrantVectorStore.search method."""

    @pytest.mark.asyncio
    async def test_search_success(self) -> None:
        """Successful search returns documents with scores."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_client.search.return_value = [
            MagicMock(
                id="doc1",
                score=0.95,
                payload={"content": "Paris is the capital", "source": "wiki"},
            ),
            MagicMock(
                id="doc2",
                score=0.82,
                payload={"content": "France is a country"},
            ),
        ]
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore(vector_size=4)
        results = await store.search([0.1, 0.2, 0.3, 0.4], k=5)

        assert len(results) == 2
        doc1, score1 = results[0]
        assert doc1.id == "doc1"
        assert doc1.content == "Paris is the capital"
        assert doc1.metadata["source"] == "wiki"
        assert score1 == 0.95

        doc2, score2 = results[1]
        assert doc2.id == "doc2"
        assert doc2.content == "France is a country"
        assert score2 == 0.82

        mock_client.search.assert_called_once_with(
            collection_name="ragnarok_documents",
            query_vector=[0.1, 0.2, 0.3, 0.4],
            limit=5,
        )

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        """Empty search results return empty list."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_client.search.return_value = []
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore(vector_size=4)
        results = await store.search([0.1, 0.2, 0.3, 0.4], k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_wrong_dimension(self) -> None:
        """Wrong query dimension raises ValueError."""
        store = QdrantVectorStore(vector_size=4)

        with pytest.raises(ValueError, match="doesn't match vector_size"):
            await store.search([0.1, 0.2], k=5)  # Wrong dimension


class TestQdrantVectorStoreDelete:
    """Tests for QdrantVectorStore.delete method."""

    @pytest.mark.asyncio
    async def test_delete_success(self) -> None:
        """Successful delete removes documents."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore()
        await store.delete(["doc1", "doc2"])

        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args.kwargs["collection_name"] == "ragnarok_documents"

    @pytest.mark.asyncio
    async def test_delete_empty_list(self) -> None:
        """Deleting empty list does nothing."""
        store = QdrantVectorStore()
        await store.delete([])
        assert store._client is None


class TestQdrantVectorStoreCount:
    """Tests for QdrantVectorStore.count method."""

    @pytest.mark.asyncio
    async def test_count_success(self) -> None:
        """Successful count returns document count."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_client.get_collection.return_value = MagicMock(points_count=42)
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore()
        count = await store.count()

        assert count == 42
        mock_client.get_collection.assert_called_once_with("ragnarok_documents")


class TestQdrantVectorStoreIsAvailable:
    """Tests for QdrantVectorStore.is_available method."""

    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        """Returns True when Qdrant responds successfully."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore()
        result = await store.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_false_on_error(self) -> None:
        """Returns False when connection fails."""
        mock_qdrant_client.AsyncQdrantClient.side_effect = Exception("Connection refused")

        store = QdrantVectorStore()
        result = await store.is_available()

        assert result is False

        # Reset for other tests
        mock_qdrant_client.AsyncQdrantClient.side_effect = None


class TestQdrantVectorStoreProtocolCompliance:
    """Tests for VectorStoreProtocol compliance."""

    def test_implements_protocol(self) -> None:
        """QdrantVectorStore implements VectorStoreProtocol."""
        from ragnarok_ai.core.protocols import VectorStoreProtocol

        store = QdrantVectorStore()

        # Protocol check using isinstance (runtime_checkable)
        assert isinstance(store, VectorStoreProtocol)

    def test_has_search_method(self) -> None:
        """QdrantVectorStore has search method."""
        store = QdrantVectorStore()

        assert hasattr(store, "search")
        assert callable(store.search)

    def test_has_add_method(self) -> None:
        """QdrantVectorStore has add method."""
        store = QdrantVectorStore()

        assert hasattr(store, "add")
        assert callable(store.add)


class TestQdrantVectorStoreContextManager:
    """Tests for QdrantVectorStore context manager functionality."""

    def test_client_is_none_initially(self) -> None:
        """Client is None before entering context manager."""
        store = QdrantVectorStore()

        assert store._client is None

    @pytest.mark.asyncio
    async def test_context_manager_creates_client(self) -> None:
        """Entering context manager creates Qdrant client."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore()

        async with store:
            assert store._client is not None
            assert store._client is mock_client

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self) -> None:
        """Exiting context manager closes Qdrant client."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore()

        async with store:
            pass

        assert store._client is None
        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self) -> None:
        """Context manager returns the QdrantVectorStore instance."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore()

        async with store as entered_store:
            assert entered_store is store


class TestQdrantVectorStoreCollectionCreation:
    """Tests for automatic collection creation."""

    @pytest.mark.asyncio
    async def test_creates_collection_if_not_exists(self) -> None:
        """Collection is created if it doesn't exist."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response([])
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore(vector_size=1536)

        async with store:
            pass

        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "ragnarok_documents"

    @pytest.mark.asyncio
    async def test_skips_creation_if_exists(self) -> None:
        """Collection is not created if it already exists."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore()

        async with store:
            pass

        mock_client.create_collection.assert_not_called()


class TestQdrantVectorStoreErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_ensure_client_already_initialized(self) -> None:
        """Test _ensure_client returns early if client already exists."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore()
        await store._ensure_client()

        # Reset mock
        mock_qdrant_client.AsyncQdrantClient.reset_mock()

        # Call again - should return early
        result = await store._ensure_client()

        mock_qdrant_client.AsyncQdrantClient.assert_not_called()
        assert result is mock_client

    @pytest.mark.asyncio
    async def test_ensure_collection_with_no_client(self) -> None:
        """Test _ensure_collection returns early if client is None."""
        store = QdrantVectorStore()
        store._client = None

        # Should not raise
        await store._ensure_collection()

    @pytest.mark.asyncio
    async def test_unexpected_response_error(self) -> None:
        """Test UnexpectedResponse raises VectorStoreConnectionError."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response([])
        # Simulate UnexpectedResponse during collection creation
        mock_client.create_collection.side_effect = mock_qdrant_client.http.exceptions.UnexpectedResponse("Collection error")
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore()

        with pytest.raises(VectorStoreConnectionError, match="Failed to create collection"):
            await store._ensure_client()

        mock_client.create_collection.side_effect = None

    @pytest.mark.asyncio
    async def test_import_error_raises_connection_error(self) -> None:
        """Test ImportError raises VectorStoreConnectionError."""
        # Temporarily make AsyncQdrantClient raise ImportError-like behavior
        original_side_effect = mock_qdrant_client.AsyncQdrantClient.side_effect
        mock_qdrant_client.AsyncQdrantClient.side_effect = ImportError("No module named 'qdrant_client'")

        store = QdrantVectorStore()

        with pytest.raises(VectorStoreConnectionError, match="qdrant-client is not installed"):
            await store._ensure_client()

        mock_qdrant_client.AsyncQdrantClient.side_effect = original_side_effect

    @pytest.mark.asyncio
    async def test_search_generic_error(self) -> None:
        """Test search raises VectorStoreConnectionError on generic error."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_client.search.side_effect = RuntimeError("Search failed")
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore(vector_size=4)

        with pytest.raises(VectorStoreConnectionError, match="Failed to search"):
            await store.search([0.1, 0.2, 0.3, 0.4], k=5)

        mock_client.search.side_effect = None

    @pytest.mark.asyncio
    async def test_search_reraises_vectorstore_connection_error(self) -> None:
        """Test search re-raises VectorStoreConnectionError."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_client.search.side_effect = VectorStoreConnectionError("Original error")
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore(vector_size=4)

        with pytest.raises(VectorStoreConnectionError, match="Original error"):
            await store.search([0.1, 0.2, 0.3, 0.4], k=5)

        mock_client.search.side_effect = None

    @pytest.mark.asyncio
    async def test_add_generic_error(self) -> None:
        """Test add raises VectorStoreConnectionError on generic error."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_client.upsert.side_effect = RuntimeError("Add failed")
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore(vector_size=4)
        doc = Document(id="doc1", content="Content", metadata={"embedding": [0.1, 0.2, 0.3, 0.4]})

        with pytest.raises(VectorStoreConnectionError, match="Failed to add"):
            await store.add([doc])

        mock_client.upsert.side_effect = None

    @pytest.mark.asyncio
    async def test_add_reraises_vectorstore_connection_error(self) -> None:
        """Test add re-raises VectorStoreConnectionError."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_client.upsert.side_effect = VectorStoreConnectionError("Original error")
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore(vector_size=4)
        doc = Document(id="doc1", content="Content", metadata={"embedding": [0.1, 0.2, 0.3, 0.4]})

        with pytest.raises(VectorStoreConnectionError, match="Original error"):
            await store.add([doc])

        mock_client.upsert.side_effect = None

    @pytest.mark.asyncio
    async def test_delete_generic_error(self) -> None:
        """Test delete raises VectorStoreConnectionError on generic error."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_client.delete.side_effect = RuntimeError("Delete failed")
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore()

        with pytest.raises(VectorStoreConnectionError, match="Failed to delete"):
            await store.delete(["doc1"])

        mock_client.delete.side_effect = None

    @pytest.mark.asyncio
    async def test_delete_reraises_vectorstore_connection_error(self) -> None:
        """Test delete re-raises VectorStoreConnectionError."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_client.delete.side_effect = VectorStoreConnectionError("Original error")
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore()

        with pytest.raises(VectorStoreConnectionError, match="Original error"):
            await store.delete(["doc1"])

        mock_client.delete.side_effect = None

    @pytest.mark.asyncio
    async def test_count_generic_error(self) -> None:
        """Test count raises VectorStoreConnectionError on generic error."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_client.get_collection.side_effect = RuntimeError("Count failed")
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore()

        with pytest.raises(VectorStoreConnectionError, match="Failed to get count"):
            await store.count()

        mock_client.get_collection.side_effect = None

    @pytest.mark.asyncio
    async def test_count_reraises_vectorstore_connection_error(self) -> None:
        """Test count re-raises VectorStoreConnectionError."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_client.get_collection.side_effect = VectorStoreConnectionError("Original error")
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore()

        with pytest.raises(VectorStoreConnectionError, match="Original error"):
            await store.count()

        mock_client.get_collection.side_effect = None


class TestQdrantVectorStoreSearchPayload:
    """Tests for search payload handling."""

    @pytest.mark.asyncio
    async def test_search_with_null_payload(self) -> None:
        """Test search handles null payload correctly."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = _make_collections_response(["ragnarok_documents"])
        mock_client.search.return_value = [
            MagicMock(id="doc1", score=0.95, payload=None),
        ]
        mock_qdrant_client.AsyncQdrantClient.return_value = mock_client

        store = QdrantVectorStore(vector_size=4)
        results = await store.search([0.1, 0.2, 0.3, 0.4], k=5)

        assert len(results) == 1
        doc, score = results[0]
        assert doc.id == "doc1"
        assert doc.content == ""
        assert doc.metadata == {}
        assert score == 0.95
