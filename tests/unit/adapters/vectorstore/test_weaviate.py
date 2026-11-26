"""Unit tests for Weaviate vector store adapter."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ragnarok_ai.adapters.vectorstore.weaviate import WeaviateVectorStore, is_cloud_url, parse_endpoint
from ragnarok_ai.core.exceptions import VectorStoreConnectionError
from ragnarok_ai.core.types import Document

# ============================================================================
# Initialization Tests
# ============================================================================


class TestEndpointFromUrl:
    """Host, port and TLS read from the URL: every form of scheme, port and path (#201)."""

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            # http, without and with a port
            ("http://weaviate.internal", ("weaviate.internal", 8080, False)),
            ("http://weaviate.internal:8081", ("weaviate.internal", 8081, False)),
            ("http://weaviate.internal:80", ("weaviate.internal", 80, False)),
            # https, without and with a port: 443 by default, not Weaviate's 8080
            ("https://weaviate.example.org", ("weaviate.example.org", 443, True)),
            ("https://weaviate.example.org:8443", ("weaviate.example.org", 8443, True)),
            ("https://weaviate.example.org:443", ("weaviate.example.org", 443, True)),
            # a trailing slash is not a path
            ("http://weaviate.internal/", ("weaviate.internal", 8080, False)),
            ("https://weaviate.example.org:8443/", ("weaviate.example.org", 8443, True)),
            # no scheme: read as http
            ("localhost:8080", ("localhost", 8080, False)),
            ("localhost", ("localhost", 8080, False)),
            # addresses
            ("http://127.0.0.1:9000", ("127.0.0.1", 9000, False)),
            ("http://[::1]:8080", ("::1", 8080, False)),
            ("HTTPS://Weaviate.Example.org", ("weaviate.example.org", 443, True)),
        ],
    )
    def test_scheme_and_port(self, url: str, expected: tuple[str, int, bool]) -> None:
        assert parse_endpoint(url) == expected

    @pytest.mark.parametrize(
        ("url", "shown"),
        [
            ("http://weaviate.internal/v1", "http://weaviate.internal"),
            ("http://weaviate.internal:8081/v1", "http://weaviate.internal:8081"),
            ("https://weaviate.example.org/weaviate", "https://weaviate.example.org"),
            ("https://weaviate.example.org:8443/v1/graphql", "https://weaviate.example.org:8443"),
            ("https://weaviate.example.org:8443/?tenant=a", "https://weaviate.example.org:8443"),
            ("localhost:8080/v1", "http://localhost:8080"),
        ],
    )
    def test_a_url_with_a_path_is_refused_and_the_usable_form_is_given(self, url: str, shown: str) -> None:
        with pytest.raises(ValueError, match="has a path") as raised:
            parse_endpoint(url)
        assert f"give the URL as {shown}" in str(raised.value) and "has no base path" in str(raised.value)
        with pytest.raises(ValueError, match="has a path"):
            WeaviateVectorStore(url=url, api_key="k")  # at construction, not at the first call
        with pytest.raises(ValueError, match="has a path"):
            WeaviateVectorStore(url=url)

    @pytest.mark.parametrize(
        ("url", "message"),
        [
            ("ftp://weaviate.internal", "scheme must be http or https, got 'ftp'"),
            ("http://", "has no host"),
            ("http://weaviate.internal:port", "the port is not a number"),
            ("http://weaviate.internal:99999", "the port is not a number"),
        ],
    )
    def test_other_unusable_urls(self, url: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_endpoint(url)

    def test_a_cloud_url_is_passed_as_it_is(self) -> None:
        assert is_cloud_url("https://my-cluster.weaviate.cloud") and is_cloud_url("https://x.wcs.api.weaviate.io")
        assert not is_cloud_url("https://weaviate.example.org")
        store = WeaviateVectorStore(url="https://my-cluster.abc.weaviate.cloud/some/path", api_key="k")
        assert store.url == "https://my-cluster.abc.weaviate.cloud/some/path"  # not parsed, not refused


class TestWeaviateVectorStoreInit:
    """Tests for WeaviateVectorStore initialization."""

    def test_grpc_settings_default_to_the_url_and_port_50051(self) -> None:
        store = WeaviateVectorStore(url="https://weaviate.example.org:8443", api_key="k")
        assert (store.grpc_host, store.grpc_port, store.grpc_secure) == (None, 50051, None)
        explicit = WeaviateVectorStore(api_key="k", grpc_host="grpc.example.org", grpc_port=50443, grpc_secure=True)
        assert (explicit.grpc_host, explicit.grpc_port, explicit.grpc_secure) == ("grpc.example.org", 50443, True)

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        store = WeaviateVectorStore()
        assert store.url == "http://localhost:8080"
        assert store.collection_name == "RagnarokDocuments"
        assert store.is_local is False

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        store = WeaviateVectorStore(
            url="https://test.weaviate.cloud",
            api_key="test_key",
        )
        assert store.api_key == "test_key"
        assert store.url == "https://test.weaviate.cloud"

    def test_init_with_env_var(self) -> None:
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"WEAVIATE_API_KEY": "env_key"}):
            store = WeaviateVectorStore()
            assert store.api_key == "env_key"

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        store = WeaviateVectorStore(
            url="https://custom.weaviate.cloud",
            api_key="test_key",
            collection_name="CustomCollection",
            timeout=120.0,
        )
        assert store.url == "https://custom.weaviate.cloud"
        assert store.collection_name == "CustomCollection"
        assert store.timeout == 120.0

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from url."""
        store = WeaviateVectorStore(url="http://localhost:8080/")
        assert store.url == "http://localhost:8080"


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestWeaviateVectorStoreProtocol:
    """Tests for protocol compliance."""

    def test_is_local_false(self) -> None:
        """Test that is_local is False for cloud adapter."""
        store = WeaviateVectorStore()
        assert store.is_local is False


# ============================================================================
# Search Tests
# ============================================================================


class TestWeaviateVectorStoreSearch:
    """Tests for vector search."""

    @pytest.mark.asyncio
    async def test_search_success(self) -> None:
        """Test successful vector search."""
        store = WeaviateVectorStore()

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        mock_obj1 = MagicMock()
        mock_obj1.properties = {"content": "Test content 1", "doc_id": "doc1"}
        mock_obj1.metadata.distance = 0.1
        mock_obj1.uuid = "uuid1"

        mock_obj2 = MagicMock()
        mock_obj2.properties = {"content": "Test content 2", "doc_id": "doc2"}
        mock_obj2.metadata.distance = 0.2
        mock_obj2.uuid = "uuid2"

        mock_results = MagicMock()
        mock_results.objects = [mock_obj1, mock_obj2]
        mock_collection.query.near_vector.return_value = mock_results

        with patch.object(store, "_ensure_client", return_value=mock_client):
            results = await store.search([0.1, 0.2, 0.3], k=2)

        assert len(results) == 2
        assert results[0][0].id == "doc1"
        assert results[0][0].content == "Test content 1"
        assert results[0][1] == pytest.approx(0.9, rel=0.01)  # 1 - 0.1

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        """Test search with no results."""
        store = WeaviateVectorStore()

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        mock_results = MagicMock()
        mock_results.objects = []
        mock_collection.query.near_vector.return_value = mock_results

        with patch.object(store, "_ensure_client", return_value=mock_client):
            results = await store.search([0.1, 0.2, 0.3], k=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_connection_error(self) -> None:
        """Test search with connection error."""
        store = WeaviateVectorStore()

        with (
            patch.object(store, "_ensure_client", side_effect=VectorStoreConnectionError("Connection failed")),
            pytest.raises(VectorStoreConnectionError, match="Connection failed"),
        ):
            await store.search([0.1, 0.2, 0.3], k=10)


# ============================================================================
# Add Tests
# ============================================================================


class TestWeaviateVectorStoreAdd:
    """Tests for adding documents."""

    @pytest.mark.asyncio
    async def test_add_success(self) -> None:
        """Test successful document addition."""
        store = WeaviateVectorStore()

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        mock_batch = MagicMock()
        mock_batch.__enter__ = MagicMock(return_value=mock_batch)
        mock_batch.__exit__ = MagicMock(return_value=None)
        mock_collection.batch.dynamic.return_value = mock_batch

        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"embedding": [0.1, 0.2, 0.3]},
        )

        with patch.object(store, "_ensure_client", return_value=mock_client):
            await store.add([doc])

        mock_batch.add_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_empty_list(self) -> None:
        """Test adding empty list does nothing."""
        store = WeaviateVectorStore()

        mock_client = MagicMock()

        with patch.object(store, "_ensure_client", return_value=mock_client):
            await store.add([])

        mock_client.collections.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_missing_embedding_raises(self) -> None:
        """Test that adding document without embedding raises ValueError."""
        store = WeaviateVectorStore()

        doc = Document(id="doc1", content="Test content", metadata={})

        with pytest.raises(ValueError, match="missing 'embedding'"):
            await store.add([doc])


# ============================================================================
# Delete Tests
# ============================================================================


class TestWeaviateVectorStoreDelete:
    """Tests for deleting documents."""

    @pytest.mark.asyncio
    async def test_delete_success(self) -> None:
        """Test successful document deletion."""
        store = WeaviateVectorStore()

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        # Mock the Filter import inside the delete method
        mock_filter = MagicMock()
        mock_filter.by_property.return_value.equal.return_value = MagicMock()

        with (
            patch.object(store, "_ensure_client", return_value=mock_client),
            patch.dict("sys.modules", {"weaviate.classes.query": MagicMock(Filter=mock_filter)}),
        ):
            await store.delete(["doc1", "doc2"])

        assert mock_collection.data.delete_many.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_empty_list(self) -> None:
        """Test deleting empty list does nothing."""
        store = WeaviateVectorStore()

        mock_client = MagicMock()

        with patch.object(store, "_ensure_client", return_value=mock_client):
            await store.delete([])

        mock_client.collections.get.assert_not_called()


# ============================================================================
# is_available Tests
# ============================================================================


class TestWeaviateVectorStoreIsAvailable:
    """Tests for availability check."""

    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        """Test availability when Weaviate is reachable."""
        store = WeaviateVectorStore()

        mock_client = MagicMock()
        mock_client.is_ready.return_value = True

        with patch.object(store, "_ensure_client", return_value=mock_client):
            result = await store.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_false_on_error(self) -> None:
        """Test availability when connection fails."""
        store = WeaviateVectorStore()

        with patch.object(store, "_ensure_client", side_effect=Exception("Connection failed")):
            result = await store.is_available()

        assert result is False


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestWeaviateVectorStoreContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test that context manager properly initializes and cleans up."""
        store = WeaviateVectorStore()

        mock_client = MagicMock()
        mock_client.is_ready.return_value = True

        async def mock_ensure():
            store._client = mock_client
            return mock_client

        with patch.object(store, "_ensure_client", side_effect=mock_ensure):
            async with store as s:
                assert s is not None
                assert s._client is not None

        # Client close should have been called
        mock_client.close.assert_called_once()
