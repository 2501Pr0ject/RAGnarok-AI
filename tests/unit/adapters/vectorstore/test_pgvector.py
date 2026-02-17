"""Unit tests for pgvector PostgreSQL vector store adapter."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragnarok_ai.adapters.vectorstore.pgvector import PgvectorVectorStore
from ragnarok_ai.core.exceptions import VectorStoreConnectionError
from ragnarok_ai.core.types import Document

# ============================================================================
# Initialization Tests
# ============================================================================


class TestPgvectorVectorStoreInit:
    """Tests for PgvectorVectorStore initialization."""

    def test_init_with_connection_string(self) -> None:
        """Test initialization with explicit connection string."""
        store = PgvectorVectorStore(connection_string="postgresql://user:pass@localhost:5432/db")
        assert store.connection_string == "postgresql://user:pass@localhost:5432/db"
        assert store.table_name == "ragnarok_documents"
        assert store.vector_size == 768
        assert store.is_local is True

    def test_init_with_env_var(self) -> None:
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://env:pass@localhost/db"}):
            store = PgvectorVectorStore()
            assert store.connection_string == "postgresql://env:pass@localhost/db"

    def test_init_without_connection_string_raises(self) -> None:
        """Test that initialization without connection string raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DATABASE_URL", None)
            with pytest.raises(ValueError, match="Connection string required"):
                PgvectorVectorStore()

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        store = PgvectorVectorStore(
            connection_string="postgresql://user:pass@localhost:5432/db",
            table_name="custom_table",
            vector_size=1024,
            timeout=120.0,
        )
        assert store.table_name == "custom_table"
        assert store.vector_size == 1024
        assert store.timeout == 120.0


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestPgvectorVectorStoreProtocol:
    """Tests for protocol compliance."""

    def test_is_local_true(self) -> None:
        """Test that is_local is True for self-hosted adapter."""
        store = PgvectorVectorStore(connection_string="postgresql://localhost/db")
        assert store.is_local is True


# ============================================================================
# Search Tests
# ============================================================================


class TestPgvectorVectorStoreSearch:
    """Tests for vector search."""

    @pytest.mark.asyncio
    async def test_search_success(self) -> None:
        """Test successful vector search."""
        store = PgvectorVectorStore(
            connection_string="postgresql://localhost/db",
            vector_size=3,
        )

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_conn.fetch = AsyncMock(
            return_value=[
                {"id": "doc1", "content": "Test content 1", "metadata": {}, "similarity": 0.95},
                {"id": "doc2", "content": "Test content 2", "metadata": {}, "similarity": 0.85},
            ]
        )

        with patch.object(store, "_ensure_pool", return_value=mock_pool):
            results = await store.search([0.1, 0.2, 0.3], k=2)

        assert len(results) == 2
        assert results[0][0].id == "doc1"
        assert results[0][0].content == "Test content 1"
        assert results[0][1] == 0.95

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        """Test search with no results."""
        store = PgvectorVectorStore(
            connection_string="postgresql://localhost/db",
            vector_size=3,
        )

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.fetch = AsyncMock(return_value=[])

        with patch.object(store, "_ensure_pool", return_value=mock_pool):
            results = await store.search([0.1, 0.2, 0.3], k=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_wrong_dimension_raises(self) -> None:
        """Test search with wrong embedding dimension raises ValueError."""
        store = PgvectorVectorStore(
            connection_string="postgresql://localhost/db",
            vector_size=768,
        )

        with pytest.raises(ValueError, match="doesn't match vector_size"):
            await store.search([0.1, 0.2, 0.3], k=10)

    @pytest.mark.asyncio
    async def test_search_connection_error(self) -> None:
        """Test search with connection error."""
        store = PgvectorVectorStore(
            connection_string="postgresql://localhost/db",
            vector_size=3,
        )

        with (
            patch.object(store, "_ensure_pool", side_effect=VectorStoreConnectionError("Connection failed")),
            pytest.raises(VectorStoreConnectionError, match="Connection failed"),
        ):
            await store.search([0.1, 0.2, 0.3], k=10)


# ============================================================================
# Add Tests
# ============================================================================


class TestPgvectorVectorStoreAdd:
    """Tests for adding documents."""

    @pytest.mark.asyncio
    async def test_add_success(self) -> None:
        """Test successful document addition."""
        store = PgvectorVectorStore(
            connection_string="postgresql://localhost/db",
            vector_size=3,
        )

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()

        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"embedding": [0.1, 0.2, 0.3]},
        )

        with patch.object(store, "_ensure_pool", return_value=mock_pool):
            await store.add([doc])

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_empty_list(self) -> None:
        """Test adding empty list does nothing."""
        store = PgvectorVectorStore(connection_string="postgresql://localhost/db")

        mock_pool = MagicMock()

        with patch.object(store, "_ensure_pool", return_value=mock_pool):
            await store.add([])

        mock_pool.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_missing_embedding_raises(self) -> None:
        """Test that adding document without embedding raises ValueError."""
        store = PgvectorVectorStore(connection_string="postgresql://localhost/db")

        doc = Document(id="doc1", content="Test content", metadata={})

        with pytest.raises(ValueError, match="missing 'embedding'"):
            await store.add([doc])

    @pytest.mark.asyncio
    async def test_add_wrong_dimension_raises(self) -> None:
        """Test that adding document with wrong dimension raises ValueError."""
        store = PgvectorVectorStore(
            connection_string="postgresql://localhost/db",
            vector_size=768,
        )

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


class TestPgvectorVectorStoreDelete:
    """Tests for deleting documents."""

    @pytest.mark.asyncio
    async def test_delete_success(self) -> None:
        """Test successful document deletion."""
        store = PgvectorVectorStore(connection_string="postgresql://localhost/db")

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()

        with patch.object(store, "_ensure_pool", return_value=mock_pool):
            await store.delete(["doc1", "doc2"])

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_empty_list(self) -> None:
        """Test deleting empty list does nothing."""
        store = PgvectorVectorStore(connection_string="postgresql://localhost/db")

        mock_pool = MagicMock()

        with patch.object(store, "_ensure_pool", return_value=mock_pool):
            await store.delete([])

        mock_pool.acquire.assert_not_called()


# ============================================================================
# Count Tests
# ============================================================================


class TestPgvectorVectorStoreCount:
    """Tests for counting documents."""

    @pytest.mark.asyncio
    async def test_count_success(self) -> None:
        """Test successful count."""
        store = PgvectorVectorStore(connection_string="postgresql://localhost/db")

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.fetchrow = AsyncMock(return_value={"count": 42})

        with patch.object(store, "_ensure_pool", return_value=mock_pool):
            count = await store.count()

        assert count == 42


# ============================================================================
# is_available Tests
# ============================================================================


class TestPgvectorVectorStoreIsAvailable:
    """Tests for availability check."""

    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        """Test availability when PostgreSQL is reachable."""
        store = PgvectorVectorStore(connection_string="postgresql://localhost/db")

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.fetchval = AsyncMock(return_value=1)

        with patch.object(store, "_ensure_pool", return_value=mock_pool):
            result = await store.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_false_on_error(self) -> None:
        """Test availability when connection fails."""
        store = PgvectorVectorStore(connection_string="postgresql://localhost/db")

        with patch.object(store, "_ensure_pool", side_effect=Exception("Connection failed")):
            result = await store.is_available()

        assert result is False


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestPgvectorVectorStoreContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test that context manager properly initializes and cleans up."""
        store = PgvectorVectorStore(connection_string="postgresql://localhost/db")

        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()

        with patch.object(store, "_ensure_pool", return_value=mock_pool):
            async with store as s:
                s._pool = mock_pool
                assert s is not None

            # Pool should be closed
            mock_pool.close.assert_called_once()
