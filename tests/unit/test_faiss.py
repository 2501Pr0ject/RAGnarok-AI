"""Unit tests for FAISS vector store adapter."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from ragnarok_ai.core.protocols import VectorStoreProtocol
from ragnarok_ai.core.types import Document

# ============================================================================
# Mock faiss and numpy modules
# ============================================================================


@pytest.fixture
def mock_faiss():
    """Create mock faiss and numpy modules."""
    mock_faiss_module = MagicMock()
    mock_np_module = MagicMock()

    # Mock index
    mock_index = MagicMock()
    mock_index.ntotal = 0
    mock_index.search.return_value = (
        MagicMock(return_value=[[0, 1]]),
        MagicMock(return_value=[[0.9, 0.8]]),
    )

    mock_faiss_module.IndexFlatIP.return_value = mock_index
    mock_faiss_module.normalize_L2 = MagicMock()
    mock_faiss_module.read_index = MagicMock(return_value=mock_index)
    mock_faiss_module.write_index = MagicMock()

    # Mock numpy array
    mock_np_module.array.return_value = MagicMock()
    mock_np_module.float32 = "float32"

    with patch.dict(sys.modules, {"faiss": mock_faiss_module, "numpy": mock_np_module}):
        if "ragnarok_ai.adapters.vectorstore.faiss" in sys.modules:
            del sys.modules["ragnarok_ai.adapters.vectorstore.faiss"]

        from ragnarok_ai.adapters.vectorstore.faiss import FAISSVectorStore

        yield {
            "faiss": mock_faiss_module,
            "numpy": mock_np_module,
            "index": mock_index,
            "FAISSVectorStore": FAISSVectorStore,
        }


# ============================================================================
# Initialization Tests
# ============================================================================


class TestFAISSVectorStoreInit:
    """Tests for FAISSVectorStore initialization."""

    def test_init_defaults(self, mock_faiss) -> None:
        """Test initialization with default values."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        store = FAISSVectorStore()
        assert store.dimension == 768
        assert store.index_path is None
        assert store.is_local is True

    def test_init_custom_values(self, mock_faiss) -> None:
        """Test initialization with custom values."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        store = FAISSVectorStore(dimension=1536, index_path="/data/faiss.index")
        assert store.dimension == 1536
        assert store.index_path == "/data/faiss.index"


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestFAISSVectorStoreProtocol:
    """Tests for VectorStoreProtocol compliance."""

    def test_implements_vectorstore_protocol(self, mock_faiss) -> None:
        """Test that FAISSVectorStore implements VectorStoreProtocol."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        store = FAISSVectorStore()
        assert isinstance(store, VectorStoreProtocol)

    def test_is_local_true(self, mock_faiss) -> None:
        """Test that is_local is True for local adapter."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        store = FAISSVectorStore()
        assert store.is_local is True


# ============================================================================
# Search Tests
# ============================================================================


class TestFAISSVectorStoreSearch:
    """Tests for similarity search."""

    @pytest.mark.asyncio
    async def test_search_empty_index(self, mock_faiss) -> None:
        """Test search on empty index."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        mock_index = mock_faiss["index"]
        mock_index.ntotal = 0

        store = FAISSVectorStore(dimension=3)
        results = await store.search([0.1, 0.2, 0.3], k=5)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_wrong_dimension(self, mock_faiss) -> None:
        """Test search with wrong embedding dimension."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]

        store = FAISSVectorStore(dimension=768)
        with pytest.raises(ValueError, match="dimension"):
            await store.search([0.1, 0.2], k=5)


# ============================================================================
# Add Tests
# ============================================================================


class TestFAISSVectorStoreAdd:
    """Tests for adding documents."""

    @pytest.mark.asyncio
    async def test_add_success(self, mock_faiss) -> None:
        """Test successful document addition."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        mock_index = mock_faiss["index"]

        store = FAISSVectorStore(dimension=2)
        docs = [
            Document(id="doc1", content="Content 1", metadata={"embedding": [0.1, 0.2]}),
            Document(id="doc2", content="Content 2", metadata={"embedding": [0.3, 0.4]}),
        ]
        await store.add(docs)

        mock_index.add.assert_called_once()
        assert store._documents[0].id == "doc1"
        assert store._documents[1].id == "doc2"

    @pytest.mark.asyncio
    async def test_add_empty_list(self, mock_faiss) -> None:
        """Test adding empty list does nothing."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        store = FAISSVectorStore()
        await store.add([])
        assert store._index is None

    @pytest.mark.asyncio
    async def test_add_missing_embedding(self, mock_faiss) -> None:
        """Test adding document without embedding raises error."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        store = FAISSVectorStore()
        docs = [Document(id="doc1", content="Content", metadata={})]

        with pytest.raises(ValueError, match="missing 'embedding'"):
            await store.add(docs)

    @pytest.mark.asyncio
    async def test_add_wrong_dimension(self, mock_faiss) -> None:
        """Test adding document with wrong embedding dimension."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        store = FAISSVectorStore(dimension=768)
        docs = [Document(id="doc1", content="Content", metadata={"embedding": [0.1, 0.2]})]

        with pytest.raises(ValueError, match="dimension"):
            await store.add(docs)


# ============================================================================
# Delete Tests
# ============================================================================


class TestFAISSVectorStoreDelete:
    """Tests for deleting documents."""

    @pytest.mark.asyncio
    async def test_delete_success(self, mock_faiss) -> None:
        """Test successful document deletion from metadata."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]

        store = FAISSVectorStore(dimension=2)
        docs = [
            Document(id="doc1", content="Content 1", metadata={"embedding": [0.1, 0.2]}),
        ]
        await store.add(docs)

        assert "doc1" in store._id_to_idx
        await store.delete(["doc1"])
        assert "doc1" not in store._id_to_idx

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, mock_faiss) -> None:
        """Test deleting empty list does nothing."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        store = FAISSVectorStore()
        await store.delete([])


# ============================================================================
# Count Tests
# ============================================================================


class TestFAISSVectorStoreCount:
    """Tests for counting documents."""

    @pytest.mark.asyncio
    async def test_count_empty(self, mock_faiss) -> None:
        """Test count on empty store."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        store = FAISSVectorStore()
        count = await store.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_count_with_documents(self, mock_faiss) -> None:
        """Test count with documents."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]

        store = FAISSVectorStore(dimension=2)
        docs = [
            Document(id="doc1", content="Content 1", metadata={"embedding": [0.1, 0.2]}),
            Document(id="doc2", content="Content 2", metadata={"embedding": [0.3, 0.4]}),
        ]
        await store.add(docs)

        count = await store.count()
        assert count == 2


# ============================================================================
# is_available Tests
# ============================================================================


class TestFAISSVectorStoreIsAvailable:
    """Tests for availability check."""

    @pytest.mark.asyncio
    async def test_is_available_true(self, mock_faiss) -> None:
        """Test availability when FAISS is initialized."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]

        store = FAISSVectorStore()
        assert await store.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false_on_error(self, mock_faiss) -> None:
        """Test availability when initialization fails."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        store = FAISSVectorStore()

        with patch.object(store, "_ensure_index", side_effect=Exception("Init failed")):
            assert await store.is_available() is False


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestFAISSVectorStoreContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self, mock_faiss) -> None:
        """Test context manager initializes index."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]

        async with FAISSVectorStore() as store:
            assert store._index is not None
