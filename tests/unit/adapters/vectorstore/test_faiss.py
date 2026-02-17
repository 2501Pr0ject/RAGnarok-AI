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

    @pytest.mark.asyncio
    async def test_context_manager_saves_on_exit(self, mock_faiss) -> None:
        """Test context manager saves index on exit if path is configured."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        mock_faiss_module = mock_faiss["faiss"]

        store = FAISSVectorStore(index_path="/tmp/test_index")
        async with store:
            pass

        mock_faiss_module.write_index.assert_called()


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestFAISSVectorStoreErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_ensure_index_already_initialized(self, mock_faiss) -> None:
        """Test _ensure_index returns early if index already exists."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        mock_faiss_module = mock_faiss["faiss"]

        store = FAISSVectorStore()
        await store._ensure_index()

        # Reset call count
        mock_faiss_module.IndexFlatIP.reset_mock()

        # Call again - should return early
        await store._ensure_index()

        mock_faiss_module.IndexFlatIP.assert_not_called()

    @pytest.mark.asyncio
    async def test_import_error_raises_connection_error(self) -> None:
        """Test that ImportError raises VectorStoreConnectionError."""
        from ragnarok_ai.core.exceptions import VectorStoreConnectionError

        with patch.dict(sys.modules, {"faiss": None}):
            if "ragnarok_ai.adapters.vectorstore.faiss" in sys.modules:
                del sys.modules["ragnarok_ai.adapters.vectorstore.faiss"]

            from ragnarok_ai.adapters.vectorstore.faiss import FAISSVectorStore

            store = FAISSVectorStore()
            store._index = None

            with patch("builtins.__import__", side_effect=ImportError("No module named 'faiss'")):
                with pytest.raises(VectorStoreConnectionError, match="faiss-cpu is not installed"):
                    await store._ensure_index()

    @pytest.mark.asyncio
    async def test_generic_exception_raises_connection_error(self, mock_faiss) -> None:
        """Test that generic exception raises VectorStoreConnectionError."""
        from ragnarok_ai.core.exceptions import VectorStoreConnectionError

        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        mock_faiss_module = mock_faiss["faiss"]

        mock_faiss_module.IndexFlatIP.side_effect = RuntimeError("Unexpected error")

        store = FAISSVectorStore()

        with pytest.raises(VectorStoreConnectionError, match="Failed to initialize FAISS"):
            await store._ensure_index()

        mock_faiss_module.IndexFlatIP.side_effect = None

    @pytest.mark.asyncio
    async def test_search_generic_error(self, mock_faiss) -> None:
        """Test search raises VectorStoreConnectionError on generic error."""
        from ragnarok_ai.core.exceptions import VectorStoreConnectionError

        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        mock_index = mock_faiss["index"]
        mock_index.ntotal = 10

        mock_index.search.side_effect = RuntimeError("Search failed")

        store = FAISSVectorStore(dimension=3)

        with pytest.raises(VectorStoreConnectionError, match="Failed to search"):
            await store.search([0.1, 0.2, 0.3], k=5)

        mock_index.search.side_effect = None

    @pytest.mark.asyncio
    async def test_search_reraises_value_error(self, mock_faiss) -> None:
        """Test search re-raises ValueError for dimension mismatch."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]

        store = FAISSVectorStore(dimension=768)

        with pytest.raises(ValueError, match="dimension"):
            await store.search([0.1, 0.2], k=5)

    @pytest.mark.asyncio
    async def test_add_generic_error(self, mock_faiss) -> None:
        """Test add raises VectorStoreConnectionError on generic error."""
        from ragnarok_ai.core.exceptions import VectorStoreConnectionError

        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        mock_index = mock_faiss["index"]

        mock_index.add.side_effect = RuntimeError("Add failed")

        store = FAISSVectorStore(dimension=2)
        docs = [Document(id="doc1", content="Content", metadata={"embedding": [0.1, 0.2]})]

        with pytest.raises(VectorStoreConnectionError, match="Failed to add"):
            await store.add(docs)

        mock_index.add.side_effect = None

    @pytest.mark.asyncio
    async def test_add_reraises_value_error(self, mock_faiss) -> None:
        """Test add re-raises ValueError."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]

        store = FAISSVectorStore(dimension=768)
        docs = [Document(id="doc1", content="Content", metadata={"embedding": [0.1, 0.2]})]

        with pytest.raises(ValueError, match="dimension"):
            await store.add(docs)


# ============================================================================
# Search with Results Tests
# ============================================================================


class TestFAISSVectorStoreSearchResults:
    """Tests for search returning results."""

    @pytest.mark.asyncio
    async def test_search_returns_documents(self) -> None:
        """Test search returns documents with scores."""
        import numpy as np

        # Import with mocked modules
        mock_faiss_module = MagicMock()
        mock_np_module = MagicMock()
        mock_np_module.array.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mock_np_module.float32 = np.float32

        with patch.dict(sys.modules, {"faiss": mock_faiss_module, "numpy": mock_np_module}):
            if "ragnarok_ai.adapters.vectorstore.faiss" in sys.modules:
                del sys.modules["ragnarok_ai.adapters.vectorstore.faiss"]

            from ragnarok_ai.adapters.vectorstore.faiss import FAISSVectorStore

            # Create a mock index
            mock_index = MagicMock()
            mock_index.ntotal = 2
            # FAISS returns (distances, indices) - code unpacks as distances, indices
            mock_index.search.return_value = (
                np.array([[0.95, 0.85]]),  # distances (scores)
                np.array([[0, 1]]),  # indices
            )

            store = FAISSVectorStore(dimension=3)
            # Set up the store with mock data directly (bypass _ensure_index)
            store._documents = {
                0: Document(id="doc1", content="Content 1", metadata={}),
                1: Document(id="doc2", content="Content 2", metadata={}),
            }
            store._index = mock_index

            results = await store.search([0.1, 0.2, 0.3], k=5)

            assert len(results) == 2
            assert results[0][0].id == "doc1"
            assert results[1][0].id == "doc2"

    @pytest.mark.asyncio
    async def test_search_handles_negative_index(self) -> None:
        """Test search handles negative indices (not found)."""
        import numpy as np

        mock_faiss_module = MagicMock()
        mock_np_module = MagicMock()
        mock_np_module.array.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mock_np_module.float32 = np.float32

        with patch.dict(sys.modules, {"faiss": mock_faiss_module, "numpy": mock_np_module}):
            if "ragnarok_ai.adapters.vectorstore.faiss" in sys.modules:
                del sys.modules["ragnarok_ai.adapters.vectorstore.faiss"]

            from ragnarok_ai.adapters.vectorstore.faiss import FAISSVectorStore

            # Create a mock index
            mock_index = MagicMock()
            mock_index.ntotal = 2
            mock_index.search.return_value = (
                np.array([[-1, 0]]),  # -1 means not found
                np.array([[0.0, 0.85]]),
            )

            store = FAISSVectorStore(dimension=3)
            store._documents = {
                0: Document(id="doc1", content="Content 1", metadata={}),
            }
            store._index = mock_index

            results = await store.search([0.1, 0.2, 0.3], k=5)

            assert len(results) == 1
            assert results[0][0].id == "doc1"


# ============================================================================
# Update/Upsert Tests
# ============================================================================


class TestFAISSVectorStoreUpdate:
    """Tests for document update handling."""

    @pytest.mark.asyncio
    async def test_add_updates_existing_document(self, mock_faiss) -> None:
        """Test adding document with same ID updates the existing entry."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]

        store = FAISSVectorStore(dimension=2)

        # Add first document
        docs1 = [Document(id="doc1", content="Content 1", metadata={"embedding": [0.1, 0.2]})]
        await store.add(docs1)

        assert store._id_to_idx["doc1"] == 0
        assert store._documents[0].content == "Content 1"

        # Add same document with updated content
        docs2 = [Document(id="doc1", content="Updated Content", metadata={"embedding": [0.3, 0.4]})]
        await store.add(docs2)

        # Old index entry should be removed
        assert 0 not in store._documents
        # New entry
        assert store._id_to_idx["doc1"] == 1
        assert store._documents[1].content == "Updated Content"


# ============================================================================
# Save Tests
# ============================================================================


class TestFAISSVectorStoreSave:
    """Tests for index persistence."""

    @pytest.mark.asyncio
    async def test_save_without_path_raises_error(self, mock_faiss) -> None:
        """Test save without index_path raises ValueError."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]

        store = FAISSVectorStore(index_path=None)

        with pytest.raises(ValueError, match="No index_path configured"):
            await store.save()

    @pytest.mark.asyncio
    async def test_save_without_index_does_nothing(self, mock_faiss) -> None:
        """Test save without initialized index returns early."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        mock_faiss_module = mock_faiss["faiss"]

        store = FAISSVectorStore(index_path="/tmp/test")
        store._index = None

        await store.save()

        mock_faiss_module.write_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_success(self, mock_faiss) -> None:
        """Test successful save writes index and metadata."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        mock_faiss_module = mock_faiss["faiss"]

        import tempfile
        import json
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = f"{tmpdir}/test_index"
            store = FAISSVectorStore(dimension=2, index_path=index_path)

            # Add document
            docs = [Document(id="doc1", content="Content 1", metadata={"embedding": [0.1, 0.2]})]
            await store.add(docs)

            await store.save()

            # Check metadata file was created
            metadata_path = Path(index_path).with_suffix(".meta")
            assert metadata_path.exists()

            with metadata_path.open() as f:
                data = json.load(f)
                assert "documents" in data
                assert "id_to_idx" in data
                assert "next_idx" in data

    @pytest.mark.asyncio
    async def test_save_error_raises_connection_error(self, mock_faiss) -> None:
        """Test save error raises VectorStoreConnectionError."""
        from ragnarok_ai.core.exceptions import VectorStoreConnectionError

        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        mock_faiss_module = mock_faiss["faiss"]

        mock_faiss_module.write_index.side_effect = RuntimeError("Write failed")

        store = FAISSVectorStore(dimension=2, index_path="/tmp/test")
        docs = [Document(id="doc1", content="Content 1", metadata={"embedding": [0.1, 0.2]})]
        await store.add(docs)

        with pytest.raises(VectorStoreConnectionError, match="Failed to save"):
            await store.save()

        mock_faiss_module.write_index.side_effect = None


# ============================================================================
# Load Existing Index Tests
# ============================================================================


class TestFAISSVectorStoreLoad:
    """Tests for loading existing index."""

    @pytest.mark.asyncio
    async def test_load_existing_index(self, mock_faiss) -> None:
        """Test loading existing index and metadata."""
        FAISSVectorStore = mock_faiss["FAISSVectorStore"]
        mock_faiss_module = mock_faiss["faiss"]

        import tempfile
        import json
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = f"{tmpdir}/test_index"

            # Create fake index file
            Path(index_path).touch()

            # Create metadata file
            metadata = {
                "documents": {"0": {"id": "doc1", "content": "Loaded content", "metadata": {}}},
                "id_to_idx": {"doc1": 0},
                "next_idx": 1,
            }
            metadata_path = Path(index_path).with_suffix(".meta")
            with metadata_path.open("w") as f:
                json.dump(metadata, f)

            store = FAISSVectorStore(dimension=768, index_path=index_path)
            await store._ensure_index()

            mock_faiss_module.read_index.assert_called_once_with(index_path)
            assert store._documents[0].id == "doc1"
            assert store._documents[0].content == "Loaded content"
            assert store._id_to_idx["doc1"] == 0
            assert store._next_idx == 1
