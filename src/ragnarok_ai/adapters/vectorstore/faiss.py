"""FAISS vector store adapter for ragnarok-ai.

This module provides an async-compatible client for FAISS,
implementing the VectorStoreProtocol for use in RAG pipelines.

FAISS is a library for efficient similarity search that runs entirely locally
with no server required - pure in-memory or file-based persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.exceptions import VectorStoreConnectionError
from ragnarok_ai.core.types import Document

if TYPE_CHECKING:
    from types import TracebackType


class FAISSVectorStore:
    """In-memory vector store using FAISS.

    Implements VectorStoreProtocol for document storage and similarity search.
    FAISS runs entirely locally with no server required.

    Attributes:
        is_local: Always True - FAISS runs entirely locally, no server needed.
        dimension: Dimension of embedding vectors.
        index_path: Optional path for persisting the index to disk.

    Example:
        Context manager (recommended):
            >>> async with FAISSVectorStore(dimension=768) as store:
            ...     await store.add([doc1, doc2])
            ...     results = await store.search(query_embedding, k=5)

        Standalone:
            >>> store = FAISSVectorStore(dimension=768)
            >>> await store.add([doc1])

        With persistence:
            >>> store = FAISSVectorStore(dimension=768, index_path="./faiss_index")
            >>> await store.add([doc1])
            >>> await store.save()  # Save to disk
    """

    is_local: bool = True

    def __init__(
        self,
        dimension: int = 768,
        index_path: str | None = None,
    ) -> None:
        """Initialize FAISSVectorStore.

        Args:
            dimension: Dimension of embedding vectors. Defaults to 768.
            index_path: Optional path for persisting index. If provided and exists,
                        the index will be loaded from disk.
        """
        self.dimension = dimension
        self.index_path = index_path
        self._index: Any = None
        self._documents: dict[int, Document] = {}
        self._id_to_idx: dict[str, int] = {}
        self._next_idx: int = 0

    async def __aenter__(self) -> FAISSVectorStore:
        """Enter async context manager, initializing FAISS index."""
        await self._ensure_index()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager, optionally saving index."""
        # Optionally save on exit if path is configured
        if self.index_path and self._index is not None:
            await self.save()

    async def _ensure_index(self) -> None:
        """Ensure we have an initialized FAISS index.

        Raises:
            VectorStoreConnectionError: If FAISS initialization fails.
        """
        if self._index is not None:
            return

        try:
            import faiss

            # Try to load existing index
            if self.index_path and Path(self.index_path).exists():
                self._index = faiss.read_index(self.index_path)
                # Load document metadata if available
                metadata_path = Path(self.index_path).with_suffix(".meta")
                if metadata_path.exists():
                    import json

                    with metadata_path.open() as f:
                        data = json.load(f)
                        self._documents = {
                            int(k): Document(**v) for k, v in data["documents"].items()
                        }
                        self._id_to_idx = data["id_to_idx"]
                        self._next_idx = data["next_idx"]
            else:
                # Create new index using IndexFlatIP (inner product for cosine similarity)
                self._index = faiss.IndexFlatIP(self.dimension)

        except ImportError as e:
            msg = "faiss-cpu is not installed. Install it with: pip install faiss-cpu"
            raise VectorStoreConnectionError(msg) from e
        except Exception as e:
            msg = f"Failed to initialize FAISS: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def search(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents.

        Args:
            query_embedding: The embedding vector to search with.
            k: Number of results to return. Defaults to 10.

        Returns:
            A list of tuples containing (document, similarity_score).
            Results are sorted by similarity in descending order.

        Raises:
            VectorStoreConnectionError: If search fails.
            ValueError: If query_embedding dimension doesn't match.
        """
        if len(query_embedding) != self.dimension:
            msg = f"Query embedding dimension ({len(query_embedding)}) doesn't match index dimension ({self.dimension})"
            raise ValueError(msg)

        try:
            import numpy as np

            await self._ensure_index()

            if self._index.ntotal == 0:
                return []

            # Normalize query for cosine similarity
            query = np.array([query_embedding], dtype=np.float32)
            faiss = __import__("faiss")
            faiss.normalize_L2(query)

            # Search
            actual_k = min(k, self._index.ntotal)
            distances, indices = self._index.search(query, actual_k)

            results: list[tuple[Document, float]] = []
            for idx, score in zip(indices[0], distances[0], strict=False):
                if idx >= 0 and idx in self._documents:
                    results.append((self._documents[idx], float(score)))

            return results

        except ValueError:
            raise
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to search in FAISS: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def add(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Embeddings must be provided in document.metadata["embedding"].

        Args:
            documents: List of documents to add. Each document must have
                an "embedding" key in its metadata.

        Raises:
            VectorStoreConnectionError: If add operation fails.
            ValueError: If a document is missing embedding or has wrong dimension.
        """
        if not documents:
            return

        # Validate all documents have embeddings with correct dimension
        for doc in documents:
            if "embedding" not in doc.metadata:
                msg = f"Document {doc.id} is missing 'embedding' in metadata"
                raise ValueError(msg)
            embedding = doc.metadata["embedding"]
            if len(embedding) != self.dimension:
                msg = f"Document {doc.id} embedding dimension ({len(embedding)}) doesn't match index dimension ({self.dimension})"
                raise ValueError(msg)

        try:
            import numpy as np

            await self._ensure_index()

            faiss = __import__("faiss")

            embeddings = []
            for doc in documents:
                embedding = doc.metadata["embedding"]
                embeddings.append(embedding)

                # Store document (without embedding in metadata to save memory)
                doc_to_store = Document(
                    id=doc.id,
                    content=doc.content,
                    metadata={k: v for k, v in doc.metadata.items() if k != "embedding"},
                )

                # Handle updates - remove old entry if exists
                if doc.id in self._id_to_idx:
                    old_idx = self._id_to_idx[doc.id]
                    if old_idx in self._documents:
                        del self._documents[old_idx]

                self._documents[self._next_idx] = doc_to_store
                self._id_to_idx[doc.id] = self._next_idx
                self._next_idx += 1

            # Normalize embeddings for cosine similarity
            vectors = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(vectors)

            self._index.add(vectors)

        except ValueError:
            raise
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to add documents to FAISS: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def delete(self, document_ids: list[str]) -> None:
        """Delete documents from the vector store.

        Note: FAISS IndexFlatIP doesn't support deletion. This method removes
        documents from the metadata but not from the index. For full deletion,
        rebuild the index.

        Args:
            document_ids: List of document IDs to delete.
        """
        if not document_ids:
            return

        for doc_id in document_ids:
            if doc_id in self._id_to_idx:
                idx = self._id_to_idx[doc_id]
                if idx in self._documents:
                    del self._documents[idx]
                del self._id_to_idx[doc_id]

    async def count(self) -> int:
        """Get the number of documents in the index.

        Returns:
            The number of active documents.
        """
        return len(self._documents)

    async def save(self) -> None:
        """Save the index and metadata to disk.

        Raises:
            VectorStoreConnectionError: If save operation fails.
            ValueError: If no index_path is configured.
        """
        if not self.index_path:
            msg = "No index_path configured for persistence"
            raise ValueError(msg)

        if self._index is None:
            return

        try:
            import json

            faiss = __import__("faiss")
            faiss.write_index(self._index, self.index_path)

            # Save metadata
            metadata_path = Path(self.index_path).with_suffix(".meta")
            with metadata_path.open("w") as f:
                json.dump(
                    {
                        "documents": {k: v.model_dump() for k, v in self._documents.items()},
                        "id_to_idx": self._id_to_idx,
                        "next_idx": self._next_idx,
                    },
                    f,
                )

        except Exception as e:
            msg = f"Failed to save FAISS index: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def is_available(self) -> bool:
        """Check if FAISS is available.

        Returns:
            True if FAISS is initialized, False otherwise.
        """
        try:
            await self._ensure_index()
            return self._index is not None
        except Exception:
            return False
