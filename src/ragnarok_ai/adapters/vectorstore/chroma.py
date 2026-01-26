"""ChromaDB vector store adapter for ragnarok-ai.

This module provides an async client for ChromaDB,
implementing the VectorStoreProtocol for use in RAG pipelines.

ChromaDB is an open-source embedding database that runs locally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.exceptions import VectorStoreConnectionError
from ragnarok_ai.core.types import Document

if TYPE_CHECKING:
    from types import TracebackType

# Default configuration
DEFAULT_COLLECTION_NAME = "ragnarok_documents"
DEFAULT_PERSIST_DIRECTORY = ".chroma"


class ChromaVectorStore:
    """Async-compatible client for ChromaDB.

    Implements VectorStoreProtocol for document storage and similarity search.
    ChromaDB runs entirely locally with optional persistence.

    Note: ChromaDB's Python client is synchronous, but we wrap it to provide
    an async-compatible interface for consistency with other adapters.

    Attributes:
        is_local: Always True - ChromaDB runs entirely locally.
        collection_name: Name of the collection to use.
        persist_directory: Directory for persistent storage. None for in-memory.

    Example:
        Context manager (recommended):
            >>> async with ChromaVectorStore() as store:
            ...     await store.add([doc1, doc2])
            ...     results = await store.search(query_embedding, k=5)

        Standalone:
            >>> store = ChromaVectorStore()
            >>> await store.add([doc1])

        Persistent storage:
            >>> store = ChromaVectorStore(persist_directory="./chroma_data")
    """

    is_local: bool = True

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: str | None = DEFAULT_PERSIST_DIRECTORY,
    ) -> None:
        """Initialize ChromaVectorStore client.

        Args:
            collection_name: Name of the collection to use. Defaults to "ragnarok_documents".
            persist_directory: Directory for persistent storage. None for in-memory only.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client: Any = None
        self._collection: Any = None

    async def __aenter__(self) -> ChromaVectorStore:
        """Enter async context manager, initializing ChromaDB client."""
        await self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager."""
        # ChromaDB client doesn't need explicit cleanup
        self._client = None
        self._collection = None

    async def _ensure_client(self) -> None:
        """Ensure we have an active ChromaDB client and collection.

        Raises:
            VectorStoreConnectionError: If ChromaDB initialization fails.
        """
        if self._client is not None:
            return

        try:
            import chromadb

            if self.persist_directory:
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        except ImportError as e:
            msg = "chromadb is not installed. Install it with: pip install chromadb"
            raise VectorStoreConnectionError(msg) from e
        except Exception as e:
            msg = f"Failed to initialize ChromaDB: {e}"
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
        """
        try:
            await self._ensure_client()

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            documents: list[tuple[Document, float]] = []

            if not results["ids"] or not results["ids"][0]:
                return documents

            ids = results["ids"][0]
            contents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

            for doc_id, content, metadata, distance in zip(ids, contents, metadatas, distances, strict=False):
                # ChromaDB returns L2 distance by default, convert to similarity
                # For cosine distance: similarity = 1 - distance
                similarity = 1.0 - distance
                doc = Document(
                    id=str(doc_id),
                    content=content or "",
                    metadata=metadata or {},
                )
                documents.append((doc, similarity))

            return documents

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to search in ChromaDB: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def add(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Embeddings must be provided in document.metadata["embedding"].

        Args:
            documents: List of documents to add. Each document must have
                an "embedding" key in its metadata.

        Raises:
            VectorStoreConnectionError: If add operation fails.
            ValueError: If a document is missing embedding.
        """
        if not documents:
            return

        # Validate all documents have embeddings
        for doc in documents:
            if "embedding" not in doc.metadata:
                msg = f"Document {doc.id} is missing 'embedding' in metadata"
                raise ValueError(msg)

        try:
            await self._ensure_client()

            ids = [doc.id for doc in documents]
            embeddings = [doc.metadata["embedding"] for doc in documents]
            contents = [doc.content for doc in documents]
            metadatas = [
                {k: v for k, v in doc.metadata.items() if k != "embedding"}
                for doc in documents
            ]

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            )

        except ValueError:
            raise
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to add documents to ChromaDB: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def delete(self, document_ids: list[str]) -> None:
        """Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete.

        Raises:
            VectorStoreConnectionError: If delete operation fails.
        """
        if not document_ids:
            return

        try:
            await self._ensure_client()
            self._collection.delete(ids=document_ids)

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to delete documents from ChromaDB: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def count(self) -> int:
        """Get the number of documents in the collection.

        Returns:
            The number of documents in the collection.

        Raises:
            VectorStoreConnectionError: If count operation fails.
        """
        try:
            await self._ensure_client()
            return int(self._collection.count())

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to get count from ChromaDB: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def is_available(self) -> bool:
        """Check if ChromaDB is available.

        Returns:
            True if ChromaDB is initialized, False otherwise.
        """
        try:
            await self._ensure_client()
            return self._collection is not None
        except Exception:
            return False
