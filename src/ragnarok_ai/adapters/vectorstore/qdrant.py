"""Qdrant vector store adapter for ragnarok-ai.

This module provides an async client for Qdrant vector database,
implementing the VectorStoreProtocol for use in RAG pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.exceptions import VectorStoreConnectionError
from ragnarok_ai.core.types import Document

if TYPE_CHECKING:
    from types import TracebackType

    from qdrant_client import AsyncQdrantClient

# Default configuration
DEFAULT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "ragnarok_documents"
DEFAULT_VECTOR_SIZE = 768
DEFAULT_TIMEOUT = 30.0


class QdrantVectorStore:
    """Async client for Qdrant vector database.

    Implements VectorStoreProtocol for document storage and similarity search.
    Uses qdrant-client for async operations with automatic collection creation.

    Can be used as a context manager for optimal performance (reuses connections),
    or standalone for simpler use cases.

    Attributes:
        url: URL for Qdrant server.
        api_key: Optional API key for authentication.
        collection_name: Name of the collection to use.
        vector_size: Dimension of embedding vectors.
        timeout: Request timeout in seconds.

    Example:
        Context manager (recommended for multiple calls):
            >>> async with QdrantVectorStore(vector_size=768) as store:
            ...     await store.add([doc1, doc2])
            ...     results = await store.search(query_embedding, k=5)

        Standalone (simpler, creates new connection per call):
            >>> store = QdrantVectorStore(vector_size=768)
            >>> await store.add([doc1])
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        api_key: str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize QdrantVectorStore client.

        Args:
            url: URL for Qdrant server. Defaults to localhost:6333.
            api_key: Optional API key for authentication.
            collection_name: Name of the collection to use. Defaults to "ragnarok_documents".
            vector_size: Dimension of embedding vectors. Defaults to 768.
            timeout: Request timeout in seconds. Defaults to 30.0.
        """
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout = timeout
        self._client: AsyncQdrantClient | None = None

    async def __aenter__(self) -> QdrantVectorStore:
        """Enter async context manager, creating a reusable Qdrant client."""
        await self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager, closing the Qdrant client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def _ensure_client(self) -> AsyncQdrantClient:
        """Ensure we have an active Qdrant client.

        Creates a new client if one doesn't exist, and ensures the collection
        is created with the proper configuration.

        Returns:
            An AsyncQdrantClient instance.

        Raises:
            VectorStoreConnectionError: If connection to Qdrant fails.
        """
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.http.exceptions import UnexpectedResponse

            self._client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=self.timeout,
            )

            # Ensure collection exists
            try:
                await self._ensure_collection()
            except UnexpectedResponse as e:
                msg = f"Failed to create collection in Qdrant: {e}"
                raise VectorStoreConnectionError(msg) from e

            return self._client

        except ImportError as e:
            msg = "qdrant-client is not installed. Install it with: pip install qdrant-client"
            raise VectorStoreConnectionError(msg) from e
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to connect to Qdrant at {self.url}: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def _ensure_collection(self) -> None:
        """Ensure the collection exists with correct configuration."""
        if self._client is None:
            return

        from qdrant_client.http.models import Distance, VectorParams

        collections = await self._client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.collection_name not in collection_names:
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

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
            VectorStoreConnectionError: If connection to Qdrant fails.
            ValueError: If query_embedding dimension doesn't match vector_size.
        """
        if len(query_embedding) != self.vector_size:
            msg = f"Query embedding dimension ({len(query_embedding)}) doesn't match vector_size ({self.vector_size})"
            raise ValueError(msg)

        try:
            client = await self._ensure_client()

            results = await client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
            )

            documents: list[tuple[Document, float]] = []
            for point in results:
                payload: dict[str, Any] = point.payload or {}
                doc = Document(
                    id=str(point.id),
                    content=payload.get("content", ""),
                    metadata={
                        k: v for k, v in payload.items() if k != "content"
                    },
                )
                documents.append((doc, point.score))

            return documents

        except ValueError:
            raise
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to search in Qdrant: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def add(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Embeddings must be provided in document.metadata["embedding"].

        Args:
            documents: List of documents to add. Each document must have
                an "embedding" key in its metadata.

        Raises:
            VectorStoreConnectionError: If connection to Qdrant fails.
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
            if len(embedding) != self.vector_size:
                msg = f"Document {doc.id} embedding dimension ({len(embedding)}) doesn't match vector_size ({self.vector_size})"
                raise ValueError(msg)

        try:
            from qdrant_client.http.models import PointStruct

            client = await self._ensure_client()

            points = []
            for doc in documents:
                embedding = doc.metadata["embedding"]
                # Store content and other metadata (excluding embedding)
                payload = {
                    "content": doc.content,
                    **{k: v for k, v in doc.metadata.items() if k != "embedding"},
                }
                points.append(
                    PointStruct(
                        id=doc.id,
                        vector=embedding,
                        payload=payload,
                    )
                )

            await client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

        except ValueError:
            raise
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to add documents to Qdrant: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def delete(self, document_ids: list[str]) -> None:
        """Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete.

        Raises:
            VectorStoreConnectionError: If connection to Qdrant fails.
        """
        if not document_ids:
            return

        try:
            from qdrant_client.http.models import PointIdsList

            client = await self._ensure_client()

            await client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=document_ids),
            )

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to delete documents from Qdrant: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def count(self) -> int:
        """Get the number of documents in the collection.

        Returns:
            The number of documents in the collection.

        Raises:
            VectorStoreConnectionError: If connection to Qdrant fails.
        """
        try:
            client = await self._ensure_client()

            collection_info = await client.get_collection(self.collection_name)
            return int(collection_info.points_count)

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to get count from Qdrant: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def is_available(self) -> bool:
        """Check if Qdrant is available and responding.

        Returns:
            True if Qdrant is reachable, False otherwise.

        Example:
            >>> if await store.is_available():
            ...     print("Qdrant is ready")
        """
        try:
            client = await self._ensure_client()
            # Try to get collections to verify connection
            await client.get_collections()
            return True
        except Exception:
            return False
