"""Pinecone vector store adapter for ragnarok-ai.

This module provides an async client for Pinecone vector database,
implementing the VectorStoreProtocol for use in RAG pipelines.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.exceptions import VectorStoreConnectionError
from ragnarok_ai.core.types import Document

if TYPE_CHECKING:
    from types import TracebackType

# Default configuration
DEFAULT_NAMESPACE = ""
DEFAULT_TIMEOUT = 30.0


class PineconeVectorStore:
    """Async client for Pinecone vector database.

    Implements VectorStoreProtocol for document storage and similarity search.
    Uses pinecone-client for operations with serverless or pod-based indexes.

    Can be used as a context manager for optimal performance (reuses connections),
    or standalone for simpler use cases.

    Attributes:
        is_local: Always False - Pinecone is a cloud-hosted service.
        api_key: API key for Pinecone authentication.
        index_name: Name of the Pinecone index to use.
        namespace: Namespace within the index.
        timeout: Request timeout in seconds.

    Example:
        Context manager (recommended for multiple calls):
            >>> async with PineconeVectorStore(
            ...     api_key="your-api-key",
            ...     index_name="my-index",
            ... ) as store:
            ...     await store.add([doc1, doc2])
            ...     results = await store.search(query_embedding, k=5)

        Standalone (simpler, creates new connection per call):
            >>> store = PineconeVectorStore(api_key="key", index_name="idx")
            >>> await store.add([doc1])
    """

    is_local: bool = False

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str = DEFAULT_NAMESPACE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize PineconeVectorStore client.

        Args:
            api_key: API key for Pinecone. If not provided, reads from
                PINECONE_API_KEY environment variable.
            index_name: Name of the Pinecone index. Required.
            namespace: Namespace within the index. Defaults to empty string.
            timeout: Request timeout in seconds. Defaults to 30.0.

        Raises:
            ValueError: If API key or index name is not provided.
        """
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            msg = "API key required. Set PINECONE_API_KEY or pass api_key parameter."
            raise ValueError(msg)

        if not index_name:
            msg = "index_name is required for Pinecone"
            raise ValueError(msg)

        self.index_name = index_name
        self.namespace = namespace
        self.timeout = timeout
        self._client: Any = None
        self._index: Any = None

    async def __aenter__(self) -> PineconeVectorStore:
        """Enter async context manager, creating a reusable Pinecone client."""
        await self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager, closing the Pinecone client."""
        self._index = None
        self._client = None

    async def _ensure_client(self) -> Any:
        """Ensure we have an active Pinecone client and index.

        Returns:
            A Pinecone Index instance.

        Raises:
            VectorStoreConnectionError: If connection to Pinecone fails.
        """
        if self._index is not None:
            return self._index

        try:
            from pinecone import Pinecone

            self._client = Pinecone(api_key=self.api_key)
            self._index = self._client.Index(self.index_name)

            return self._index

        except ImportError as e:
            msg = "pinecone-client is not installed. Install it with: pip install pinecone-client"
            raise VectorStoreConnectionError(msg) from e
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to connect to Pinecone index {self.index_name}: {e}"
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
            VectorStoreConnectionError: If connection to Pinecone fails.
        """
        try:
            index = await self._ensure_client()

            results = index.query(
                vector=query_embedding,
                top_k=k,
                namespace=self.namespace,
                include_metadata=True,
            )

            documents: list[tuple[Document, float]] = []
            for match in results.get("matches", []):
                metadata: dict[str, Any] = match.get("metadata", {})
                doc = Document(
                    id=match["id"],
                    content=metadata.pop("content", ""),
                    metadata=metadata,
                )
                documents.append((doc, match["score"]))

            return documents

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to search in Pinecone: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def add(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Embeddings must be provided in document.metadata["embedding"].

        Args:
            documents: List of documents to add. Each document must have
                an "embedding" key in its metadata.

        Raises:
            VectorStoreConnectionError: If connection to Pinecone fails.
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
            index = await self._ensure_client()

            vectors = []
            for doc in documents:
                embedding = doc.metadata["embedding"]
                # Store content and other metadata (excluding embedding)
                metadata = {
                    "content": doc.content,
                    **{k: v for k, v in doc.metadata.items() if k != "embedding"},
                }
                vectors.append({
                    "id": doc.id,
                    "values": embedding,
                    "metadata": metadata,
                })

            # Upsert in batches of 100 (Pinecone recommendation)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                index.upsert(vectors=batch, namespace=self.namespace)

        except ValueError:
            raise
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to add documents to Pinecone: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def delete(self, document_ids: list[str]) -> None:
        """Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete.

        Raises:
            VectorStoreConnectionError: If connection to Pinecone fails.
        """
        if not document_ids:
            return

        try:
            index = await self._ensure_client()

            index.delete(ids=document_ids, namespace=self.namespace)

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to delete documents from Pinecone: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def count(self) -> int:
        """Get the number of documents in the namespace.

        Returns:
            The number of documents in the namespace.

        Raises:
            VectorStoreConnectionError: If connection to Pinecone fails.
        """
        try:
            index = await self._ensure_client()

            stats = index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            namespace_stats = namespaces.get(self.namespace or "", {})
            return int(namespace_stats.get("vector_count", 0))

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to get count from Pinecone: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def is_available(self) -> bool:
        """Check if Pinecone is available and responding.

        Returns:
            True if Pinecone is reachable, False otherwise.

        Example:
            >>> if await store.is_available():
            ...     print("Pinecone is ready")
        """
        try:
            index = await self._ensure_client()
            # Try to describe index stats to verify connection
            index.describe_index_stats()
            return True
        except Exception:
            return False
