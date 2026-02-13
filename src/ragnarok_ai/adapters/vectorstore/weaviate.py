"""Weaviate vector store adapter for ragnarok-ai.

This module provides an async client for Weaviate vector database,
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
DEFAULT_URL = "http://localhost:8080"
DEFAULT_COLLECTION_NAME = "RagnarokDocuments"
DEFAULT_TIMEOUT = 30.0


class WeaviateVectorStore:
    """Async client for Weaviate vector database.

    Implements VectorStoreProtocol for document storage and similarity search.
    Uses weaviate-client v4 for operations with automatic schema creation.

    Can be used as a context manager for optimal performance (reuses connections),
    or standalone for simpler use cases.

    Attributes:
        is_local: False - Weaviate Cloud is typically used, though self-hosting is possible.
        url: URL for Weaviate server (local or cloud).
        api_key: Optional API key for Weaviate Cloud.
        collection_name: Name of the collection to use.
        timeout: Request timeout in seconds.

    Example:
        Context manager (recommended for multiple calls):
            >>> async with WeaviateVectorStore(
            ...     url="https://your-cluster.weaviate.cloud",
            ...     api_key="your-api-key",
            ... ) as store:
            ...     await store.add([doc1, doc2])
            ...     results = await store.search(query_embedding, k=5)

        Local instance:
            >>> store = WeaviateVectorStore(url="http://localhost:8080")
            >>> await store.add([doc1])
    """

    is_local: bool = False

    def __init__(
        self,
        url: str = DEFAULT_URL,
        api_key: str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize WeaviateVectorStore client.

        Args:
            url: URL for Weaviate server. Defaults to localhost:8080.
            api_key: Optional API key for Weaviate Cloud. If not provided,
                reads from WEAVIATE_API_KEY environment variable.
            collection_name: Name of the collection to use. Defaults to "RagnarokDocuments".
            timeout: Request timeout in seconds. Defaults to 30.0.
        """
        self.url = url.rstrip("/")
        self.api_key = api_key or os.environ.get("WEAVIATE_API_KEY")
        self.collection_name = collection_name
        self.timeout = timeout
        self._client: Any = None

    async def __aenter__(self) -> WeaviateVectorStore:
        """Enter async context manager, creating a reusable Weaviate client."""
        await self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager, closing the Weaviate client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def _ensure_client(self) -> Any:
        """Ensure we have an active Weaviate client.

        Returns:
            A Weaviate client instance.

        Raises:
            VectorStoreConnectionError: If connection to Weaviate fails.
        """
        if self._client is not None:
            return self._client

        try:
            import weaviate
            from weaviate.classes.init import Auth

            # Determine if this is a cloud URL or local
            is_cloud = "weaviate.cloud" in self.url or "wcs.api.weaviate" in self.url

            if is_cloud and self.api_key:
                self._client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.url,
                    auth_credentials=Auth.api_key(self.api_key),
                )
            elif self.api_key:
                self._client = weaviate.connect_to_custom(
                    http_host=self.url.replace("http://", "").replace("https://", "").split(":")[0],
                    http_port=int(self.url.split(":")[-1]) if ":" in self.url.split("/")[-1] else 8080,
                    http_secure=self.url.startswith("https"),
                    auth_credentials=Auth.api_key(self.api_key),
                )
            else:
                self._client = weaviate.connect_to_local(
                    host=self.url.replace("http://", "").replace("https://", "").split(":")[0],
                    port=int(self.url.split(":")[-1]) if ":" in self.url.split("/")[-1] else 8080,
                )

            # Ensure collection exists
            await self._ensure_collection()

            return self._client

        except ImportError as e:
            msg = "weaviate-client is not installed. Install it with: pip install weaviate-client"
            raise VectorStoreConnectionError(msg) from e
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to connect to Weaviate at {self.url}: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def _ensure_collection(self) -> None:
        """Ensure the collection exists with correct configuration."""
        if self._client is None:
            return

        try:
            from weaviate.classes.config import Configure, DataType, Property

            if not self._client.collections.exists(self.collection_name):
                self._client.collections.create(
                    name=self.collection_name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="doc_id", data_type=DataType.TEXT),
                    ],
                )
        except Exception as e:
            msg = f"Failed to create collection in Weaviate: {e}"
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
            VectorStoreConnectionError: If connection to Weaviate fails.
        """
        try:
            client = await self._ensure_client()

            collection = client.collections.get(self.collection_name)
            results = collection.query.near_vector(
                near_vector=query_embedding,
                limit=k,
                return_metadata=["distance"],
            )

            documents: list[tuple[Document, float]] = []
            for obj in results.objects:
                props = obj.properties
                # Convert distance to similarity score (1 - distance for cosine)
                score = 1.0 - (obj.metadata.distance or 0.0)
                doc = Document(
                    id=props.get("doc_id", str(obj.uuid)),
                    content=props.get("content", ""),
                    metadata={k: v for k, v in props.items() if k not in ("content", "doc_id")},
                )
                documents.append((doc, score))

            return documents

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to search in Weaviate: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def add(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Embeddings must be provided in document.metadata["embedding"].

        Args:
            documents: List of documents to add. Each document must have
                an "embedding" key in its metadata.

        Raises:
            VectorStoreConnectionError: If connection to Weaviate fails.
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
            client = await self._ensure_client()

            collection = client.collections.get(self.collection_name)

            with collection.batch.dynamic() as batch:
                for doc in documents:
                    embedding = doc.metadata["embedding"]
                    # Store content and other metadata (excluding embedding)
                    properties = {
                        "content": doc.content,
                        "doc_id": doc.id,
                        **{k: v for k, v in doc.metadata.items() if k != "embedding"},
                    }
                    batch.add_object(
                        properties=properties,
                        vector=embedding,
                    )

        except ValueError:
            raise
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to add documents to Weaviate: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def delete(self, document_ids: list[str]) -> None:
        """Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete.

        Raises:
            VectorStoreConnectionError: If connection to Weaviate fails.
        """
        if not document_ids:
            return

        try:
            from weaviate.classes.query import Filter

            client = await self._ensure_client()

            collection = client.collections.get(self.collection_name)

            for doc_id in document_ids:
                collection.data.delete_many(
                    where=Filter.by_property("doc_id").equal(doc_id)
                )

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to delete documents from Weaviate: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def count(self) -> int:
        """Get the number of documents in the collection.

        Returns:
            The number of documents in the collection.

        Raises:
            VectorStoreConnectionError: If connection to Weaviate fails.
        """
        try:
            client = await self._ensure_client()

            collection = client.collections.get(self.collection_name)
            result = collection.aggregate.over_all(total_count=True)
            return result.total_count or 0

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to get count from Weaviate: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def is_available(self) -> bool:
        """Check if Weaviate is available and responding.

        Returns:
            True if Weaviate is reachable, False otherwise.

        Example:
            >>> if await store.is_available():
            ...     print("Weaviate is ready")
        """
        try:
            client = await self._ensure_client()
            return client.is_ready()
        except Exception:
            return False
