"""Weaviate vector store adapter for ragnarok-ai.

This module provides an async client for Weaviate vector database,
implementing the VectorStoreProtocol for use in RAG pipelines.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ragnarok_ai.adapters.vectorstore._errors import is_connection_failure
from ragnarok_ai.core.exceptions import VectorStoreConnectionError, VectorStoreDependencyError
from ragnarok_ai.core.types import Document

if TYPE_CHECKING:
    from types import TracebackType

# The errors of weaviate-client that mean the instance could not be reached. Names the installed version
# does not have are skipped (WeaviateTimeoutError is not in 4.6.2). Not in the list: the responses of the
# server (UnexpectedStatusCodeError, AuthenticationFailedError, WeaviateQueryError) and
# WeaviateClosedClientError, a misuse of the client
_WEAVIATE_EXCEPTIONS = "weaviate.exceptions"
_WEAVIATE_CONNECTION_ERRORS = (
    "WeaviateConnectionError",
    "WeaviateStartUpError",
    "WeaviateTimeoutError",
    "WeaviateGRPCUnavailableError",
)


def _connection_failure(exc: BaseException) -> bool:
    return is_connection_failure(exc, _WEAVIATE_EXCEPTIONS, _WEAVIATE_CONNECTION_ERRORS)


# Default configuration
DEFAULT_URL = "http://localhost:8080"
DEFAULT_COLLECTION_NAME = "RagnarokDocuments"
DEFAULT_TIMEOUT = 30.0
# Weaviate's default gRPC port, next to 8080 for HTTP
DEFAULT_GRPC_PORT = 50051


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
        grpc_host: gRPC host of a self-hosted instance with an API key (default: the host of ``url``).
        grpc_port: gRPC port of that instance (default: 50051).
        grpc_secure: Whether its gRPC endpoint uses TLS (default: the scheme of ``url``).

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

        Self-hosted instance with an API key (gRPC on the same host, port 50051, unless said otherwise):
            >>> store = WeaviateVectorStore(
            ...     url="https://weaviate.example.org:8443",
            ...     api_key="your-api-key",
            ...     grpc_port=50443,
            ... )
    """

    is_local: bool = False

    def __init__(
        self,
        url: str = DEFAULT_URL,
        api_key: str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        timeout: float = DEFAULT_TIMEOUT,
        grpc_host: str | None = None,
        grpc_port: int = DEFAULT_GRPC_PORT,
        grpc_secure: bool | None = None,
    ) -> None:
        """Initialize WeaviateVectorStore client.

        Args:
            url: URL for Weaviate server. Defaults to localhost:8080.
            api_key: Optional API key for Weaviate Cloud. If not provided,
                reads from WEAVIATE_API_KEY environment variable.
            collection_name: Name of the collection to use. Defaults to "RagnarokDocuments".
            timeout: Request timeout in seconds. Defaults to 30.0.
            grpc_host: Host of the gRPC endpoint of a self-hosted instance reached
                with an API key. Defaults to the host of *url*.
            grpc_port: Port of that gRPC endpoint. Defaults to 50051, Weaviate's default.
            grpc_secure: Whether that endpoint uses TLS. Defaults to the scheme of *url*
                (``https`` gives ``True``).

        The three gRPC settings are used for a self-hosted instance with an API key
        only. Weaviate Cloud and a local instance without a key find their gRPC
        endpoint themselves.
        """
        self.url = url.rstrip("/")
        self.api_key = api_key or os.environ.get("WEAVIATE_API_KEY")
        self.collection_name = collection_name
        self.timeout = timeout
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.grpc_secure = grpc_secure
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
                # A self-hosted instance with an API key. connect_to_custom requires the gRPC
                # endpoint next to the HTTP one: by default the same host and scheme, port 50051
                http_host = self.url.replace("http://", "").replace("https://", "").split(":")[0]
                http_secure = self.url.startswith("https")
                self._client = weaviate.connect_to_custom(
                    http_host=http_host,
                    http_port=int(self.url.split(":")[-1]) if ":" in self.url.split("/")[-1] else 8080,
                    http_secure=http_secure,
                    grpc_host=self.grpc_host or http_host,
                    grpc_port=self.grpc_port,
                    grpc_secure=http_secure if self.grpc_secure is None else self.grpc_secure,
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
            raise VectorStoreDependencyError(msg) from e
        except Exception as e:
            if not _connection_failure(e):
                raise  # a response of the server, a missing package, a bug: not a connection failure
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
            if not _connection_failure(e):
                raise  # a response of the server, a missing package, a bug: not a connection failure
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
            if not _connection_failure(e):
                raise  # a response of the server, a missing package, a bug: not a connection failure
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
            if not _connection_failure(e):
                raise  # a response of the server, a missing package, a bug: not a connection failure
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
            # The client first: without the package, its error carries the install message
            client = await self._ensure_client()

            from weaviate.classes.query import Filter

            collection = client.collections.get(self.collection_name)

            for doc_id in document_ids:
                collection.data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))

        except Exception as e:
            if not _connection_failure(e):
                raise  # a response of the server, a missing package, a bug: not a connection failure
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
            if not _connection_failure(e):
                raise  # a response of the server, a missing package, a bug: not a connection failure
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
            return bool(client.is_ready())
        except Exception as e:
            # False for an unreachable store only: anything else is not about availability
            if isinstance(e, VectorStoreConnectionError) or _connection_failure(e):
                return False
            raise
