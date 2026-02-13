"""Milvus vector store adapter for ragnarok-ai.

This module provides an async client for Milvus vector database,
implementing the VectorStoreProtocol for use in RAG pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.exceptions import VectorStoreConnectionError
from ragnarok_ai.core.types import Document

if TYPE_CHECKING:
    from types import TracebackType

# Default configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 19530
DEFAULT_COLLECTION_NAME = "ragnarok_documents"
DEFAULT_VECTOR_SIZE = 768
DEFAULT_TIMEOUT = 30.0


class MilvusVectorStore:
    """Async client for Milvus vector database.

    Implements VectorStoreProtocol for document storage and similarity search.
    Uses pymilvus for operations with automatic collection creation.

    Can be used as a context manager for optimal performance (reuses connections),
    or standalone for simpler use cases.

    Attributes:
        is_local: Always True - Milvus is typically self-hosted.
        host: Hostname for Milvus server.
        port: Port for Milvus server.
        collection_name: Name of the collection to use.
        vector_size: Dimension of embedding vectors.
        timeout: Request timeout in seconds.

    Example:
        Context manager (recommended for multiple calls):
            >>> async with MilvusVectorStore(vector_size=768) as store:
            ...     await store.add([doc1, doc2])
            ...     results = await store.search(query_embedding, k=5)

        Standalone (simpler, creates new connection per call):
            >>> store = MilvusVectorStore(vector_size=768)
            >>> await store.add([doc1])
    """

    is_local: bool = True

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize MilvusVectorStore client.

        Args:
            host: Hostname for Milvus server. Defaults to localhost.
            port: Port for Milvus server. Defaults to 19530.
            collection_name: Name of the collection to use. Defaults to "ragnarok_documents".
            vector_size: Dimension of embedding vectors. Defaults to 768.
            timeout: Request timeout in seconds. Defaults to 30.0.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout = timeout
        self._connected: bool = False
        self._collection: Any = None

    async def __aenter__(self) -> MilvusVectorStore:
        """Enter async context manager, creating a reusable Milvus connection."""
        await self._ensure_connection()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager, closing the Milvus connection."""
        if self._connected:
            try:
                from pymilvus import connections

                connections.disconnect("default")
            except Exception:
                pass
            self._connected = False
            self._collection = None

    async def _ensure_connection(self) -> Any:
        """Ensure we have an active Milvus connection.

        Returns:
            A Milvus Collection instance.

        Raises:
            VectorStoreConnectionError: If connection to Milvus fails.
        """
        if self._collection is not None:
            return self._collection

        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

            # Connect to Milvus
            connections.connect(
                alias="default",
                host=self.host,
                port=str(self.port),
                timeout=self.timeout,
            )
            self._connected = True

            # Ensure collection exists
            if not utility.has_collection(self.collection_name):
                # Define schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_size),
                    FieldSchema(name="metadata", dtype=DataType.JSON),
                ]
                schema = CollectionSchema(fields=fields, description="RAGnarok documents")
                self._collection = Collection(name=self.collection_name, schema=schema)

                # Create index for vector field
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 256},
                }
                self._collection.create_index(field_name="embedding", index_params=index_params)
            else:
                self._collection = Collection(name=self.collection_name)

            # Load collection into memory
            self._collection.load()

            return self._collection

        except ImportError as e:
            msg = "pymilvus is not installed. Install it with: pip install pymilvus"
            raise VectorStoreConnectionError(msg) from e
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to connect to Milvus at {self.host}:{self.port}: {e}"
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
            VectorStoreConnectionError: If connection to Milvus fails.
            ValueError: If query_embedding dimension doesn't match vector_size.
        """
        if len(query_embedding) != self.vector_size:
            msg = f"Query embedding dimension ({len(query_embedding)}) doesn't match vector_size ({self.vector_size})"
            raise ValueError(msg)

        try:
            collection = await self._ensure_connection()

            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 128},
            }

            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=k,
                output_fields=["id", "content", "metadata"],
            )

            documents: list[tuple[Document, float]] = []
            for hits in results:
                for hit in hits:
                    doc = Document(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content", ""),
                        metadata=hit.entity.get("metadata", {}),
                    )
                    # Cosine similarity from Milvus distance
                    score = 1.0 - hit.distance if hit.distance < 1.0 else hit.distance
                    documents.append((doc, score))

            return documents

        except ValueError:
            raise
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to search in Milvus: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def add(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Embeddings must be provided in document.metadata["embedding"].

        Args:
            documents: List of documents to add. Each document must have
                an "embedding" key in its metadata.

        Raises:
            VectorStoreConnectionError: If connection to Milvus fails.
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
            collection = await self._ensure_connection()

            # Prepare data for insertion
            ids = []
            contents = []
            embeddings = []
            metadatas = []

            for doc in documents:
                ids.append(doc.id)
                contents.append(doc.content)
                embeddings.append(doc.metadata["embedding"])
                # Store metadata without embedding
                metadatas.append({k: v for k, v in doc.metadata.items() if k != "embedding"})

            # Insert data
            collection.insert([ids, contents, embeddings, metadatas])
            collection.flush()

        except ValueError:
            raise
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to add documents to Milvus: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def delete(self, document_ids: list[str]) -> None:
        """Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete.

        Raises:
            VectorStoreConnectionError: If connection to Milvus fails.
        """
        if not document_ids:
            return

        try:
            collection = await self._ensure_connection()

            # Build expression for deletion
            ids_str = ", ".join(f'"{id}"' for id in document_ids)
            expr = f"id in [{ids_str}]"

            collection.delete(expr)
            collection.flush()

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to delete documents from Milvus: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def count(self) -> int:
        """Get the number of documents in the collection.

        Returns:
            The number of documents in the collection.

        Raises:
            VectorStoreConnectionError: If connection to Milvus fails.
        """
        try:
            collection = await self._ensure_connection()

            return collection.num_entities

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to get count from Milvus: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def is_available(self) -> bool:
        """Check if Milvus is available and responding.

        Returns:
            True if Milvus is reachable, False otherwise.

        Example:
            >>> if await store.is_available():
            ...     print("Milvus is ready")
        """
        try:
            await self._ensure_connection()
            return self._connected and self._collection is not None
        except Exception:
            return False
