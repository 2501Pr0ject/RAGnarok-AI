"""pgvector PostgreSQL vector store adapter for ragnarok-ai.

This module provides an async client for PostgreSQL with pgvector extension,
implementing the VectorStoreProtocol for use in RAG pipelines.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.exceptions import VectorStoreConnectionError
from ragnarok_ai.core.types import Document

if TYPE_CHECKING:
    from types import TracebackType

# Default configuration
DEFAULT_TABLE_NAME = "ragnarok_documents"
DEFAULT_VECTOR_SIZE = 768
DEFAULT_TIMEOUT = 30.0


class PgvectorVectorStore:
    """Async client for PostgreSQL with pgvector extension.

    Implements VectorStoreProtocol for document storage and similarity search.
    Uses asyncpg for async operations with automatic table creation.

    Can be used as a context manager for optimal performance (reuses connections),
    or standalone for simpler use cases.

    Attributes:
        is_local: Always True - PostgreSQL is typically self-hosted.
        connection_string: PostgreSQL connection string.
        table_name: Name of the table to use.
        vector_size: Dimension of embedding vectors.
        timeout: Request timeout in seconds.

    Example:
        Context manager (recommended for multiple calls):
            >>> async with PgvectorVectorStore(
            ...     connection_string="postgresql://user:pass@localhost:5432/db",
            ...     vector_size=768,
            ... ) as store:
            ...     await store.add([doc1, doc2])
            ...     results = await store.search(query_embedding, k=5)

        Standalone:
            >>> store = PgvectorVectorStore(connection_string="...")
            >>> await store.add([doc1])
    """

    is_local: bool = True

    def __init__(
        self,
        connection_string: str | None = None,
        table_name: str = DEFAULT_TABLE_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize PgvectorVectorStore client.

        Args:
            connection_string: PostgreSQL connection string. If not provided,
                reads from DATABASE_URL environment variable.
            table_name: Name of the table to use. Defaults to "ragnarok_documents".
            vector_size: Dimension of embedding vectors. Defaults to 768.
            timeout: Request timeout in seconds. Defaults to 30.0.

        Raises:
            ValueError: If connection string is not provided.
        """
        self.connection_string = connection_string or os.environ.get("DATABASE_URL")
        if not self.connection_string:
            msg = "Connection string required. Set DATABASE_URL or pass connection_string parameter."
            raise ValueError(msg)

        self.table_name = table_name
        self.vector_size = vector_size
        self.timeout = timeout
        self._pool: Any = None

    async def __aenter__(self) -> PgvectorVectorStore:
        """Enter async context manager, creating a reusable connection pool."""
        await self._ensure_pool()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager, closing the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _ensure_pool(self) -> Any:
        """Ensure we have an active connection pool.

        Returns:
            An asyncpg connection pool.

        Raises:
            VectorStoreConnectionError: If connection to PostgreSQL fails.
        """
        if self._pool is not None:
            return self._pool

        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self.connection_string,
                command_timeout=self.timeout,
            )

            # Ensure pgvector extension and table exist
            await self._ensure_table()

            return self._pool

        except ImportError as e:
            msg = "asyncpg is not installed. Install it with: pip install asyncpg pgvector"
            raise VectorStoreConnectionError(msg) from e
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to connect to PostgreSQL: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def _ensure_table(self) -> None:
        """Ensure the pgvector extension and table exist."""
        if self._pool is None:
            return

        try:
            async with self._pool.acquire() as conn:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Create table if not exists
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id VARCHAR(256) PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector({self.vector_size}) NOT NULL,
                        metadata JSONB DEFAULT '{{}}'::jsonb
                    )
                """)

                # Create index for vector similarity search
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                    ON {self.table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)

        except Exception as e:
            msg = f"Failed to create pgvector table: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def search(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents using cosine similarity.

        Args:
            query_embedding: The embedding vector to search with.
            k: Number of results to return. Defaults to 10.

        Returns:
            A list of tuples containing (document, similarity_score).
            Results are sorted by similarity in descending order.

        Raises:
            VectorStoreConnectionError: If connection to PostgreSQL fails.
            ValueError: If query_embedding dimension doesn't match vector_size.
        """
        if len(query_embedding) != self.vector_size:
            msg = f"Query embedding dimension ({len(query_embedding)}) doesn't match vector_size ({self.vector_size})"
            raise ValueError(msg)

        try:
            pool = await self._ensure_pool()

            # Convert embedding to pgvector format
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT id, content, metadata,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM {self.table_name}
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    embedding_str,
                    k,
                )

            documents: list[tuple[Document, float]] = []
            for row in rows:
                metadata = row["metadata"] if row["metadata"] else {}
                doc = Document(
                    id=row["id"],
                    content=row["content"],
                    metadata=dict(metadata),
                )
                documents.append((doc, float(row["similarity"])))

            return documents

        except ValueError:
            raise
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to search in pgvector: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def add(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Embeddings must be provided in document.metadata["embedding"].

        Args:
            documents: List of documents to add. Each document must have
                an "embedding" key in its metadata.

        Raises:
            VectorStoreConnectionError: If connection to PostgreSQL fails.
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
            pool = await self._ensure_pool()

            async with pool.acquire() as conn:
                # Use upsert (INSERT ... ON CONFLICT)
                for doc in documents:
                    embedding = doc.metadata["embedding"]
                    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                    # Store metadata without embedding
                    metadata = {k: v for k, v in doc.metadata.items() if k != "embedding"}

                    await conn.execute(
                        f"""
                        INSERT INTO {self.table_name} (id, content, embedding, metadata)
                        VALUES ($1, $2, $3::vector, $4::jsonb)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata
                        """,
                        doc.id,
                        doc.content,
                        embedding_str,
                        json.dumps(metadata),
                    )

        except ValueError:
            raise
        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to add documents to pgvector: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def delete(self, document_ids: list[str]) -> None:
        """Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete.

        Raises:
            VectorStoreConnectionError: If connection to PostgreSQL fails.
        """
        if not document_ids:
            return

        try:
            pool = await self._ensure_pool()

            async with pool.acquire() as conn:
                await conn.execute(
                    f"DELETE FROM {self.table_name} WHERE id = ANY($1)",
                    document_ids,
                )

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to delete documents from pgvector: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def count(self) -> int:
        """Get the number of documents in the table.

        Returns:
            The number of documents in the table.

        Raises:
            VectorStoreConnectionError: If connection to PostgreSQL fails.
        """
        try:
            pool = await self._ensure_pool()

            async with pool.acquire() as conn:
                row = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {self.table_name}")
                return int(row["count"]) if row else 0

        except Exception as e:
            if isinstance(e, VectorStoreConnectionError):
                raise
            msg = f"Failed to get count from pgvector: {e}"
            raise VectorStoreConnectionError(msg) from e

    async def is_available(self) -> bool:
        """Check if PostgreSQL with pgvector is available and responding.

        Returns:
            True if PostgreSQL is reachable, False otherwise.

        Example:
            >>> if await store.is_available():
            ...     print("pgvector is ready")
        """
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False
