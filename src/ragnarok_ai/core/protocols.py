"""Protocol definitions for ragnarok-ai.

This module defines the abstract interfaces (protocols) that adapters
must implement. Using protocols enables duck typing and loose coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ragnarok_ai.core.types import Document, RAGResponse


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM providers.

    Any class implementing these methods can be used as an LLM provider,
    without needing to inherit from a base class.

    Attributes:
        is_local: Whether the LLM runs locally (no data leaves the network).

    Example:
        >>> class MyLLM:
        ...     is_local: bool = True
        ...
        ...     async def generate(self, prompt: str) -> str:
        ...         return "Generated response"
        ...
        ...     async def embed(self, text: str) -> list[float]:
        ...         return [0.1, 0.2, 0.3]
        ...
        >>> assert isinstance(MyLLM(), LLMProtocol)
        >>> assert MyLLM().is_local
    """

    is_local: bool

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt for text generation.

        Returns:
            The generated text response.

        Raises:
            LLMConnectionError: If the LLM provider is unreachable.
        """
        ...

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            LLMConnectionError: If the LLM provider is unreachable.
        """
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector store providers.

    Any class implementing these methods can be used as a vector store,
    without needing to inherit from a base class.

    Attributes:
        is_local: Whether the vector store runs locally (no data leaves the network).

    Example:
        >>> class MyVectorStore:
        ...     is_local: bool = True
        ...
        ...     async def search(
        ...         self, query_embedding: list[float], k: int = 10
        ...     ) -> list[tuple[Document, float]]:
        ...         return []
        ...
        ...     async def add(self, documents: list[Document]) -> None:
        ...         pass
        ...
        >>> assert isinstance(MyVectorStore(), VectorStoreProtocol)
        >>> assert MyVectorStore().is_local
    """

    is_local: bool

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
            ConnectionError: If the vector store is unreachable.
        """
        ...

    async def add(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.

        Raises:
            ConnectionError: If the vector store is unreachable.
        """
        ...


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Protocol for metric evaluators.

    Any class implementing this method can be used as an evaluator,
    without needing to inherit from a base class.

    Example:
        >>> class MyEvaluator:
        ...     async def evaluate(
        ...         self, response: str, context: str, query: str | None = None
        ...     ) -> float:
        ...         return 0.85
        ...
        >>> assert isinstance(MyEvaluator(), EvaluatorProtocol)
    """

    async def evaluate(
        self,
        response: str,
        context: str,
        query: str | None = None,
    ) -> float:
        """Evaluate a response against its context.

        Args:
            response: The generated response to evaluate.
            context: The retrieved context used for generation.
            query: Optional original query for relevance evaluation.

        Returns:
            A score between 0.0 and 1.0, where 1.0 is the best.

        Raises:
            EvaluationError: If evaluation fails.
        """
        ...


@runtime_checkable
class RAGProtocol(Protocol):
    """Protocol for RAG pipeline implementations."""

    async def query(self, question: str) -> RAGResponse:
        """Execute RAG pipeline and return response with retrieved docs.

        Args:
            question: The question or query to answer.

        Returns:
            RAGResponse containing the answer and retrieved documents.
        """
        ...
