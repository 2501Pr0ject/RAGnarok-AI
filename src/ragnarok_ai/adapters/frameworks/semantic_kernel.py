"""Semantic Kernel adapter for ragnarok-ai.

This module provides adapters to evaluate Microsoft Semantic Kernel RAG pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.types import Document, RAGResponse

if TYPE_CHECKING:
    from collections.abc import Callable

    from semantic_kernel import Kernel
    from semantic_kernel.functions import KernelFunction


class SemanticKernelAdapter:
    """Adapter for Microsoft Semantic Kernel.

    Wraps a Semantic Kernel function or plugin for use with ragnarok-ai evaluation.
    Supports both simple functions and RAG pipelines with memory/retrieval.

    Attributes:
        kernel: The Semantic Kernel instance.
        is_local: Always True - Semantic Kernel runs locally.

    Example:
        >>> from semantic_kernel import Kernel
        >>> from ragnarok_ai.adapters.frameworks import SemanticKernelAdapter
        >>>
        >>> kernel = Kernel()
        >>> kernel.add_plugin(rag_plugin, "rag")
        >>>
        >>> adapter = SemanticKernelAdapter(
        ...     kernel,
        ...     function_name="answer_question",
        ...     plugin_name="rag",
        ... )
        >>> results = await evaluate(adapter, testset)

    Example with custom function:
        >>> @kernel.function
        >>> async def my_rag_function(question: str) -> str:
        ...     # Your RAG logic
        ...     return answer
        >>>
        >>> adapter = SemanticKernelAdapter(kernel, function=my_rag_function)
    """

    is_local: bool = True

    def __init__(
        self,
        kernel: Kernel,
        *,
        function: KernelFunction | None = None,
        function_name: str | None = None,
        plugin_name: str | None = None,
        input_key: str = "question",
        docs_extractor: Callable[[Any], list[Document]] | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            kernel: A Semantic Kernel instance.
            function: A KernelFunction to invoke. If not provided,
                function_name and plugin_name must be set.
            function_name: Name of the function to invoke.
            plugin_name: Name of the plugin containing the function.
            input_key: Input variable name for the question. Defaults to "question".
            docs_extractor: Optional function to extract documents from the result.
                If not provided, no documents will be returned.
        """
        self._kernel = kernel
        self._function = function
        self._function_name = function_name
        self._plugin_name = plugin_name
        self._input_key = input_key
        self._docs_extractor = docs_extractor

        # Validate configuration
        if function is None and function_name is None:
            msg = "Either 'function' or 'function_name' must be provided"
            raise ValueError(msg)

    @property
    def kernel(self) -> Kernel:
        """Get the underlying Semantic Kernel instance."""
        return self._kernel

    def _get_function(self) -> KernelFunction:
        """Get the function to invoke.

        Returns:
            The KernelFunction to invoke.

        Raises:
            ValueError: If function cannot be found.
        """
        if self._function is not None:
            return self._function

        # Get function from kernel
        if self._plugin_name:
            plugin = self._kernel.get_plugin(self._plugin_name)
            if plugin is None:
                msg = f"Plugin '{self._plugin_name}' not found in kernel"
                raise ValueError(msg)
            func = plugin.get(self._function_name)
        else:
            # Try to get function directly
            func = self._kernel.get_function(None, self._function_name)

        if func is None:
            msg = f"Function '{self._function_name}' not found"
            raise ValueError(msg)

        return func

    async def query(self, question: str) -> RAGResponse:
        """Execute the Semantic Kernel function and return response.

        Args:
            question: The question to answer.

        Returns:
            RAGResponse with answer and retrieved documents.
        """
        func = self._get_function()

        # Invoke the function
        result = await self._kernel.invoke(
            func,
            **{self._input_key: question},
        )

        # Extract answer from result
        answer = str(result) if result else ""

        # Extract documents if extractor provided
        docs: list[Document] = []
        if self._docs_extractor:
            docs = self._docs_extractor(result)

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            metadata={"adapter": "semantic_kernel"},
        )


class SemanticKernelMemoryAdapter:
    """Adapter for Semantic Kernel with Memory (retrieval-focused).

    Wraps Semantic Kernel's memory capabilities for retrieval evaluation.
    Use this when you want to evaluate the retrieval quality of
    Semantic Kernel's memory search.

    Attributes:
        kernel: The Semantic Kernel instance.
        is_local: Always True - runs locally.

    Example:
        >>> from semantic_kernel import Kernel
        >>> from semantic_kernel.memory import SemanticTextMemory
        >>> from ragnarok_ai.adapters.frameworks import SemanticKernelMemoryAdapter
        >>>
        >>> kernel = Kernel()
        >>> memory = SemanticTextMemory(storage, embeddings)
        >>>
        >>> adapter = SemanticKernelMemoryAdapter(
        ...     kernel,
        ...     memory=memory,
        ...     collection="documents",
        ... )
        >>> results = await evaluate(adapter, testset)
    """

    is_local: bool = True

    def __init__(
        self,
        kernel: Kernel,
        *,
        memory: Any = None,
        collection: str = "default",
        limit: int = 10,
        min_relevance: float = 0.0,
        answer_generator: Callable[[str, list[Document]], str] | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            kernel: A Semantic Kernel instance.
            memory: A SemanticTextMemory instance. If not provided,
                attempts to get memory from the kernel.
            collection: Memory collection name. Defaults to "default".
            limit: Maximum number of results to retrieve. Defaults to 10.
            min_relevance: Minimum relevance score (0-1). Defaults to 0.0.
            answer_generator: Optional function to generate answers from
                query and retrieved docs.
        """
        self._kernel = kernel
        self._memory = memory
        self._collection = collection
        self._limit = limit
        self._min_relevance = min_relevance
        self._answer_generator = answer_generator

    @property
    def kernel(self) -> Kernel:
        """Get the underlying Semantic Kernel instance."""
        return self._kernel

    async def query(self, question: str) -> RAGResponse:
        """Execute memory search and return response.

        Args:
            question: The question to search for.

        Returns:
            RAGResponse with retrieved documents.
        """
        memory = self._memory

        if memory is None:
            msg = "No memory configured. Provide 'memory' parameter."
            raise ValueError(msg)

        # Search memory
        results = await memory.search(
            collection=self._collection,
            query=question,
            limit=self._limit,
            min_relevance_score=self._min_relevance,
        )

        # Convert to RAGnarok documents
        docs: list[Document] = []
        for i, result in enumerate(results):
            doc = Document(
                id=result.id if hasattr(result, "id") else str(i),
                content=result.text if hasattr(result, "text") else str(result),
                metadata={
                    "relevance": result.relevance if hasattr(result, "relevance") else 0.0,
                    **(result.metadata if hasattr(result, "metadata") else {}),
                },
            )
            docs.append(doc)

        # Generate answer
        if self._answer_generator:
            answer = self._answer_generator(question, docs)
        else:
            answer = f"Retrieved {len(docs)} documents for: {question}"

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            metadata={"adapter": "semantic_kernel_memory"},
        )
