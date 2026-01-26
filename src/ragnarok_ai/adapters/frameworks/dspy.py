"""DSPy adapter for ragnarok-ai.

This module provides adapters to evaluate DSPy RAG pipelines.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ragnarok_ai.core.types import Document, RAGResponse


def _convert_passage_to_document(passage: Any, index: int) -> Document:
    """Convert a DSPy passage to a RAGnarok Document.

    DSPy passages can be strings, dicts, or objects with various attributes.

    Args:
        passage: DSPy passage (string, dict, or object).
        index: Index for ID generation if no ID available.

    Returns:
        RAGnarok Document.
    """
    # Handle string passages
    if isinstance(passage, str):
        return Document(
            id=str(hash(passage)),
            content=passage,
            metadata={"index": index},
        )

    # Handle dict passages
    if isinstance(passage, dict):
        doc_id = passage.get("id") or passage.get("pid") or str(index)
        content = passage.get("text") or passage.get("content") or passage.get("passage") or str(passage)
        metadata = {k: v for k, v in passage.items() if k not in ("id", "pid", "text", "content", "passage")}
        metadata["index"] = index
        return Document(id=str(doc_id), content=content, metadata=metadata)

    # Handle object passages (e.g., dspy.Example)
    if hasattr(passage, "text"):
        content = passage.text
    elif hasattr(passage, "content"):
        content = passage.content
    elif hasattr(passage, "passage"):
        content = passage.passage
    else:
        content = str(passage)

    doc_id = getattr(passage, "id", None) or getattr(passage, "pid", None) or str(index)

    # Extract metadata from object attributes
    obj_metadata: dict[str, Any] = {"index": index}
    for attr in ("title", "source", "score", "long_text"):
        if hasattr(passage, attr):
            obj_metadata[attr] = getattr(passage, attr)

    return Document(id=str(doc_id), content=content, metadata=obj_metadata)


def _convert_passages_to_documents(passages: list[Any]) -> list[Document]:
    """Convert a list of DSPy passages to RAGnarok Documents.

    Args:
        passages: List of DSPy passages.

    Returns:
        List of RAGnarok Documents.
    """
    return [_convert_passage_to_document(p, i) for i, p in enumerate(passages)]


class DSPyRetrieverAdapter:
    """Adapter for DSPy retrievers.

    Wraps a DSPy retriever (dspy.Retrieve or custom) for use with
    ragnarok-ai evaluation.

    Attributes:
        retriever: The wrapped DSPy retriever.

    Example:
        >>> import dspy
        >>> from ragnarok_ai.adapters.frameworks import DSPyRetrieverAdapter
        >>>
        >>> retriever = dspy.Retrieve(k=10)
        >>> adapter = DSPyRetrieverAdapter(retriever)
        >>>
        >>> # Use with evaluate()
        >>> results = await evaluate(adapter, testset)
    """

    def __init__(
        self,
        retriever: Any,
        *,
        answer_generator: Any | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            retriever: A DSPy retriever (dspy.Retrieve or custom module).
            answer_generator: Optional function to generate answers from
                             query and retrieved docs. If not provided,
                             returns a placeholder answer.
        """
        self._retriever = retriever
        self._answer_generator = answer_generator

    @property
    def retriever(self) -> Any:
        """Get the underlying DSPy retriever."""
        return self._retriever

    async def query(self, question: str) -> RAGResponse:
        """Execute retrieval and return response.

        Args:
            question: The question to retrieve passages for.

        Returns:
            RAGResponse with retrieved documents and answer.
        """
        # DSPy retrievers are typically sync
        result = await asyncio.to_thread(self._retriever, question)

        # Extract passages from result
        passages: list[Any] = []
        if hasattr(result, "passages"):
            passages = result.passages
        elif isinstance(result, list):
            passages = result
        elif hasattr(result, "__iter__"):
            passages = list(result)

        docs = _convert_passages_to_documents(passages)

        # Generate answer if generator provided
        if self._answer_generator:
            answer = self._answer_generator(question, docs)
        else:
            answer = f"Retrieved {len(docs)} passages for: {question}"

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            metadata={"adapter": "dspy_retriever"},
        )


class DSPyModuleAdapter:
    """Adapter for DSPy modules (RAG pipelines).

    Wraps a DSPy Module for use with ragnarok-ai evaluation.
    Automatically extracts the answer and retrieved passages.

    Attributes:
        module: The wrapped DSPy module.

    Example:
        >>> import dspy
        >>> from ragnarok_ai.adapters.frameworks import DSPyModuleAdapter
        >>>
        >>> class RAG(dspy.Module):
        ...     def __init__(self):
        ...         self.retrieve = dspy.Retrieve(k=3)
        ...         self.generate = dspy.ChainOfThought("context, question -> answer")
        ...
        ...     def forward(self, question):
        ...         context = self.retrieve(question).passages
        ...         return self.generate(context=context, question=question)
        >>>
        >>> rag = RAG()
        >>> adapter = DSPyModuleAdapter(rag)
        >>>
        >>> # Use with evaluate()
        >>> results = await evaluate(adapter, testset)

    Example with custom extraction:
        >>> adapter = DSPyModuleAdapter(
        ...     rag,
        ...     answer_field="response",
        ...     passages_field="retrieved_docs",
        ... )
    """

    def __init__(
        self,
        module: Any,
        *,
        answer_field: str = "answer",
        passages_field: str | None = None,
        input_field: str = "question",
    ) -> None:
        """Initialize the adapter.

        Args:
            module: A DSPy Module instance.
            answer_field: Field name to extract the answer from output.
            passages_field: Field name to extract passages from output.
                           If None, tries common field names.
            input_field: Field name to pass the question to the module.
        """
        self._module = module
        self._answer_field = answer_field
        self._passages_field = passages_field
        self._input_field = input_field

    @property
    def module(self) -> Any:
        """Get the underlying DSPy module."""
        return self._module

    def _extract_passages(self, result: Any) -> list[Any]:
        """Extract passages from DSPy result.

        Args:
            result: DSPy module output.

        Returns:
            List of passages.
        """
        # Try explicit field first
        if self._passages_field and hasattr(result, self._passages_field):
            value = getattr(result, self._passages_field)
            if isinstance(value, list):
                return value
            return [value] if value else []

        # Try common field names
        for field in ("passages", "context", "contexts", "retrieved", "docs", "documents"):
            if hasattr(result, field):
                value = getattr(result, field)
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    # Context might be a joined string
                    return [value]

        return []

    def _extract_answer(self, result: Any) -> str:
        """Extract answer from DSPy result.

        Args:
            result: DSPy module output.

        Returns:
            Answer string.
        """
        # Try explicit field first
        if hasattr(result, self._answer_field):
            return str(getattr(result, self._answer_field))

        # Try common field names
        for field in ("answer", "response", "output", "prediction", "rationale"):
            if hasattr(result, field):
                return str(getattr(result, field))

        # Fallback to string representation
        return str(result)

    async def query(self, question: str) -> RAGResponse:
        """Execute the DSPy module and return response.

        Args:
            question: The question to answer.

        Returns:
            RAGResponse with answer and retrieved documents.
        """
        # DSPy modules are typically sync
        # Call with keyword argument
        kwargs = {self._input_field: question}
        result = await asyncio.to_thread(self._module, **kwargs)

        answer = self._extract_answer(result)
        passages = self._extract_passages(result)
        docs = _convert_passages_to_documents(passages)

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            metadata={
                "adapter": "dspy_module",
                "module_type": type(self._module).__name__,
            },
        )


class DSPyRAGAdapter:
    """Simplified adapter for DSPy RAG patterns.

    This adapter is designed for the common DSPy RAG pattern where
    a retriever and generator are combined.

    Example:
        >>> import dspy
        >>> from ragnarok_ai.adapters.frameworks import DSPyRAGAdapter
        >>>
        >>> # Configure DSPy
        >>> dspy.configure(lm=lm, rm=retriever_model)
        >>>
        >>> # Create RAG with built-in module
        >>> rag = dspy.ChainOfThought("context, question -> answer")
        >>> retriever = dspy.Retrieve(k=5)
        >>>
        >>> adapter = DSPyRAGAdapter(retriever, rag)
        >>> results = await evaluate(adapter, testset)
    """

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        *,
        answer_field: str = "answer",
    ) -> None:
        """Initialize the adapter.

        Args:
            retriever: DSPy retriever (dspy.Retrieve or custom).
            generator: DSPy generator module (dspy.Predict, ChainOfThought, etc.).
            answer_field: Field name to extract answer from generator output.
        """
        self._retriever = retriever
        self._generator = generator
        self._answer_field = answer_field

    @property
    def retriever(self) -> Any:
        """Get the DSPy retriever."""
        return self._retriever

    @property
    def generator(self) -> Any:
        """Get the DSPy generator."""
        return self._generator

    async def query(self, question: str) -> RAGResponse:
        """Execute RAG pipeline and return response.

        Args:
            question: The question to answer.

        Returns:
            RAGResponse with answer and retrieved documents.
        """
        # Step 1: Retrieve
        retrieval_result = await asyncio.to_thread(self._retriever, question)

        # Extract passages
        if hasattr(retrieval_result, "passages"):
            passages = retrieval_result.passages
        elif isinstance(retrieval_result, list):
            passages = retrieval_result
        else:
            passages = []

        docs = _convert_passages_to_documents(passages)

        # Step 2: Generate
        # Join passages as context
        if passages:
            if all(isinstance(p, str) for p in passages):
                context = "\n\n".join(passages)
            else:
                context = "\n\n".join(
                    getattr(p, "text", None) or getattr(p, "content", None) or str(p) for p in passages
                )
        else:
            context = ""

        gen_result = await asyncio.to_thread(self._generator, context=context, question=question)

        # Extract answer
        if hasattr(gen_result, self._answer_field):
            answer = str(getattr(gen_result, self._answer_field))
        else:
            answer = str(gen_result)

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            metadata={"adapter": "dspy_rag"},
        )


def create_dspy_adapter(
    obj: Any,
    **kwargs: Any,
) -> DSPyRetrieverAdapter | DSPyModuleAdapter:
    """Factory function to create the appropriate DSPy adapter.

    Attempts to detect the type of DSPy object and returns
    the appropriate adapter.

    Args:
        obj: A DSPy Module, Retriever, or other callable.
        **kwargs: Additional arguments passed to the adapter constructor.

    Returns:
        Appropriate DSPy adapter.

    Example:
        >>> adapter = create_dspy_adapter(retriever)  # Returns DSPyRetrieverAdapter
        >>> adapter = create_dspy_adapter(rag_module)  # Returns DSPyModuleAdapter
    """
    try:
        import dspy
    except ImportError as e:
        msg = "dspy is required for DSPy adapters. Install with: pip install dspy-ai"
        raise ImportError(msg) from e

    # Check if it's a Retrieve module
    if isinstance(obj, dspy.Retrieve):
        retriever_kwargs = {k: v for k, v in kwargs.items() if k in ("answer_generator",)}
        return DSPyRetrieverAdapter(obj, **retriever_kwargs)

    # Default to module adapter
    module_kwargs = {k: v for k, v in kwargs.items() if k in ("answer_field", "passages_field", "input_field")}
    return DSPyModuleAdapter(obj, **module_kwargs)
