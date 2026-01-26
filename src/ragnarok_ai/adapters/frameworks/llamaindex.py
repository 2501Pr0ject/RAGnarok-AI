"""LlamaIndex adapter for ragnarok-ai.

This module provides adapters to evaluate LlamaIndex RAG pipelines.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.types import Document, RAGResponse

if TYPE_CHECKING:
    from llama_index.core.base.base_query_engine import BaseQueryEngine
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.indices.base import BaseIndex
    from llama_index.core.schema import NodeWithScore


def _convert_node_to_document(node: NodeWithScore) -> Document:
    """Convert a LlamaIndex NodeWithScore to a RAGnarok Document.

    Args:
        node: LlamaIndex NodeWithScore.

    Returns:
        RAGnarok Document.
    """
    node_obj = node.node
    doc_id = node_obj.node_id or node_obj.id_ or str(hash(node_obj.get_content()))

    return Document(
        id=str(doc_id),
        content=node_obj.get_content(),
        metadata={
            **node_obj.metadata,
            "score": node.score,
        },
    )


def _convert_nodes_to_documents(nodes: list[NodeWithScore]) -> list[Document]:
    """Convert a list of LlamaIndex nodes to RAGnarok Documents.

    Args:
        nodes: List of LlamaIndex NodeWithScore.

    Returns:
        List of RAGnarok Documents.
    """
    return [_convert_node_to_document(node) for node in nodes]


class LlamaIndexRetrieverAdapter:
    """Adapter for LlamaIndex retrievers.

    Wraps a LlamaIndex BaseRetriever for use with ragnarok-ai evaluation.
    This is useful when you only want to evaluate retrieval quality
    without a full query engine.

    Attributes:
        retriever: The wrapped LlamaIndex retriever.

    Example:
        >>> from llama_index.core import VectorStoreIndex
        >>> from ragnarok_ai.adapters.frameworks import LlamaIndexRetrieverAdapter
        >>>
        >>> index = VectorStoreIndex.from_documents(documents)
        >>> retriever = index.as_retriever(similarity_top_k=10)
        >>> adapter = LlamaIndexRetrieverAdapter(retriever)
        >>>
        >>> # Use with evaluate()
        >>> results = await evaluate(adapter, testset)
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        answer_generator: Any | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            retriever: A LlamaIndex BaseRetriever instance.
            answer_generator: Optional function to generate answers from
                             query and retrieved docs. If not provided,
                             returns a placeholder answer.
        """
        self._retriever = retriever
        self._answer_generator = answer_generator

    @property
    def retriever(self) -> BaseRetriever:
        """Get the underlying LlamaIndex retriever."""
        return self._retriever

    async def query(self, question: str) -> RAGResponse:
        """Execute retrieval and return response.

        Args:
            question: The question to retrieve documents for.

        Returns:
            RAGResponse with retrieved documents and answer.
        """
        # LlamaIndex retrievers may be sync or async
        if hasattr(self._retriever, "aretrieve"):
            nodes = await self._retriever.aretrieve(question)
        else:
            nodes = await asyncio.to_thread(self._retriever.retrieve, question)

        docs = _convert_nodes_to_documents(nodes)

        # Generate answer if generator provided
        if self._answer_generator:
            answer = self._answer_generator(question, docs)
        else:
            answer = f"Retrieved {len(docs)} documents for: {question}"

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            metadata={"adapter": "llamaindex_retriever"},
        )


class LlamaIndexQueryEngineAdapter:
    """Adapter for LlamaIndex query engines.

    Wraps a LlamaIndex QueryEngine for use with ragnarok-ai evaluation.
    Automatically extracts the response and source nodes.

    Attributes:
        query_engine: The wrapped LlamaIndex query engine.

    Example:
        >>> from llama_index.core import VectorStoreIndex
        >>> from ragnarok_ai.adapters.frameworks import LlamaIndexQueryEngineAdapter
        >>>
        >>> index = VectorStoreIndex.from_documents(documents)
        >>> query_engine = index.as_query_engine()
        >>> adapter = LlamaIndexQueryEngineAdapter(query_engine)
        >>>
        >>> # Use with evaluate()
        >>> results = await evaluate(adapter, testset)
    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
    ) -> None:
        """Initialize the adapter.

        Args:
            query_engine: A LlamaIndex BaseQueryEngine instance.
        """
        self._query_engine = query_engine

    @property
    def query_engine(self) -> BaseQueryEngine:
        """Get the underlying LlamaIndex query engine."""
        return self._query_engine

    async def query(self, question: str) -> RAGResponse:
        """Execute the query engine and return response.

        Args:
            question: The question to answer.

        Returns:
            RAGResponse with answer and retrieved documents.
        """
        # LlamaIndex query engines may be sync or async
        if hasattr(self._query_engine, "aquery"):
            response = await self._query_engine.aquery(question)
        else:
            response = await asyncio.to_thread(self._query_engine.query, question)

        # Extract answer
        answer = str(response)

        # Extract source nodes if available
        docs: list[Document] = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            docs = _convert_nodes_to_documents(response.source_nodes)

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            metadata={"adapter": "llamaindex_query_engine"},
        )


class LlamaIndexAdapter:
    """Adapter for LlamaIndex indexes.

    Wraps a LlamaIndex Index directly. Creates a query engine internally.
    This is the simplest way to adapt a LlamaIndex index.

    Attributes:
        index: The wrapped LlamaIndex index.

    Example:
        >>> from llama_index.core import VectorStoreIndex
        >>> from ragnarok_ai.adapters.frameworks import LlamaIndexAdapter
        >>>
        >>> index = VectorStoreIndex.from_documents(documents)
        >>> adapter = LlamaIndexAdapter(index)
        >>>
        >>> # Use with evaluate()
        >>> results = await evaluate(adapter, testset)

    Example with custom settings:
        >>> adapter = LlamaIndexAdapter(
        ...     index,
        ...     similarity_top_k=5,
        ...     response_mode="tree_summarize",
        ... )
    """

    def __init__(
        self,
        index: BaseIndex,
        *,
        similarity_top_k: int = 10,
        response_mode: str = "compact",
        **query_engine_kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            index: A LlamaIndex BaseIndex instance.
            similarity_top_k: Number of top similar nodes to retrieve.
            response_mode: Response synthesis mode.
                          Options: "refine", "compact", "tree_summarize", "simple_summarize".
            **query_engine_kwargs: Additional kwargs for query engine creation.
        """
        self._index = index
        self._query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode=response_mode,
            **query_engine_kwargs,
        )

    @property
    def index(self) -> BaseIndex:
        """Get the underlying LlamaIndex index."""
        return self._index

    @property
    def query_engine(self) -> BaseQueryEngine:
        """Get the query engine created from the index."""
        return self._query_engine

    async def query(self, question: str) -> RAGResponse:
        """Execute the query engine and return response.

        Args:
            question: The question to answer.

        Returns:
            RAGResponse with answer and retrieved documents.
        """
        # Delegate to query engine
        if hasattr(self._query_engine, "aquery"):
            response = await self._query_engine.aquery(question)
        else:
            response = await asyncio.to_thread(self._query_engine.query, question)

        # Extract answer
        answer = str(response)

        # Extract source nodes if available
        docs: list[Document] = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            docs = _convert_nodes_to_documents(response.source_nodes)

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            metadata={
                "adapter": "llamaindex",
                "index_type": type(self._index).__name__,
            },
        )


def create_llamaindex_adapter(
    obj: BaseIndex | BaseQueryEngine | BaseRetriever,
    **kwargs: Any,
) -> LlamaIndexAdapter | LlamaIndexQueryEngineAdapter | LlamaIndexRetrieverAdapter:
    """Factory function to create the appropriate LlamaIndex adapter.

    Automatically detects the type of LlamaIndex object and returns
    the appropriate adapter.

    Args:
        obj: A LlamaIndex Index, QueryEngine, or Retriever.
        **kwargs: Additional arguments passed to the adapter constructor.

    Returns:
        Appropriate LlamaIndex adapter.

    Example:
        >>> adapter = create_llamaindex_adapter(index)  # Returns LlamaIndexAdapter
        >>> adapter = create_llamaindex_adapter(query_engine)  # Returns LlamaIndexQueryEngineAdapter
        >>> adapter = create_llamaindex_adapter(retriever)  # Returns LlamaIndexRetrieverAdapter
    """
    try:
        from llama_index.core.base.base_query_engine import BaseQueryEngine
        from llama_index.core.base.base_retriever import BaseRetriever
        from llama_index.core.indices.base import BaseIndex
    except ImportError as e:
        msg = "llama-index-core is required for LlamaIndex adapters. Install with: pip install llama-index-core"
        raise ImportError(msg) from e

    if isinstance(obj, BaseRetriever):
        retriever_kwargs = {k: v for k, v in kwargs.items() if k in ("answer_generator",)}
        return LlamaIndexRetrieverAdapter(obj, **retriever_kwargs)

    if isinstance(obj, BaseQueryEngine):
        return LlamaIndexQueryEngineAdapter(obj)

    if isinstance(obj, BaseIndex):
        return LlamaIndexAdapter(obj, **kwargs)

    msg = f"Unsupported LlamaIndex object type: {type(obj)}"
    raise TypeError(msg)
