"""Haystack adapter for ragnarok-ai.

This module provides adapters to evaluate Haystack RAG pipelines.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.types import Document, RAGResponse

if TYPE_CHECKING:
    from haystack import Pipeline
    from haystack.dataclasses import Document as HaystackDocument


def _convert_haystack_document(hs_doc: HaystackDocument) -> Document:
    """Convert a Haystack Document to a RAGnarok Document.

    Args:
        hs_doc: Haystack Document.

    Returns:
        RAGnarok Document.
    """
    # Haystack 2.x Document has id, content, and meta attributes
    doc_id = hs_doc.id or str(hash(hs_doc.content or ""))

    return Document(
        id=str(doc_id),
        content=hs_doc.content or "",
        metadata=hs_doc.meta or {},
    )


def _convert_haystack_documents(hs_docs: list[HaystackDocument]) -> list[Document]:
    """Convert a list of Haystack Documents to RAGnarok Documents.

    Args:
        hs_docs: List of Haystack Documents.

    Returns:
        List of RAGnarok Documents.
    """
    return [_convert_haystack_document(doc) for doc in hs_docs]


class HaystackAdapter:
    """Adapter for Haystack 2.x pipelines.

    Wraps a Haystack Pipeline for use with ragnarok-ai evaluation.
    Supports both RAG pipelines and retrieval-only pipelines.

    Attributes:
        pipeline: The wrapped Haystack pipeline.
        is_local: Always True - Haystack pipelines run locally.

    Example:
        >>> from haystack import Pipeline
        >>> from haystack.components.retrievers import InMemoryBM25Retriever
        >>> from haystack.components.generators import OpenAIGenerator
        >>> from ragnarok_ai.adapters.frameworks import HaystackAdapter
        >>>
        >>> pipeline = Pipeline()
        >>> pipeline.add_component("retriever", InMemoryBM25Retriever(document_store))
        >>> pipeline.add_component("generator", OpenAIGenerator())
        >>> pipeline.connect("retriever", "generator")
        >>>
        >>> adapter = HaystackAdapter(pipeline)
        >>> results = await evaluate(adapter, testset)

    Example with custom keys:
        >>> adapter = HaystackAdapter(
        ...     pipeline,
        ...     query_key="query",
        ...     answer_key="replies",
        ...     docs_key="documents",
        ... )
    """

    is_local: bool = True

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        query_key: str = "query",
        answer_key: str = "replies",
        docs_key: str = "documents",
        answer_component: str | None = None,
        docs_component: str | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            pipeline: A Haystack Pipeline instance.
            query_key: Input key for the query. Defaults to "query".
            answer_key: Output key for the generated answer. Defaults to "replies".
            docs_key: Output key for retrieved documents. Defaults to "documents".
            answer_component: Component name to get answer from. If None, searches
                all components for the answer_key.
            docs_component: Component name to get documents from. If None, searches
                all components for the docs_key.
        """
        self._pipeline = pipeline
        self._query_key = query_key
        self._answer_key = answer_key
        self._docs_key = docs_key
        self._answer_component = answer_component
        self._docs_component = docs_component

    @property
    def pipeline(self) -> Pipeline:
        """Get the underlying Haystack pipeline."""
        return self._pipeline

    def _extract_from_result(
        self,
        result: dict[str, Any],
        key: str,
        component: str | None = None,
    ) -> Any:
        """Extract a value from pipeline result.

        Args:
            result: Pipeline output dictionary.
            key: Key to look for.
            component: Specific component to look in.

        Returns:
            Extracted value or None.
        """
        if component and component in result:
            return result[component].get(key)

        # Search all components
        for comp_result in result.values():
            if isinstance(comp_result, dict) and key in comp_result:
                return comp_result[key]

        return None

    def _extract_answer(self, result: dict[str, Any]) -> str:
        """Extract answer from pipeline result.

        Args:
            result: Pipeline output dictionary.

        Returns:
            Extracted answer string.
        """
        raw_answer = self._extract_from_result(
            result,
            self._answer_key,
            self._answer_component,
        )

        if raw_answer is None:
            return ""

        # Handle list of replies (common in Haystack generators)
        if isinstance(raw_answer, list):
            return str(raw_answer[0]) if raw_answer else ""

        return str(raw_answer)

    def _extract_documents(self, result: dict[str, Any]) -> list[Document]:
        """Extract documents from pipeline result.

        Args:
            result: Pipeline output dictionary.

        Returns:
            List of RAGnarok Documents.
        """
        raw_docs = self._extract_from_result(
            result,
            self._docs_key,
            self._docs_component,
        )

        if not raw_docs:
            return []

        # Check if these are Haystack documents
        if hasattr(raw_docs[0], "content"):
            return _convert_haystack_documents(raw_docs)

        # Handle dict format
        if isinstance(raw_docs[0], dict):
            return [
                Document(
                    id=str(d.get("id", i)),
                    content=str(d.get("content", "")),
                    metadata=d.get("meta", d.get("metadata", {})),
                )
                for i, d in enumerate(raw_docs)
            ]

        return []

    async def query(self, question: str) -> RAGResponse:
        """Execute the Haystack pipeline and return response.

        Args:
            question: The question to answer.

        Returns:
            RAGResponse with answer and retrieved documents.
        """
        # Haystack pipelines are synchronous
        result = await asyncio.to_thread(
            self._pipeline.run,
            {self._query_key: question},
        )

        answer = self._extract_answer(result)
        docs = self._extract_documents(result)

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            metadata={"adapter": "haystack"},
        )


class HaystackRetrieverAdapter:
    """Adapter for Haystack retrievers (retrieval-only evaluation).

    Wraps a Haystack retriever component for use with ragnarok-ai
    retrieval evaluation. Use this when you only want to evaluate
    retrieval quality without generation.

    Attributes:
        retriever: The wrapped Haystack retriever.
        is_local: Always True - runs locally.

    Example:
        >>> from haystack.components.retrievers import InMemoryBM25Retriever
        >>> from ragnarok_ai.adapters.frameworks import HaystackRetrieverAdapter
        >>>
        >>> retriever = InMemoryBM25Retriever(document_store)
        >>> adapter = HaystackRetrieverAdapter(retriever)
        >>> results = await evaluate(adapter, testset)
    """

    is_local: bool = True

    def __init__(
        self,
        retriever: Any,
        *,
        query_key: str = "query",
        docs_key: str = "documents",
    ) -> None:
        """Initialize the adapter.

        Args:
            retriever: A Haystack retriever component.
            query_key: Input key for the query. Defaults to "query".
            docs_key: Output key for documents. Defaults to "documents".
        """
        self._retriever = retriever
        self._query_key = query_key
        self._docs_key = docs_key

    @property
    def retriever(self) -> Any:
        """Get the underlying Haystack retriever."""
        return self._retriever

    async def query(self, question: str) -> RAGResponse:
        """Execute retrieval and return response.

        Args:
            question: The question to retrieve documents for.

        Returns:
            RAGResponse with retrieved documents.
        """
        # Run retriever
        result = await asyncio.to_thread(
            self._retriever.run,
            **{self._query_key: question},
        )

        # Extract documents
        raw_docs = result.get(self._docs_key, [])
        docs: list[Document] = []

        if raw_docs and hasattr(raw_docs[0], "content"):
            docs = _convert_haystack_documents(raw_docs)

        return RAGResponse(
            answer=f"Retrieved {len(docs)} documents for: {question}",
            retrieved_docs=docs,
            metadata={"adapter": "haystack_retriever"},
        )
