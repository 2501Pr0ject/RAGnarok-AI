"""LangChain adapter for ragnarok-ai.

This module provides adapters to evaluate LangChain RAG pipelines.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.types import Document, RAGResponse

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.documents import Document as LCDocument
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables import Runnable


def _convert_lc_document(lc_doc: LCDocument) -> Document:
    """Convert a LangChain Document to a RAGnarok Document.

    Args:
        lc_doc: LangChain Document.

    Returns:
        RAGnarok Document.
    """
    # Generate ID from metadata or content hash
    doc_id = lc_doc.metadata.get("id") or lc_doc.metadata.get("source") or str(hash(lc_doc.page_content))

    return Document(
        id=str(doc_id),
        content=lc_doc.page_content,
        metadata=lc_doc.metadata,
    )


def _convert_lc_documents(lc_docs: list[LCDocument]) -> list[Document]:
    """Convert a list of LangChain Documents to RAGnarok Documents.

    Args:
        lc_docs: List of LangChain Documents.

    Returns:
        List of RAGnarok Documents.
    """
    return [_convert_lc_document(doc) for doc in lc_docs]


class LangChainRetrieverAdapter:
    """Adapter for LangChain retrievers.

    Wraps a LangChain BaseRetriever for use with ragnarok-ai evaluation.
    This is useful when you only want to evaluate retrieval quality
    without a full RAG chain.

    Attributes:
        retriever: The wrapped LangChain retriever.

    Example:
        >>> from langchain_community.vectorstores import FAISS
        >>> from ragnarok_ai.adapters.frameworks import LangChainRetrieverAdapter
        >>>
        >>> vectorstore = FAISS.from_documents(docs, embeddings)
        >>> retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        >>> adapter = LangChainRetrieverAdapter(retriever)
        >>>
        >>> # Use with evaluate()
        >>> results = await evaluate(adapter, testset)
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        answer_generator: Callable[[str, list[Document]], str] | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            retriever: A LangChain BaseRetriever instance.
            answer_generator: Optional function to generate answers from
                             query and retrieved docs. If not provided,
                             returns a placeholder answer.
        """
        self._retriever = retriever
        self._answer_generator = answer_generator

    @property
    def retriever(self) -> BaseRetriever:
        """Get the underlying LangChain retriever."""
        return self._retriever

    async def query(self, question: str) -> RAGResponse:
        """Execute retrieval and return response.

        Args:
            question: The question to retrieve documents for.

        Returns:
            RAGResponse with retrieved documents and answer.
        """
        # LangChain retrievers may be sync or async
        if hasattr(self._retriever, "ainvoke"):
            lc_docs = await self._retriever.ainvoke(question)
        else:
            lc_docs = await asyncio.to_thread(self._retriever.invoke, question)

        docs = _convert_lc_documents(lc_docs)

        # Generate answer if generator provided
        if self._answer_generator:
            answer = self._answer_generator(question, docs)
        else:
            # Default: concatenate retrieved content as context
            answer = f"Retrieved {len(docs)} documents for: {question}"

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            metadata={"adapter": "langchain_retriever"},
        )


class LangChainAdapter:
    """Adapter for LangChain RAG chains and LCEL runnables.

    Wraps a LangChain Runnable (chain, agent, or LCEL pipeline) for use
    with ragnarok-ai evaluation. Automatically extracts retrieved documents
    from chain outputs.

    Attributes:
        chain: The wrapped LangChain runnable.

    Example:
        >>> from langchain.chains import RetrievalQA
        >>> from ragnarok_ai.adapters.frameworks import LangChainAdapter
        >>>
        >>> chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        >>> adapter = LangChainAdapter(chain)
        >>>
        >>> # Use with evaluate()
        >>> results = await evaluate(adapter, testset)

    Example with LCEL:
        >>> from langchain_core.runnables import RunnablePassthrough
        >>> from langchain_core.output_parsers import StrOutputParser
        >>>
        >>> rag_chain = (
        ...     {"context": retriever, "question": RunnablePassthrough()}
        ...     | prompt
        ...     | llm
        ...     | StrOutputParser()
        ... )
        >>> adapter = LangChainAdapter(
        ...     rag_chain,
        ...     input_key="question",
        ...     output_key=None,  # Direct string output
        ...     docs_key="context",
        ... )
    """

    def __init__(
        self,
        chain: Runnable[Any, Any],
        *,
        input_key: str = "input",
        output_key: str | None = "answer",
        docs_key: str = "source_documents",
        input_transform: Callable[[str], Any] | None = None,
        output_transform: Callable[[Any], tuple[str, list[Document]]] | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            chain: A LangChain Runnable (chain, agent, or LCEL pipeline).
            input_key: Key to use when passing input to the chain.
                      Set to None to pass the question directly.
            output_key: Key to extract the answer from chain output.
                       Set to None if chain returns string directly.
            docs_key: Key to extract retrieved documents from chain output.
                     Common keys: "source_documents", "context", "documents".
            input_transform: Optional function to transform the question
                            before passing to the chain.
            output_transform: Optional function to extract (answer, docs) from
                             chain output. If provided, output_key and docs_key
                             are ignored.
        """
        self._chain = chain
        self._input_key = input_key
        self._output_key = output_key
        self._docs_key = docs_key
        self._input_transform = input_transform
        self._output_transform = output_transform

    @property
    def chain(self) -> Runnable[Any, Any]:
        """Get the underlying LangChain runnable."""
        return self._chain

    def _prepare_input(self, question: str) -> Any:
        """Prepare input for the chain.

        Args:
            question: The question to process.

        Returns:
            Input in the format expected by the chain.
        """
        if self._input_transform:
            return self._input_transform(question)
        if self._input_key is None:
            return question
        return {self._input_key: question}

    def _extract_output(self, result: Any) -> tuple[str, list[Document]]:
        """Extract answer and documents from chain output.

        Args:
            result: Raw output from the chain.

        Returns:
            Tuple of (answer, retrieved_documents).
        """
        if self._output_transform:
            return self._output_transform(result)

        # Handle string output directly
        if isinstance(result, str):
            return result, []

        # Handle dict output
        if isinstance(result, dict):
            # Extract answer
            if self._output_key is None:
                answer = str(result)
            else:
                answer = str(result.get(self._output_key, result.get("result", "")))

            # Extract documents
            docs: list[Document] = []
            raw_docs = result.get(self._docs_key, [])

            if raw_docs:
                # Check if these are LangChain documents
                if hasattr(raw_docs[0], "page_content"):
                    docs = _convert_lc_documents(raw_docs)
                elif isinstance(raw_docs[0], Document):
                    docs = raw_docs
                elif isinstance(raw_docs[0], dict):
                    docs = [
                        Document(
                            id=str(d.get("id", i)),
                            content=str(d.get("content", d.get("page_content", ""))),
                            metadata=d.get("metadata", {}),
                        )
                        for i, d in enumerate(raw_docs)
                    ]

            return answer, docs

        # Fallback
        return str(result), []

    async def query(self, question: str) -> RAGResponse:
        """Execute the RAG chain and return response.

        Args:
            question: The question to answer.

        Returns:
            RAGResponse with answer and retrieved documents.
        """
        chain_input = self._prepare_input(question)

        # LangChain chains may be sync or async
        if hasattr(self._chain, "ainvoke"):
            result = await self._chain.ainvoke(chain_input)
        else:
            result = await asyncio.to_thread(self._chain.invoke, chain_input)

        answer, docs = self._extract_output(result)

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            metadata={"adapter": "langchain"},
        )


def create_langchain_adapter(
    chain: Runnable[Any, Any] | BaseRetriever,
    **kwargs: Any,
) -> LangChainAdapter | LangChainRetrieverAdapter:
    """Factory function to create the appropriate LangChain adapter.

    Automatically detects whether the input is a retriever or a chain
    and returns the appropriate adapter.

    Args:
        chain: A LangChain Runnable or BaseRetriever.
        **kwargs: Additional arguments passed to the adapter constructor.

    Returns:
        LangChainAdapter or LangChainRetrieverAdapter.

    Example:
        >>> adapter = create_langchain_adapter(retriever)  # Returns LangChainRetrieverAdapter
        >>> adapter = create_langchain_adapter(rag_chain)  # Returns LangChainAdapter
    """
    # Import here to avoid requiring langchain at module level
    try:
        from langchain_core.retrievers import BaseRetriever
    except ImportError as e:
        msg = "langchain-core is required for LangChain adapters. Install with: pip install langchain-core"
        raise ImportError(msg) from e

    if isinstance(chain, BaseRetriever):
        # Filter kwargs for retriever adapter
        retriever_kwargs = {k: v for k, v in kwargs.items() if k in ("answer_generator",)}
        return LangChainRetrieverAdapter(chain, **retriever_kwargs)

    return LangChainAdapter(chain, **kwargs)
