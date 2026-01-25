"""LangGraph adapter for ragnarok-ai.

This module provides adapters to evaluate LangGraph stateful RAG agents.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.types import Document, RAGResponse

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langgraph.graph import StateGraph
    from langgraph.graph.state import CompiledStateGraph


def _extract_documents_from_state(
    state: dict[str, Any],
    docs_keys: Sequence[str],
) -> list[Document]:
    """Extract documents from graph state.

    Args:
        state: The graph state dictionary.
        docs_keys: Keys to search for documents in state.

    Returns:
        List of extracted documents.
    """
    documents: list[Document] = []

    for key in docs_keys:
        if key not in state:
            continue

        raw_docs = state[key]
        if not raw_docs:
            continue

        # Handle list of documents
        if isinstance(raw_docs, list):
            for i, doc in enumerate(raw_docs):
                if isinstance(doc, Document):
                    documents.append(doc)
                elif hasattr(doc, "page_content"):
                    # LangChain Document
                    doc_id = doc.metadata.get("id") or doc.metadata.get("source") or str(hash(doc.page_content))
                    documents.append(
                        Document(
                            id=str(doc_id),
                            content=doc.page_content,
                            metadata=doc.metadata,
                        )
                    )
                elif isinstance(doc, dict):
                    documents.append(
                        Document(
                            id=str(doc.get("id", i)),
                            content=str(doc.get("content", doc.get("page_content", ""))),
                            metadata=doc.get("metadata", {}),
                        )
                    )
                elif isinstance(doc, str):
                    documents.append(
                        Document(
                            id=f"{key}_{i}",
                            content=doc,
                            metadata={"source_key": key},
                        )
                    )

    return documents


def _extract_answer_from_state(
    state: dict[str, Any],
    answer_keys: Sequence[str],
) -> str:
    """Extract answer from graph state.

    Args:
        state: The graph state dictionary.
        answer_keys: Keys to search for answer in state.

    Returns:
        Extracted answer string.
    """
    for key in answer_keys:
        value = state.get(key)
        if value:
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value:
                # Take last message if it's a list
                last = value[-1]
                if isinstance(last, str):
                    return last
                if hasattr(last, "content"):
                    return str(last.content)
                return str(last)
            return str(value)
    return ""


class LangGraphAdapter:
    """Adapter for LangGraph StateGraph and CompiledStateGraph.

    Wraps a LangGraph graph for use with ragnarok-ai evaluation.
    Automatically extracts retrieved documents and answers from graph state.

    Attributes:
        graph: The wrapped LangGraph graph.

    Example:
        >>> from langgraph.graph import StateGraph
        >>> from ragnarok_ai.adapters.frameworks import LangGraphAdapter
        >>>
        >>> # Define your graph
        >>> graph = StateGraph(AgentState)
        >>> graph.add_node("retrieve", retrieve_node)
        >>> graph.add_node("generate", generate_node)
        >>> # ... add edges
        >>> compiled = graph.compile()
        >>>
        >>> # Wrap for evaluation
        >>> adapter = LangGraphAdapter(compiled)
        >>> results = await evaluate(adapter, testset)

    Example with custom extraction:
        >>> adapter = LangGraphAdapter(
        ...     compiled,
        ...     input_key="question",
        ...     answer_keys=["response", "output"],
        ...     docs_keys=["documents", "context", "retrieved_docs"],
        ... )
    """

    def __init__(
        self,
        graph: StateGraph[Any] | CompiledStateGraph,
        *,
        input_key: str = "question",
        answer_keys: Sequence[str] | None = None,
        docs_keys: Sequence[str] | None = None,
        input_transform: Callable[[str], dict[str, Any]] | None = None,
        output_transform: Callable[[dict[str, Any]], tuple[str, list[Document]]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            graph: A LangGraph StateGraph or CompiledStateGraph.
            input_key: Key to use when passing the question to the graph.
            answer_keys: Keys to search for the answer in final state.
                        Defaults to common answer keys.
            docs_keys: Keys to search for retrieved documents in state.
                      Defaults to common document keys.
            input_transform: Optional function to transform the question
                            into graph input format.
            output_transform: Optional function to extract (answer, docs)
                             from final state. If provided, answer_keys
                             and docs_keys are ignored.
            config: Optional LangGraph config to pass to invoke/ainvoke.
        """
        self._graph = graph
        self._input_key = input_key
        self._answer_keys = answer_keys or [
            "answer",
            "response",
            "output",
            "result",
            "generation",
            "messages",
        ]
        self._docs_keys = docs_keys or [
            "documents",
            "docs",
            "context",
            "retrieved_docs",
            "retrieved_documents",
            "sources",
        ]
        self._input_transform = input_transform
        self._output_transform = output_transform
        self._config = config or {}

        # Compile if needed
        if hasattr(graph, "compile") and not hasattr(graph, "invoke"):
            self._compiled = graph.compile()
        else:
            self._compiled = graph

    @property
    def graph(self) -> StateGraph[Any] | CompiledStateGraph:
        """Get the underlying LangGraph graph."""
        return self._graph

    def _prepare_input(self, question: str) -> dict[str, Any]:
        """Prepare input for the graph.

        Args:
            question: The question to process.

        Returns:
            Input dict for the graph.
        """
        if self._input_transform:
            return self._input_transform(question)
        return {self._input_key: question}

    def _extract_output(self, state: dict[str, Any]) -> tuple[str, list[Document]]:
        """Extract answer and documents from final state.

        Args:
            state: The final graph state.

        Returns:
            Tuple of (answer, documents).
        """
        if self._output_transform:
            return self._output_transform(state)

        answer = _extract_answer_from_state(state, self._answer_keys)
        documents = _extract_documents_from_state(state, self._docs_keys)

        return answer, documents

    async def query(self, question: str) -> RAGResponse:
        """Execute the graph and return response.

        Args:
            question: The question to answer.

        Returns:
            RAGResponse with answer and retrieved documents.
        """
        graph_input = self._prepare_input(question)

        # LangGraph graphs may be sync or async
        if hasattr(self._compiled, "ainvoke"):
            final_state = await self._compiled.ainvoke(graph_input, config=self._config)
        else:
            final_state = await asyncio.to_thread(self._compiled.invoke, graph_input, config=self._config)

        answer, documents = self._extract_output(final_state)

        return RAGResponse(
            answer=answer,
            retrieved_docs=documents,
            metadata={
                "adapter": "langgraph",
                "final_state_keys": list(final_state.keys()) if isinstance(final_state, dict) else [],
            },
        )


class LangGraphStreamAdapter(LangGraphAdapter):
    """Adapter for LangGraph with streaming state updates.

    Extends LangGraphAdapter to capture intermediate states during
    graph execution. Useful for debugging and tracing.

    Example:
        >>> adapter = LangGraphStreamAdapter(compiled)
        >>> response = await adapter.query("What is X?")
        >>> print(response.metadata["trace"])  # List of state updates
    """

    async def query(self, question: str) -> RAGResponse:
        """Execute the graph with streaming and return response.

        Args:
            question: The question to answer.

        Returns:
            RAGResponse with answer, documents, and execution trace.
        """
        graph_input = self._prepare_input(question)
        trace: list[dict[str, Any]] = []
        final_state: dict[str, Any] = {}

        # Stream execution if supported
        if hasattr(self._compiled, "astream"):
            async for state_update in self._compiled.astream(graph_input, config=self._config):
                trace.append({"update": state_update})
                if isinstance(state_update, dict):
                    final_state.update(state_update)
        elif hasattr(self._compiled, "stream"):
            # Sync stream in thread
            def run_stream() -> tuple[list[dict[str, Any]], dict[str, Any]]:
                t: list[dict[str, Any]] = []
                s: dict[str, Any] = {}
                for state_update in self._compiled.stream(graph_input, config=self._config):
                    t.append({"update": state_update})
                    if isinstance(state_update, dict):
                        s.update(state_update)
                return t, s

            trace, final_state = await asyncio.to_thread(run_stream)
        else:
            # Fallback to regular invoke
            if hasattr(self._compiled, "ainvoke"):
                final_state = await self._compiled.ainvoke(graph_input, config=self._config)
            else:
                final_state = await asyncio.to_thread(self._compiled.invoke, graph_input, config=self._config)

        answer, documents = self._extract_output(final_state)

        return RAGResponse(
            answer=answer,
            retrieved_docs=documents,
            metadata={
                "adapter": "langgraph_stream",
                "trace": trace,
                "trace_length": len(trace),
                "final_state_keys": list(final_state.keys()) if isinstance(final_state, dict) else [],
            },
        )
