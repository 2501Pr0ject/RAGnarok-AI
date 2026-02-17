"""Unit tests for the LangGraph adapter."""

from __future__ import annotations

from typing import Any

import pytest

from ragnarok_ai.adapters.frameworks.langgraph import (
    LangGraphAdapter,
    LangGraphStreamAdapter,
    _extract_answer_from_state,
    _extract_documents_from_state,
)
from ragnarok_ai.core.types import Document, RAGResponse

# ============================================================================
# Mock LangGraph Classes
# ============================================================================


class MockLCDocument:
    """Mock LangChain Document."""

    def __init__(self, page_content: str, metadata: dict[str, Any] | None = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}


class MockMessage:
    """Mock LangChain message."""

    def __init__(self, content: str) -> None:
        self.content = content


class MockCompiledGraph:
    """Mock LangGraph CompiledStateGraph (sync)."""

    def __init__(self, final_state: dict[str, Any]) -> None:
        self._final_state = final_state

    def invoke(
        self,
        _input: dict[str, Any],
        config: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        return self._final_state


class MockAsyncCompiledGraph:
    """Mock LangGraph CompiledStateGraph (async)."""

    def __init__(self, final_state: dict[str, Any]) -> None:
        self._final_state = final_state

    async def ainvoke(
        self,
        _input: dict[str, Any],
        config: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        return self._final_state


class MockStreamingGraph:
    """Mock LangGraph CompiledStateGraph with streaming (sync)."""

    def __init__(self, states: list[dict[str, Any]]) -> None:
        self._states = states

    def stream(
        self,
        _input: dict[str, Any],
        config: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        return self._states

    def invoke(
        self,
        _input: dict[str, Any],
        config: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for state in self._states:
            result.update(state)
        return result


class MockAsyncStreamingGraph:
    """Mock LangGraph CompiledStateGraph with async streaming."""

    def __init__(self, states: list[dict[str, Any]]) -> None:
        self._states = states

    async def astream(
        self,
        _input: dict[str, Any],
        config: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> Any:
        for state in self._states:
            yield state

    async def ainvoke(
        self,
        _input: dict[str, Any],
        config: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for state in self._states:
            result.update(state)
        return result


class MockStateGraph:
    """Mock LangGraph StateGraph (needs compilation)."""

    def __init__(self, compiled: MockCompiledGraph) -> None:
        self._compiled = compiled

    def compile(self) -> MockCompiledGraph:
        return self._compiled


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestExtractDocumentsFromState:
    """Tests for _extract_documents_from_state."""

    def test_extract_ragnarok_documents(self) -> None:
        """Test extracting RAGnarok Document objects."""
        state = {
            "documents": [
                Document(id="doc1", content="Content 1"),
                Document(id="doc2", content="Content 2"),
            ]
        }
        docs = _extract_documents_from_state(state, ["documents"])
        assert len(docs) == 2
        assert docs[0].id == "doc1"

    def test_extract_langchain_documents(self) -> None:
        """Test extracting LangChain Document objects."""
        state = {
            "context": [
                MockLCDocument("Content 1", {"id": "lc1"}),
                MockLCDocument("Content 2", {"source": "file.txt"}),
            ]
        }
        docs = _extract_documents_from_state(state, ["context"])
        assert len(docs) == 2
        assert docs[0].id == "lc1"
        assert docs[1].id == "file.txt"

    def test_extract_dict_documents(self) -> None:
        """Test extracting dict-format documents."""
        state = {
            "docs": [
                {"id": "d1", "content": "Content 1"},
                {"id": "d2", "page_content": "Content 2"},
            ]
        }
        docs = _extract_documents_from_state(state, ["docs"])
        assert len(docs) == 2
        assert docs[0].content == "Content 1"
        assert docs[1].content == "Content 2"

    def test_extract_string_documents(self) -> None:
        """Test extracting string documents."""
        state = {"sources": ["Source 1 text", "Source 2 text"]}
        docs = _extract_documents_from_state(state, ["sources"])
        assert len(docs) == 2
        assert docs[0].content == "Source 1 text"
        assert docs[0].id == "sources_0"

    def test_extract_from_multiple_keys(self) -> None:
        """Test extracting from multiple state keys."""
        state = {
            "documents": [Document(id="doc1", content="C1")],
            "context": [MockLCDocument("C2", {"id": "doc2"})],
        }
        docs = _extract_documents_from_state(state, ["documents", "context"])
        assert len(docs) == 2

    def test_empty_state(self) -> None:
        """Test with empty state."""
        docs = _extract_documents_from_state({}, ["documents"])
        assert docs == []

    def test_missing_key(self) -> None:
        """Test with missing key."""
        state = {"other_key": "value"}
        docs = _extract_documents_from_state(state, ["documents"])
        assert docs == []


class TestExtractAnswerFromState:
    """Tests for _extract_answer_from_state."""

    def test_extract_string_answer(self) -> None:
        """Test extracting string answer."""
        state = {"answer": "The answer is 42."}
        answer = _extract_answer_from_state(state, ["answer"])
        assert answer == "The answer is 42."

    def test_extract_from_messages(self) -> None:
        """Test extracting from messages list."""
        state = {"messages": [MockMessage("First"), MockMessage("Final answer")]}
        answer = _extract_answer_from_state(state, ["messages"])
        assert answer == "Final answer"

    def test_extract_from_string_list(self) -> None:
        """Test extracting from string list."""
        state = {"output": ["Step 1", "Step 2", "Final"]}
        answer = _extract_answer_from_state(state, ["output"])
        assert answer == "Final"

    def test_priority_order(self) -> None:
        """Test that first matching key is used."""
        state = {
            "answer": "First answer",
            "response": "Second response",
        }
        answer = _extract_answer_from_state(state, ["answer", "response"])
        assert answer == "First answer"

    def test_empty_state(self) -> None:
        """Test with empty state."""
        answer = _extract_answer_from_state({}, ["answer"])
        assert answer == ""

    def test_none_value(self) -> None:
        """Test with None value."""
        state = {"answer": None, "response": "Fallback"}
        answer = _extract_answer_from_state(state, ["answer", "response"])
        assert answer == "Fallback"


# ============================================================================
# LangGraphAdapter Tests
# ============================================================================


class TestLangGraphAdapter:
    """Tests for LangGraphAdapter."""

    @pytest.mark.asyncio
    async def test_query_sync_graph(self) -> None:
        """Test querying with a sync compiled graph."""
        graph = MockCompiledGraph(
            {
                "answer": "The capital is Paris.",
                "documents": [
                    Document(id="doc1", content="Paris is the capital of France."),
                ],
            }
        )
        adapter = LangGraphAdapter(graph)

        response = await adapter.query("What is the capital of France?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "The capital is Paris."
        assert len(response.retrieved_docs) == 1
        assert response.retrieved_docs[0].id == "doc1"

    @pytest.mark.asyncio
    async def test_query_async_graph(self) -> None:
        """Test querying with an async compiled graph."""
        graph = MockAsyncCompiledGraph(
            {
                "response": "Async response",
                "context": [MockLCDocument("Content", {"id": "async_doc"})],
            }
        )
        adapter = LangGraphAdapter(graph)

        response = await adapter.query("Test question")

        assert response.answer == "Async response"
        assert len(response.retrieved_docs) == 1
        assert response.retrieved_docs[0].id == "async_doc"

    @pytest.mark.asyncio
    async def test_query_with_state_graph(self) -> None:
        """Test querying with an uncompiled StateGraph."""
        compiled = MockCompiledGraph({"answer": "Compiled answer", "documents": []})
        graph = MockStateGraph(compiled)
        adapter = LangGraphAdapter(graph)

        response = await adapter.query("Test")

        assert response.answer == "Compiled answer"

    @pytest.mark.asyncio
    async def test_custom_input_key(self) -> None:
        """Test custom input key."""

        class InputCheckGraph:
            def invoke(
                self,
                data: dict[str, Any],
                config: dict[str, Any] | None = None,  # noqa: ARG002
            ) -> dict[str, Any]:
                assert "query" in data
                return {"answer": "Input OK", "documents": []}

        graph = InputCheckGraph()
        adapter = LangGraphAdapter(graph, input_key="query")

        response = await adapter.query("Test")
        assert response.answer == "Input OK"

    @pytest.mark.asyncio
    async def test_custom_answer_keys(self) -> None:
        """Test custom answer keys."""
        graph = MockCompiledGraph(
            {
                "generation": "Custom answer key",
                "documents": [],
            }
        )
        adapter = LangGraphAdapter(graph, answer_keys=["generation"])

        response = await adapter.query("Test")
        assert response.answer == "Custom answer key"

    @pytest.mark.asyncio
    async def test_custom_docs_keys(self) -> None:
        """Test custom docs keys."""
        graph = MockCompiledGraph(
            {
                "answer": "Answer",
                "retrieved_chunks": [Document(id="chunk1", content="Chunk content")],
            }
        )
        adapter = LangGraphAdapter(graph, docs_keys=["retrieved_chunks"])

        response = await adapter.query("Test")
        assert len(response.retrieved_docs) == 1
        assert response.retrieved_docs[0].id == "chunk1"

    @pytest.mark.asyncio
    async def test_input_transform(self) -> None:
        """Test custom input transformation."""

        class TransformCheckGraph:
            def invoke(
                self,
                data: dict[str, Any],
                config: dict[str, Any] | None = None,  # noqa: ARG002
            ) -> dict[str, Any]:
                assert data["messages"][0]["role"] == "user"
                return {"answer": "Transform OK", "documents": []}

        def transform(question: str) -> dict[str, Any]:
            return {"messages": [{"role": "user", "content": question}]}

        graph = TransformCheckGraph()
        adapter = LangGraphAdapter(graph, input_transform=transform)

        response = await adapter.query("Test")
        assert response.answer == "Transform OK"

    @pytest.mark.asyncio
    async def test_output_transform(self) -> None:
        """Test custom output transformation."""
        graph = MockCompiledGraph(
            {
                "custom_answer": "Transformed",
                "custom_docs": [{"id": "t1", "text": "Doc text"}],
            }
        )

        def transform(state: dict[str, Any]) -> tuple[str, list[Document]]:
            return state["custom_answer"], [Document(id=d["id"], content=d["text"]) for d in state["custom_docs"]]

        adapter = LangGraphAdapter(graph, output_transform=transform)

        response = await adapter.query("Test")
        assert response.answer == "Transformed"
        assert response.retrieved_docs[0].content == "Doc text"

    @pytest.mark.asyncio
    async def test_with_config(self) -> None:
        """Test passing config to graph."""

        class ConfigCheckGraph:
            def invoke(self, _data: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
                assert config is not None
                assert config.get("recursion_limit") == 10
                return {"answer": "Config OK", "documents": []}

        graph = ConfigCheckGraph()
        adapter = LangGraphAdapter(graph, config={"recursion_limit": 10})

        response = await adapter.query("Test")
        assert response.answer == "Config OK"

    def test_graph_property(self) -> None:
        """Test accessing the underlying graph."""
        graph = MockCompiledGraph({"answer": "", "documents": []})
        adapter = LangGraphAdapter(graph)
        assert adapter.graph is graph

    @pytest.mark.asyncio
    async def test_metadata_includes_state_keys(self) -> None:
        """Test that metadata includes final state keys."""
        graph = MockCompiledGraph(
            {
                "answer": "Test",
                "documents": [],
                "extra_key": "value",
            }
        )
        adapter = LangGraphAdapter(graph)

        response = await adapter.query("Test")
        assert "final_state_keys" in response.metadata
        assert "answer" in response.metadata["final_state_keys"]
        assert "extra_key" in response.metadata["final_state_keys"]


# ============================================================================
# LangGraphStreamAdapter Tests
# ============================================================================


class TestLangGraphStreamAdapter:
    """Tests for LangGraphStreamAdapter."""

    @pytest.mark.asyncio
    async def test_stream_sync_graph(self) -> None:
        """Test streaming with sync graph."""
        states = [
            {"step": "retrieve", "documents": [Document(id="d1", content="C1")]},
            {"step": "generate", "answer": "Final answer"},
        ]
        graph = MockStreamingGraph(states)
        adapter = LangGraphStreamAdapter(graph)

        response = await adapter.query("Test")

        assert response.answer == "Final answer"
        assert "trace" in response.metadata
        assert response.metadata["trace_length"] == 2

    @pytest.mark.asyncio
    async def test_stream_async_graph(self) -> None:
        """Test streaming with async graph."""
        states = [
            {"documents": [MockLCDocument("Doc content", {"id": "ad1"})]},
            {"answer": "Async streamed answer"},
        ]
        graph = MockAsyncStreamingGraph(states)
        adapter = LangGraphStreamAdapter(graph)

        response = await adapter.query("Test")

        assert response.answer == "Async streamed answer"
        assert response.metadata["trace_length"] == 2

    @pytest.mark.asyncio
    async def test_stream_captures_trace(self) -> None:
        """Test that streaming captures state trace."""
        states = [
            {"step": 1, "status": "retrieving"},
            {"step": 2, "status": "generating"},
            {"step": 3, "answer": "Done"},
        ]
        graph = MockStreamingGraph(states)
        adapter = LangGraphStreamAdapter(graph)

        response = await adapter.query("Test")

        trace = response.metadata["trace"]
        assert len(trace) == 3
        assert trace[0]["update"]["step"] == 1

    @pytest.mark.asyncio
    async def test_fallback_to_invoke(self) -> None:
        """Test fallback to invoke when stream not available."""
        graph = MockCompiledGraph(
            {
                "answer": "Fallback answer",
                "documents": [],
            }
        )
        adapter = LangGraphStreamAdapter(graph)

        response = await adapter.query("Test")

        assert response.answer == "Fallback answer"
        # No trace when using fallback
        assert response.metadata.get("trace", []) == []


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestProtocolCompliance:
    """Tests for RAGProtocol compliance."""

    @pytest.mark.asyncio
    async def test_adapter_implements_protocol(self) -> None:
        """Test that LangGraphAdapter implements RAGProtocol."""
        from ragnarok_ai.core.protocols import RAGProtocol

        graph = MockCompiledGraph({"answer": "", "documents": []})
        adapter = LangGraphAdapter(graph)

        assert hasattr(adapter, "query")
        assert callable(adapter.query)
        assert isinstance(adapter, RAGProtocol)

    @pytest.mark.asyncio
    async def test_stream_adapter_implements_protocol(self) -> None:
        """Test that LangGraphStreamAdapter implements RAGProtocol."""
        from ragnarok_ai.core.protocols import RAGProtocol

        graph = MockCompiledGraph({"answer": "", "documents": []})
        adapter = LangGraphStreamAdapter(graph)

        assert isinstance(adapter, RAGProtocol)


# ============================================================================
# Integration with Evaluate Tests
# ============================================================================


class TestEvaluateIntegration:
    """Tests for integration with evaluate function."""

    @pytest.mark.asyncio
    async def test_evaluate_with_langgraph_adapter(self) -> None:
        """Test using LangGraph adapter with evaluate."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        graph = MockCompiledGraph(
            {
                "answer": "Paris is the capital of France.",
                "documents": [Document(id="doc1", content="France info")],
            }
        )
        adapter = LangGraphAdapter(graph)

        testset = TestSet(
            queries=[
                Query(text="What is the capital of France?", ground_truth_docs=["doc1"]),
            ]
        )

        result = await evaluate(adapter, testset)

        assert result is not None
        assert len(result.responses) == 1
        assert "Paris" in result.responses[0]

    @pytest.mark.asyncio
    async def test_evaluate_with_stream_adapter(self) -> None:
        """Test using stream adapter with evaluate."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        states = [
            {"documents": [Document(id="doc1", content="Relevant")]},
            {"answer": "Streamed answer"},
        ]
        graph = MockStreamingGraph(states)
        adapter = LangGraphStreamAdapter(graph)

        testset = TestSet(
            queries=[
                Query(text="Test?", ground_truth_docs=["doc1"]),
            ]
        )

        result = await evaluate(adapter, testset)

        assert len(result.responses) == 1
