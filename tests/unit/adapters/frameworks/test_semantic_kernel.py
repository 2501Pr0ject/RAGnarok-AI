"""Unit tests for Semantic Kernel framework adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragnarok_ai.adapters.frameworks.semantic_kernel import (
    SemanticKernelAdapter,
    SemanticKernelMemoryAdapter,
)
from ragnarok_ai.core.types import Document

# ============================================================================
# Initialization Tests
# ============================================================================


class TestSemanticKernelAdapterInit:
    """Tests for SemanticKernelAdapter initialization."""

    def test_init_with_function(self) -> None:
        """Test initialization with a function."""
        mock_kernel = MagicMock()
        mock_function = MagicMock()
        adapter = SemanticKernelAdapter(mock_kernel, function=mock_function)
        assert adapter.kernel is mock_kernel
        assert adapter.is_local is True

    def test_init_with_function_name(self) -> None:
        """Test initialization with function name."""
        mock_kernel = MagicMock()
        adapter = SemanticKernelAdapter(
            mock_kernel,
            function_name="my_function",
            plugin_name="my_plugin",
        )
        assert adapter._function_name == "my_function"
        assert adapter._plugin_name == "my_plugin"

    def test_init_without_function_raises(self) -> None:
        """Test that initialization without function raises ValueError."""
        mock_kernel = MagicMock()
        with pytest.raises(ValueError, match="Either 'function' or 'function_name'"):
            SemanticKernelAdapter(mock_kernel)

    def test_init_with_custom_input_key(self) -> None:
        """Test initialization with custom input key."""
        mock_kernel = MagicMock()
        mock_function = MagicMock()
        adapter = SemanticKernelAdapter(
            mock_kernel,
            function=mock_function,
            input_key="query",
        )
        assert adapter._input_key == "query"


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestSemanticKernelAdapterProtocol:
    """Tests for protocol compliance."""

    def test_is_local_true(self) -> None:
        """Test that is_local is True."""
        mock_kernel = MagicMock()
        mock_function = MagicMock()
        adapter = SemanticKernelAdapter(mock_kernel, function=mock_function)
        assert adapter.is_local is True


# ============================================================================
# Query Tests
# ============================================================================


class TestSemanticKernelAdapterQuery:
    """Tests for query execution."""

    @pytest.mark.asyncio
    async def test_query_with_function(self) -> None:
        """Test query with provided function."""
        mock_kernel = MagicMock()
        mock_function = MagicMock()
        mock_kernel.invoke = AsyncMock(return_value="This is the answer")

        adapter = SemanticKernelAdapter(mock_kernel, function=mock_function)
        response = await adapter.query("What is the answer?")

        assert response.answer == "This is the answer"
        assert response.retrieved_docs == []
        mock_kernel.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_with_function_name(self) -> None:
        """Test query with function name lookup."""
        mock_kernel = MagicMock()
        mock_function = MagicMock()
        mock_plugin = MagicMock()
        mock_plugin.get.return_value = mock_function
        mock_kernel.get_plugin.return_value = mock_plugin
        mock_kernel.invoke = AsyncMock(return_value="Answer from plugin")

        adapter = SemanticKernelAdapter(
            mock_kernel,
            function_name="answer",
            plugin_name="rag",
        )
        response = await adapter.query("Question?")

        assert response.answer == "Answer from plugin"

    @pytest.mark.asyncio
    async def test_query_with_docs_extractor(self) -> None:
        """Test query with custom docs extractor."""
        mock_kernel = MagicMock()
        mock_function = MagicMock()
        mock_kernel.invoke = AsyncMock(return_value="Answer")

        def extract_docs(result: str) -> list[Document]:  # noqa: ARG001
            return [Document(id="1", content="Extracted doc", metadata={})]

        adapter = SemanticKernelAdapter(
            mock_kernel,
            function=mock_function,
            docs_extractor=extract_docs,
        )
        response = await adapter.query("Question?")

        assert response.answer == "Answer"
        assert len(response.retrieved_docs) == 1
        assert response.retrieved_docs[0].content == "Extracted doc"

    @pytest.mark.asyncio
    async def test_query_empty_result(self) -> None:
        """Test query with empty result."""
        mock_kernel = MagicMock()
        mock_function = MagicMock()
        mock_kernel.invoke = AsyncMock(return_value=None)

        adapter = SemanticKernelAdapter(mock_kernel, function=mock_function)
        response = await adapter.query("Question?")

        assert response.answer == ""

    @pytest.mark.asyncio
    async def test_query_plugin_not_found(self) -> None:
        """Test query when plugin is not found."""
        mock_kernel = MagicMock()
        mock_kernel.get_plugin.return_value = None

        adapter = SemanticKernelAdapter(
            mock_kernel,
            function_name="func",
            plugin_name="nonexistent",
        )

        with pytest.raises(ValueError, match="Plugin 'nonexistent' not found"):
            await adapter.query("Question?")


# ============================================================================
# Memory Adapter Tests
# ============================================================================


class TestSemanticKernelMemoryAdapter:
    """Tests for SemanticKernelMemoryAdapter."""

    def test_init(self) -> None:
        """Test initialization."""
        mock_kernel = MagicMock()
        mock_memory = MagicMock()
        adapter = SemanticKernelMemoryAdapter(mock_kernel, memory=mock_memory)
        assert adapter.kernel is mock_kernel
        assert adapter.is_local is True

    def test_init_with_options(self) -> None:
        """Test initialization with options."""
        mock_kernel = MagicMock()
        mock_memory = MagicMock()
        adapter = SemanticKernelMemoryAdapter(
            mock_kernel,
            memory=mock_memory,
            collection="my_collection",
            limit=20,
            min_relevance=0.5,
        )
        assert adapter._collection == "my_collection"
        assert adapter._limit == 20
        assert adapter._min_relevance == 0.5

    @pytest.mark.asyncio
    async def test_query_success(self) -> None:
        """Test successful memory search."""
        mock_kernel = MagicMock()
        mock_memory = MagicMock()

        mock_result = MagicMock()
        mock_result.id = "doc1"
        mock_result.text = "Memory content"
        mock_result.relevance = 0.9
        mock_result.metadata = {"key": "value"}

        mock_memory.search = AsyncMock(return_value=[mock_result])

        adapter = SemanticKernelMemoryAdapter(mock_kernel, memory=mock_memory)
        response = await adapter.query("Search query")

        assert len(response.retrieved_docs) == 1
        assert response.retrieved_docs[0].id == "doc1"
        assert response.retrieved_docs[0].content == "Memory content"
        assert response.retrieved_docs[0].metadata["relevance"] == 0.9

    @pytest.mark.asyncio
    async def test_query_with_answer_generator(self) -> None:
        """Test query with custom answer generator."""
        mock_kernel = MagicMock()
        mock_memory = MagicMock()
        mock_memory.search = AsyncMock(return_value=[])

        def generate_answer(question: str, docs: list[Document]) -> str:  # noqa: ARG001
            return f"Custom answer for: {question}"

        adapter = SemanticKernelMemoryAdapter(
            mock_kernel,
            memory=mock_memory,
            answer_generator=generate_answer,
        )
        response = await adapter.query("Test query")

        assert response.answer == "Custom answer for: Test query"

    @pytest.mark.asyncio
    async def test_query_no_memory_raises(self) -> None:
        """Test query without memory configured raises ValueError."""
        mock_kernel = MagicMock()
        adapter = SemanticKernelMemoryAdapter(mock_kernel)

        with pytest.raises(ValueError, match="No memory configured"):
            await adapter.query("Query")

    @pytest.mark.asyncio
    async def test_query_empty_results(self) -> None:
        """Test query with no results."""
        mock_kernel = MagicMock()
        mock_memory = MagicMock()
        mock_memory.search = AsyncMock(return_value=[])

        adapter = SemanticKernelMemoryAdapter(mock_kernel, memory=mock_memory)
        response = await adapter.query("Query")

        assert response.retrieved_docs == []
        assert "Retrieved 0 documents" in response.answer
