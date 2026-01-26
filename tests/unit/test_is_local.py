"""Tests for is_local property on adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragnarok_ai.adapters import (
    CLOUD_LLM_ADAPTERS,
    CLOUD_VECTORSTORE_ADAPTERS,
    LOCAL_LLM_ADAPTERS,
    LOCAL_VECTORSTORE_ADAPTERS,
    OllamaLLM,
    QdrantVectorStore,
)
from ragnarok_ai.core.protocols import LLMProtocol, VectorStoreProtocol

if TYPE_CHECKING:
    from ragnarok_ai.core.types import Document


# ============================================================================
# LLM Protocol Tests
# ============================================================================


class TestLLMIsLocal:
    """Tests for is_local on LLM adapters."""

    def test_ollama_is_local(self) -> None:
        """OllamaLLM should be marked as local."""
        llm = OllamaLLM()
        assert llm.is_local is True

    def test_ollama_in_local_adapters(self) -> None:
        """OllamaLLM should be in LOCAL_LLM_ADAPTERS."""
        assert OllamaLLM in LOCAL_LLM_ADAPTERS

    def test_ollama_not_in_cloud_adapters(self) -> None:
        """OllamaLLM should not be in CLOUD_LLM_ADAPTERS."""
        assert OllamaLLM not in CLOUD_LLM_ADAPTERS

    def test_llm_protocol_requires_is_local(self) -> None:
        """LLMProtocol should require is_local attribute."""

        class IncompleteAdapter:
            async def generate(self, prompt: str) -> str:  # noqa: ARG002
                return "response"

            async def embed(self, text: str) -> list[float]:  # noqa: ARG002
                return [0.1, 0.2]

        # Without is_local, should not be a valid LLMProtocol
        adapter = IncompleteAdapter()
        assert not isinstance(adapter, LLMProtocol)

    def test_llm_protocol_with_is_local(self) -> None:
        """LLMProtocol should accept adapters with is_local."""

        class CompleteAdapter:
            is_local: bool = True

            async def generate(self, prompt: str) -> str:  # noqa: ARG002
                return "response"

            async def embed(self, text: str) -> list[float]:  # noqa: ARG002
                return [0.1, 0.2]

        adapter = CompleteAdapter()
        assert isinstance(adapter, LLMProtocol)
        assert adapter.is_local is True


# ============================================================================
# VectorStore Protocol Tests
# ============================================================================


class TestVectorStoreIsLocal:
    """Tests for is_local on VectorStore adapters."""

    def test_qdrant_is_local(self) -> None:
        """QdrantVectorStore should be marked as local."""
        store = QdrantVectorStore()
        assert store.is_local is True

    def test_qdrant_in_local_adapters(self) -> None:
        """QdrantVectorStore should be in LOCAL_VECTORSTORE_ADAPTERS."""
        assert QdrantVectorStore in LOCAL_VECTORSTORE_ADAPTERS

    def test_qdrant_not_in_cloud_adapters(self) -> None:
        """QdrantVectorStore should not be in CLOUD_VECTORSTORE_ADAPTERS."""
        assert QdrantVectorStore not in CLOUD_VECTORSTORE_ADAPTERS

    def test_vectorstore_protocol_requires_is_local(self) -> None:
        """VectorStoreProtocol should require is_local attribute."""

        class IncompleteAdapter:
            async def search(
                self,
                query_embedding: list[float],  # noqa: ARG002
                k: int = 10,  # noqa: ARG002
            ) -> list[tuple[Document, float]]:
                return []

            async def add(self, documents: list[Document]) -> None:
                pass

        # Without is_local, should not be a valid VectorStoreProtocol
        adapter = IncompleteAdapter()
        assert not isinstance(adapter, VectorStoreProtocol)

    def test_vectorstore_protocol_with_is_local(self) -> None:
        """VectorStoreProtocol should accept adapters with is_local."""

        class CompleteAdapter:
            is_local: bool = False  # Cloud adapter

            async def search(
                self,
                query_embedding: list[float],  # noqa: ARG002
                k: int = 10,  # noqa: ARG002
            ) -> list[tuple[Document, float]]:
                return []

            async def add(self, documents: list[Document]) -> None:
                pass

        adapter = CompleteAdapter()
        assert isinstance(adapter, VectorStoreProtocol)
        assert adapter.is_local is False


# ============================================================================
# Classification Lists Tests
# ============================================================================


class TestAdapterClassification:
    """Tests for adapter classification lists."""

    def test_all_local_llm_adapters_have_is_local_true(self) -> None:
        """All LOCAL_LLM_ADAPTERS should have is_local=True."""
        for adapter_cls in LOCAL_LLM_ADAPTERS:
            instance = adapter_cls()
            assert instance.is_local is True, f"{adapter_cls.__name__} should be local"

    def test_all_local_vectorstore_adapters_have_is_local_true(self) -> None:
        """All LOCAL_VECTORSTORE_ADAPTERS should have is_local=True."""
        for adapter_cls in LOCAL_VECTORSTORE_ADAPTERS:
            instance = adapter_cls()
            assert instance.is_local is True, f"{adapter_cls.__name__} should be local"

    def test_no_overlap_llm_adapters(self) -> None:
        """LOCAL and CLOUD LLM adapters should not overlap."""
        local_set = set(LOCAL_LLM_ADAPTERS)
        cloud_set = set(CLOUD_LLM_ADAPTERS)
        assert local_set.isdisjoint(cloud_set)

    def test_no_overlap_vectorstore_adapters(self) -> None:
        """LOCAL and CLOUD VectorStore adapters should not overlap."""
        local_set = set(LOCAL_VECTORSTORE_ADAPTERS)
        cloud_set = set(CLOUD_VECTORSTORE_ADAPTERS)
        assert local_set.isdisjoint(cloud_set)
