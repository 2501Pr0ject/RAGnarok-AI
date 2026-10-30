"""Adapters module for ragnarok-ai.

This module provides adapters for external services:
- LLM providers (Ollama, vLLM, etc.)
- Vector stores (Qdrant, ChromaDB, etc.)
- RAG frameworks (LangChain, LangGraph, etc.)

Adapter Classification:
- LOCAL adapters run entirely on your infrastructure (no data leaves your network)
- CLOUD adapters send data to external APIs (OpenAI, Anthropic, etc.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragnarok_ai.adapters.agents import (
    ChainOfThoughtAdapter,
    ReActAdapter,
    ReActParser,
)
from ragnarok_ai.adapters.frameworks import (
    DSPyModuleAdapter,
    DSPyRAGAdapter,
    DSPyRetrieverAdapter,
    LangChainAdapter,
    LangChainRetrieverAdapter,
    LangGraphAdapter,
    LangGraphStreamAdapter,
    LlamaIndexAdapter,
    LlamaIndexQueryEngineAdapter,
    LlamaIndexRetrieverAdapter,
)
from ragnarok_ai.adapters.vectorstore import (
    ChromaVectorStore,
    FAISSVectorStore,
    MilvusVectorStore,
    PgvectorVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    WeaviateVectorStore,
)

if TYPE_CHECKING:
    from ragnarok_ai.adapters.llm import (
        AnthropicLLM,
        GroqLLM,
        MistralLLM,
        OllamaLLM,
        OpenAILLM,
        TogetherLLM,
        VLLMAdapter,
    )

    LOCAL_LLM_ADAPTERS: tuple[type, ...]
    CLOUD_LLM_ADAPTERS: tuple[type, ...]

# VectorStore adapter classification
LOCAL_VECTORSTORE_ADAPTERS: tuple[type, ...] = (
    ChromaVectorStore,
    FAISSVectorStore,
    MilvusVectorStore,
    PgvectorVectorStore,
    QdrantVectorStore,
)

CLOUD_VECTORSTORE_ADAPTERS: tuple[type, ...] = (PineconeVectorStore, WeaviateVectorStore)

# LLM adapters (and their classification) need httpx, an optional dependency:
# they are resolved on first access so that importing this package works on a
# base install.
_LLM_ADAPTERS = ("AnthropicLLM", "GroqLLM", "MistralLLM", "OllamaLLM", "OpenAILLM", "TogetherLLM", "VLLMAdapter")
_LLM_CLASSIFICATION = {
    "LOCAL_LLM_ADAPTERS": ("OllamaLLM", "VLLMAdapter"),
    "CLOUD_LLM_ADAPTERS": ("AnthropicLLM", "GroqLLM", "MistralLLM", "OpenAILLM", "TogetherLLM"),
}


def __getattr__(name: str) -> Any:
    from ragnarok_ai.adapters import llm

    if name in _LLM_ADAPTERS:
        value: Any = getattr(llm, name)
    elif name in _LLM_CLASSIFICATION:
        value = tuple(getattr(llm, adapter) for adapter in _LLM_CLASSIFICATION[name])
    else:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    globals()[name] = value  # cache: later lookups bypass __getattr__
    return value


__all__ = [
    "CLOUD_LLM_ADAPTERS",
    "CLOUD_VECTORSTORE_ADAPTERS",
    "LOCAL_LLM_ADAPTERS",
    "LOCAL_VECTORSTORE_ADAPTERS",
    "AnthropicLLM",
    "ChainOfThoughtAdapter",
    "ChromaVectorStore",
    "DSPyModuleAdapter",
    "DSPyRAGAdapter",
    "DSPyRetrieverAdapter",
    "FAISSVectorStore",
    "GroqLLM",
    "LangChainAdapter",
    "LangChainRetrieverAdapter",
    "LangGraphAdapter",
    "LangGraphStreamAdapter",
    "LlamaIndexAdapter",
    "LlamaIndexQueryEngineAdapter",
    "LlamaIndexRetrieverAdapter",
    "MilvusVectorStore",
    "MistralLLM",
    "OllamaLLM",
    "OpenAILLM",
    "PgvectorVectorStore",
    "PineconeVectorStore",
    "QdrantVectorStore",
    "ReActAdapter",
    "ReActParser",
    "TogetherLLM",
    "VLLMAdapter",
    "WeaviateVectorStore",
]
