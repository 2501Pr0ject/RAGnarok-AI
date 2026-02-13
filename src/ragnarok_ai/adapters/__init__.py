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
from ragnarok_ai.adapters.llm import (
    AnthropicLLM,
    GroqLLM,
    MistralLLM,
    OllamaLLM,
    OpenAILLM,
    TogetherLLM,
    VLLMAdapter,
)
from ragnarok_ai.adapters.vectorstore import ChromaVectorStore, FAISSVectorStore, QdrantVectorStore

# LLM adapter classification
LOCAL_LLM_ADAPTERS: tuple[type, ...] = (OllamaLLM, VLLMAdapter)
CLOUD_LLM_ADAPTERS: tuple[type, ...] = (AnthropicLLM, GroqLLM, MistralLLM, OpenAILLM, TogetherLLM)

# VectorStore adapter classification
LOCAL_VECTORSTORE_ADAPTERS: tuple[type, ...] = (ChromaVectorStore, FAISSVectorStore, QdrantVectorStore)
CLOUD_VECTORSTORE_ADAPTERS: tuple[type, ...] = ()  # Pinecone, Weaviate coming later

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
    "MistralLLM",
    "OllamaLLM",
    "OpenAILLM",
    "QdrantVectorStore",
    "ReActAdapter",
    "ReActParser",
    "TogetherLLM",
    "VLLMAdapter",
]
