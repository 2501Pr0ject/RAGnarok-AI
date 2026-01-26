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

from ragnarok_ai.adapters.frameworks import (
    LangChainAdapter,
    LangChainRetrieverAdapter,
    LangGraphAdapter,
    LangGraphStreamAdapter,
)
from ragnarok_ai.adapters.llm import AnthropicLLM, OllamaLLM, OpenAILLM, VLLMAdapter
from ragnarok_ai.adapters.vectorstore import QdrantVectorStore

# LLM adapter classification
LOCAL_LLM_ADAPTERS: tuple[type, ...] = (OllamaLLM, VLLMAdapter)
CLOUD_LLM_ADAPTERS: tuple[type, ...] = (AnthropicLLM, OpenAILLM)

# VectorStore adapter classification
LOCAL_VECTORSTORE_ADAPTERS: tuple[type, ...] = (QdrantVectorStore,)
CLOUD_VECTORSTORE_ADAPTERS: tuple[type, ...] = ()  # Pinecone, Weaviate coming later

__all__ = [
    "CLOUD_LLM_ADAPTERS",
    "CLOUD_VECTORSTORE_ADAPTERS",
    "LOCAL_LLM_ADAPTERS",
    "LOCAL_VECTORSTORE_ADAPTERS",
    "AnthropicLLM",
    "LangChainAdapter",
    "LangChainRetrieverAdapter",
    "LangGraphAdapter",
    "LangGraphStreamAdapter",
    "OllamaLLM",
    "OpenAILLM",
    "QdrantVectorStore",
    "VLLMAdapter",
]
