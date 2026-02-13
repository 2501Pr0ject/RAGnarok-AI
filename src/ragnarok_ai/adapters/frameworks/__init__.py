"""Framework adapters for ragnarok-ai.

This module provides adapters for popular RAG frameworks like LangChain, LangGraph, LlamaIndex, DSPy,
Haystack, and Semantic Kernel.
"""

from __future__ import annotations

from ragnarok_ai.adapters.frameworks.dspy import (
    DSPyModuleAdapter,
    DSPyRAGAdapter,
    DSPyRetrieverAdapter,
)
from ragnarok_ai.adapters.frameworks.haystack import (
    HaystackAdapter,
    HaystackRetrieverAdapter,
)
from ragnarok_ai.adapters.frameworks.langchain import (
    LangChainAdapter,
    LangChainRetrieverAdapter,
)
from ragnarok_ai.adapters.frameworks.langgraph import (
    LangGraphAdapter,
    LangGraphStreamAdapter,
)
from ragnarok_ai.adapters.frameworks.llamaindex import (
    LlamaIndexAdapter,
    LlamaIndexQueryEngineAdapter,
    LlamaIndexRetrieverAdapter,
)
from ragnarok_ai.adapters.frameworks.semantic_kernel import (
    SemanticKernelAdapter,
    SemanticKernelMemoryAdapter,
)

__all__ = [
    "DSPyModuleAdapter",
    "DSPyRAGAdapter",
    "DSPyRetrieverAdapter",
    "HaystackAdapter",
    "HaystackRetrieverAdapter",
    "LangChainAdapter",
    "LangChainRetrieverAdapter",
    "LangGraphAdapter",
    "LangGraphStreamAdapter",
    "LlamaIndexAdapter",
    "LlamaIndexQueryEngineAdapter",
    "LlamaIndexRetrieverAdapter",
    "SemanticKernelAdapter",
    "SemanticKernelMemoryAdapter",
]
