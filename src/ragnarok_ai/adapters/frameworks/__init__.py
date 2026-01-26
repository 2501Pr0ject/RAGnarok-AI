"""Framework adapters for ragnarok-ai.

This module provides adapters for popular RAG frameworks like LangChain, LangGraph, LlamaIndex, and DSPy.
"""

from __future__ import annotations

from ragnarok_ai.adapters.frameworks.dspy import (
    DSPyModuleAdapter,
    DSPyRAGAdapter,
    DSPyRetrieverAdapter,
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

__all__ = [
    "DSPyModuleAdapter",
    "DSPyRAGAdapter",
    "DSPyRetrieverAdapter",
    "LangChainAdapter",
    "LangChainRetrieverAdapter",
    "LangGraphAdapter",
    "LangGraphStreamAdapter",
    "LlamaIndexAdapter",
    "LlamaIndexQueryEngineAdapter",
    "LlamaIndexRetrieverAdapter",
]
