"""Plugin system for ragnarok-ai.

This module provides a plugin architecture based on Python entry points,
allowing external packages to register adapters for LLM providers,
vector stores, RAG frameworks, and evaluators.

Example - Using the Plugin Registry:
    >>> from ragnarok_ai.plugins import PluginRegistry
    >>>
    >>> registry = PluginRegistry.get()
    >>> registry.discover()
    >>>
    >>> # List all LLM adapters
    >>> for info in registry.list_adapters(adapter_type="llm"):
    ...     print(f"{info.name}: local={info.is_local}")
    ollama: local=True
    openai: local=False
    >>>
    >>> # Get a specific adapter
    >>> OllamaLLM = registry.get_adapter("ollama")
    >>> llm = OllamaLLM(model="mistral")

Example - Creating a Plugin Package:
    # In your pyproject.toml:
    [project.entry-points."ragnarok_ai.adapters.llm"]
    groq = "ragnarok_ai_groq:GroqLLM"

    # Your adapter must implement the LLMProtocol:
    class GroqLLM:
        is_local = False  # Groq is a cloud service

        async def generate(self, prompt: str) -> str:
            ...

        async def embed(self, text: str) -> list[float]:
            ...
"""

from __future__ import annotations

from ragnarok_ai.plugins.discovery import (
    ALL_NAMESPACES,
    NAMESPACE_EVALUATOR,
    NAMESPACE_FRAMEWORK,
    NAMESPACE_LLM,
    NAMESPACE_VECTORSTORE,
    discover_plugins,
    discover_plugins_safe,
    list_entry_points,
)
from ragnarok_ai.plugins.exceptions import (
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
)
from ragnarok_ai.plugins.registry import (
    PluginInfo,
    PluginRegistry,
    get_framework_adapters,
    get_llm_adapters,
    get_vectorstore_adapters,
)

__all__ = [
    "ALL_NAMESPACES",
    "NAMESPACE_EVALUATOR",
    "NAMESPACE_FRAMEWORK",
    "NAMESPACE_LLM",
    "NAMESPACE_VECTORSTORE",
    "PluginError",
    "PluginInfo",
    "PluginLoadError",
    "PluginNotFoundError",
    "PluginRegistry",
    "discover_plugins",
    "discover_plugins_safe",
    "get_framework_adapters",
    "get_llm_adapters",
    "get_vectorstore_adapters",
    "list_entry_points",
]
