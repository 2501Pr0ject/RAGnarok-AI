"""Plugin registry for ragnarok-ai.

Provides a central registry for all available adapters, combining
builtin adapters with externally registered plugins.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from ragnarok_ai.plugins.discovery import (
    NAMESPACE_EVALUATOR,
    NAMESPACE_FRAMEWORK,
    NAMESPACE_LLM,
    NAMESPACE_VECTORSTORE,
    discover_plugins_safe,
)
from ragnarok_ai.plugins.exceptions import PluginNotFoundError, PluginUnavailableError

logger = logging.getLogger(__name__)

AdapterType = Literal["llm", "vectorstore", "framework", "evaluator"]


@dataclass(frozen=True)
class PluginInfo:
    """Metadata about a registered plugin.

    Attributes:
        name: Unique identifier for the plugin (e.g., "ollama", "groq").
        adapter_class: The adapter class implementing the protocol.
        adapter_type: Category of adapter ("llm", "vectorstore", etc.).
        is_local: Whether the adapter runs locally (no data leaves network).
        is_builtin: Whether this is a builtin adapter (shipped with ragnarok-ai).
        version: Optional version string.
        description: Optional human-readable description.
        available: Whether the adapter's dependencies are installed. When
            False, ``adapter_class`` is None.
        requires_extra: Extra providing the adapter's optional dependencies
            (``pip install ragnarok-ai[<extra>]``), for builtin adapters.
    """

    name: str
    adapter_class: type | None
    adapter_type: AdapterType
    is_local: bool
    is_builtin: bool = True
    version: str | None = None
    description: str | None = None
    available: bool = True
    requires_extra: str | None = None


@dataclass(frozen=True)
class _BuiltinAdapter:
    """Where a builtin adapter lives and what it needs to be importable."""

    name: str
    module: str
    class_name: str
    adapter_type: AdapterType
    is_local: bool
    extra: str
    requires: tuple[str, ...]


# Builtin adapters with the extra providing their optional dependencies and
# the top-level modules those dependencies install.
_BUILTIN_ADAPTERS: tuple[_BuiltinAdapter, ...] = (
    # LLM
    _BuiltinAdapter("ollama", "ragnarok_ai.adapters.llm.ollama", "OllamaLLM", "llm", True, "ollama", ("httpx",)),
    _BuiltinAdapter("vllm", "ragnarok_ai.adapters.llm.vllm", "VLLMAdapter", "llm", True, "vllm", ("httpx",)),
    _BuiltinAdapter(
        "anthropic", "ragnarok_ai.adapters.llm.anthropic", "AnthropicLLM", "llm", False, "anthropic", ("httpx",)
    ),
    _BuiltinAdapter("groq", "ragnarok_ai.adapters.llm.groq", "GroqLLM", "llm", False, "groq", ("httpx",)),
    _BuiltinAdapter("mistral", "ragnarok_ai.adapters.llm.mistral", "MistralLLM", "llm", False, "mistral", ("httpx",)),
    _BuiltinAdapter("openai", "ragnarok_ai.adapters.llm.openai", "OpenAILLM", "llm", False, "openai", ("httpx",)),
    _BuiltinAdapter(
        "together", "ragnarok_ai.adapters.llm.together", "TogetherLLM", "llm", False, "together", ("httpx",)
    ),
    # Vector stores
    _BuiltinAdapter(
        "chroma",
        "ragnarok_ai.adapters.vectorstore.chroma",
        "ChromaVectorStore",
        "vectorstore",
        True,
        "chroma",
        ("chromadb",),
    ),
    _BuiltinAdapter(
        "faiss",
        "ragnarok_ai.adapters.vectorstore.faiss",
        "FAISSVectorStore",
        "vectorstore",
        True,
        "faiss",
        ("faiss", "numpy"),
    ),
    _BuiltinAdapter(
        "milvus",
        "ragnarok_ai.adapters.vectorstore.milvus",
        "MilvusVectorStore",
        "vectorstore",
        True,
        "milvus",
        ("pymilvus",),
    ),
    _BuiltinAdapter(
        "pgvector",
        "ragnarok_ai.adapters.vectorstore.pgvector",
        "PgvectorVectorStore",
        "vectorstore",
        True,
        "pgvector",
        ("asyncpg", "pgvector"),
    ),
    _BuiltinAdapter(
        "qdrant",
        "ragnarok_ai.adapters.vectorstore.qdrant",
        "QdrantVectorStore",
        "vectorstore",
        True,
        "qdrant",
        ("qdrant_client",),
    ),
    _BuiltinAdapter(
        "pinecone",
        "ragnarok_ai.adapters.vectorstore.pinecone",
        "PineconeVectorStore",
        "vectorstore",
        False,
        "pinecone",
        ("pinecone",),
    ),
    _BuiltinAdapter(
        "weaviate",
        "ragnarok_ai.adapters.vectorstore.weaviate",
        "WeaviateVectorStore",
        "vectorstore",
        False,
        "weaviate",
        ("weaviate",),
    ),
)


def _is_importable(module: str) -> bool:
    """Whether a top-level module is installed, without importing it."""
    if module in sys.modules:
        return sys.modules[module] is not None  # None marks a blocked import
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class PluginRegistry:
    """Central registry for all available adapters.

    Combines builtin adapters with dynamically discovered plugins.
    Implements singleton pattern for global access.

    Example:
        >>> registry = PluginRegistry.get()
        >>> registry.discover()
        >>> for info in registry.list_adapters(adapter_type="llm"):
        ...     print(f"{info.name}: local={info.is_local}")
        ollama: local=True
        openai: local=False
        groq: local=False
    """

    _instance: ClassVar[PluginRegistry | None] = None

    _plugins: dict[str, PluginInfo] = field(default_factory=dict)
    _discovered: bool = field(default=False)

    @classmethod
    def get(cls) -> PluginRegistry:
        """Get the singleton registry instance.

        Returns:
            The global PluginRegistry instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Used for testing."""
        cls._instance = None

    def discover(self, *, force: bool = False) -> None:
        """Discover and register all available plugins.

        Loads builtin adapters first, then scans entry points for
        external plugins. External plugins with the same name as
        builtins will override them.

        Args:
            force: If True, re-discover even if already done.
        """
        if self._discovered and not force:
            return

        self._plugins.clear()
        self._register_builtins()
        self._discover_entry_points()
        self._discovered = True

        logger.info(f"Plugin discovery complete: {len(self._plugins)} adapters available")

    def _register_builtins(self) -> None:
        """Register all builtin adapters.

        Adapters whose optional dependencies are missing are registered as
        unavailable (with the extra to install) instead of failing discovery.
        """
        for spec in _BUILTIN_ADAPTERS:
            self._plugins[spec.name] = self._load_builtin(spec)

        logger.debug(f"Registered {len(self._plugins)} builtin adapters")

    @staticmethod
    def _load_builtin(spec: _BuiltinAdapter) -> PluginInfo:
        """Import a builtin adapter, or describe it as unavailable."""
        adapter_class: type | None = None
        if all(_is_importable(module) for module in spec.requires):
            try:
                adapter_class = getattr(importlib.import_module(spec.module), spec.class_name)
            except ImportError as e:
                logger.debug(f"Builtin adapter {spec.name} unavailable: {e}")

        return PluginInfo(
            name=spec.name,
            adapter_class=adapter_class,
            adapter_type=spec.adapter_type,
            is_local=spec.is_local,
            is_builtin=True,
            available=adapter_class is not None,
            requires_extra=spec.extra,
        )

    def _discover_entry_points(self) -> None:
        """Discover plugins from entry points."""
        namespace_map: dict[str, AdapterType] = {
            NAMESPACE_LLM: "llm",
            NAMESPACE_VECTORSTORE: "vectorstore",
            NAMESPACE_FRAMEWORK: "framework",
            NAMESPACE_EVALUATOR: "evaluator",
        }

        for namespace, adapter_type in namespace_map.items():
            plugins = discover_plugins_safe(namespace)
            for name, cls in plugins.items():
                # Determine if local by checking class attribute
                is_local = getattr(cls, "is_local", False)

                self._plugins[name] = PluginInfo(
                    name=name,
                    adapter_class=cls,
                    adapter_type=adapter_type,
                    is_local=is_local,
                    is_builtin=False,
                )
                logger.debug(f"Registered external plugin: {name} ({adapter_type})")

    def get_adapter(self, name: str) -> type:
        """Get an adapter class by name.

        Args:
            name: The plugin/adapter name.

        Returns:
            The adapter class.

        Raises:
            PluginNotFoundError: If the adapter is not found.
            PluginUnavailableError: If the adapter's optional dependencies
                are not installed.
        """
        if not self._discovered:
            self.discover()

        info = self._plugins.get(name.lower())
        if info is None:
            raise PluginNotFoundError(name)
        if info.adapter_class is None:
            raise PluginUnavailableError(info.name, info.requires_extra)
        return info.adapter_class

    def get_plugin_info(self, name: str) -> PluginInfo:
        """Get plugin metadata by name.

        Args:
            name: The plugin/adapter name.

        Returns:
            PluginInfo with metadata about the plugin.

        Raises:
            PluginNotFoundError: If the adapter is not found.
        """
        if not self._discovered:
            self.discover()

        info = self._plugins.get(name.lower())
        if info is None:
            raise PluginNotFoundError(name)
        return info

    def list_adapters(
        self,
        adapter_type: AdapterType | None = None,
        *,
        local_only: bool = False,
        builtin_only: bool = False,
        external_only: bool = False,
        available_only: bool = False,
    ) -> list[PluginInfo]:
        """List available adapters with optional filtering.

        Args:
            adapter_type: Filter by adapter type ("llm", "vectorstore", etc.).
            local_only: Only return local adapters.
            builtin_only: Only return builtin adapters.
            external_only: Only return external (plugin) adapters.
            available_only: Only return adapters whose dependencies are installed.

        Returns:
            List of PluginInfo objects matching the filters.

        Example:
            >>> registry = PluginRegistry.get()
            >>> registry.discover()
            >>> local_llms = registry.list_adapters(adapter_type="llm", local_only=True)
            >>> for info in local_llms:
            ...     print(info.name)
            ollama
            vllm
        """
        if not self._discovered:
            self.discover()

        plugins = list(self._plugins.values())

        if adapter_type:
            plugins = [p for p in plugins if p.adapter_type == adapter_type]

        if local_only:
            plugins = [p for p in plugins if p.is_local]

        if builtin_only:
            plugins = [p for p in plugins if p.is_builtin]

        if external_only:
            plugins = [p for p in plugins if not p.is_builtin]

        if available_only:
            plugins = [p for p in plugins if p.available]

        return sorted(plugins, key=lambda p: p.name)

    def is_registered(self, name: str) -> bool:
        """Check if a plugin is registered.

        Args:
            name: The plugin/adapter name.

        Returns:
            True if the plugin exists in the registry.
        """
        if not self._discovered:
            self.discover()
        return name.lower() in self._plugins


# Convenience functions for common operations
def get_llm_adapters() -> dict[str, type]:
    """Get all available LLM adapters.

    Returns:
        Dictionary mapping adapter names to classes.
    """
    registry = PluginRegistry.get()
    registry.discover()
    return {
        info.name: info.adapter_class
        for info in registry.list_adapters(adapter_type="llm", available_only=True)
        if info.adapter_class is not None
    }


def get_vectorstore_adapters() -> dict[str, type]:  # pragma: no cover
    """Get all available vector store adapters.

    Returns:
        Dictionary mapping adapter names to classes.
    """
    registry = PluginRegistry.get()
    registry.discover()
    return {
        info.name: info.adapter_class
        for info in registry.list_adapters(adapter_type="vectorstore", available_only=True)
        if info.adapter_class is not None
    }


def get_framework_adapters() -> dict[str, type]:  # pragma: no cover
    """Get all available framework adapters.

    Returns:
        Dictionary mapping adapter names to classes.
    """
    registry = PluginRegistry.get()
    registry.discover()
    return {
        info.name: info.adapter_class
        for info in registry.list_adapters(adapter_type="framework", available_only=True)
        if info.adapter_class is not None
    }
