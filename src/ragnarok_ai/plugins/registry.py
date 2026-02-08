"""Plugin registry for ragnarok-ai.

Provides a central registry for all available adapters, combining
builtin adapters with externally registered plugins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from ragnarok_ai.plugins.discovery import (
    NAMESPACE_EVALUATOR,
    NAMESPACE_FRAMEWORK,
    NAMESPACE_LLM,
    NAMESPACE_VECTORSTORE,
    discover_plugins_safe,
)
from ragnarok_ai.plugins.exceptions import PluginNotFoundError

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
    """

    name: str
    adapter_class: type
    adapter_type: AdapterType
    is_local: bool
    is_builtin: bool = True
    version: str | None = None
    description: str | None = None


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
        """Register all builtin adapters."""
        # Import here to avoid circular imports
        from ragnarok_ai.adapters import (
            CLOUD_LLM_ADAPTERS,
            CLOUD_VECTORSTORE_ADAPTERS,
            LOCAL_LLM_ADAPTERS,
            LOCAL_VECTORSTORE_ADAPTERS,
        )

        # LLM adapters
        for cls in LOCAL_LLM_ADAPTERS:
            name = self._class_to_name(cls)
            self._plugins[name] = PluginInfo(
                name=name,
                adapter_class=cls,
                adapter_type="llm",
                is_local=True,
                is_builtin=True,
            )

        for cls in CLOUD_LLM_ADAPTERS:
            name = self._class_to_name(cls)
            self._plugins[name] = PluginInfo(
                name=name,
                adapter_class=cls,
                adapter_type="llm",
                is_local=False,
                is_builtin=True,
            )

        # VectorStore adapters
        for cls in LOCAL_VECTORSTORE_ADAPTERS:
            name = self._class_to_name(cls)
            self._plugins[name] = PluginInfo(
                name=name,
                adapter_class=cls,
                adapter_type="vectorstore",
                is_local=True,
                is_builtin=True,
            )

        for cls in CLOUD_VECTORSTORE_ADAPTERS:
            name = self._class_to_name(cls)
            self._plugins[name] = PluginInfo(
                name=name,
                adapter_class=cls,
                adapter_type="vectorstore",
                is_local=False,
                is_builtin=True,
            )

        logger.debug(f"Registered {len(self._plugins)} builtin adapters")

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

    def _class_to_name(self, cls: type) -> str:
        """Convert class name to plugin name.

        OllamaLLM -> ollama
        OpenAILLM -> openai
        ChromaVectorStore -> chroma
        """
        name = cls.__name__
        # Remove common suffixes
        for suffix in ["LLM", "VectorStore", "Adapter", "Framework", "Evaluator"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name.lower()

    def get_adapter(self, name: str) -> type:
        """Get an adapter class by name.

        Args:
            name: The plugin/adapter name.

        Returns:
            The adapter class.

        Raises:
            PluginNotFoundError: If the adapter is not found.
        """
        if not self._discovered:
            self.discover()

        info = self._plugins.get(name.lower())
        if info is None:
            raise PluginNotFoundError(name)
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
    ) -> list[PluginInfo]:
        """List available adapters with optional filtering.

        Args:
            adapter_type: Filter by adapter type ("llm", "vectorstore", etc.).
            local_only: Only return local adapters.
            builtin_only: Only return builtin adapters.
            external_only: Only return external (plugin) adapters.

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
    return {info.name: info.adapter_class for info in registry.list_adapters(adapter_type="llm")}


def get_vectorstore_adapters() -> dict[str, type]:
    """Get all available vector store adapters.

    Returns:
        Dictionary mapping adapter names to classes.
    """
    registry = PluginRegistry.get()
    registry.discover()
    return {info.name: info.adapter_class for info in registry.list_adapters(adapter_type="vectorstore")}


def get_framework_adapters() -> dict[str, type]:
    """Get all available framework adapters.

    Returns:
        Dictionary mapping adapter names to classes.
    """
    registry = PluginRegistry.get()
    registry.discover()
    return {info.name: info.adapter_class for info in registry.list_adapters(adapter_type="framework")}
