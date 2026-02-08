"""Entry point discovery for ragnarok-ai plugins.

This module provides utilities for discovering plugins registered via
Python entry points (setuptools/importlib.metadata).

Plugin authors can register their adapters by adding entry points
to their pyproject.toml:

    [project.entry-points."ragnarok_ai.adapters.llm"]
    groq = "ragnarok_ai_groq:GroqLLM"
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Entry point group namespaces
NAMESPACE_LLM = "ragnarok_ai.adapters.llm"
NAMESPACE_VECTORSTORE = "ragnarok_ai.adapters.vectorstore"
NAMESPACE_FRAMEWORK = "ragnarok_ai.adapters.framework"
NAMESPACE_EVALUATOR = "ragnarok_ai.adapters.evaluator"

ALL_NAMESPACES = [
    NAMESPACE_LLM,
    NAMESPACE_VECTORSTORE,
    NAMESPACE_FRAMEWORK,
    NAMESPACE_EVALUATOR,
]


def _get_entry_points(group: str) -> Iterator[Any]:
    """Get entry points for a group.

    Args:
        group: The entry point group name.

    Yields:
        Entry point objects.
    """
    from importlib.metadata import entry_points

    eps = entry_points(group=group)
    yield from eps


def discover_plugins(namespace: str) -> dict[str, type]:
    """Discover all plugins registered for a namespace.

    Scans Python entry points for plugins registered under the given namespace
    and attempts to load each one. Failed loads are logged as warnings.

    Args:
        namespace: The entry point group to scan (e.g., "ragnarok_ai.adapters.llm").

    Returns:
        Dictionary mapping plugin names to their adapter classes.

    Example:
        >>> plugins = discover_plugins("ragnarok_ai.adapters.llm")
        >>> for name, cls in plugins.items():
        ...     print(f"{name}: {cls}")
        groq: <class 'ragnarok_ai_groq.GroqLLM'>
    """
    from ragnarok_ai.plugins.exceptions import PluginLoadError

    plugins: dict[str, type] = {}

    for ep in _get_entry_points(namespace):
        try:
            plugin_class = ep.load()
            plugins[ep.name] = plugin_class
            logger.debug(f"Loaded plugin '{ep.name}' from {namespace}")
        except Exception as e:
            logger.warning(f"Failed to load plugin '{ep.name}' from {namespace}: {e}")
            # Re-raise as PluginLoadError for callers who want to handle it
            raise PluginLoadError(ep.name, str(e)) from e

    return plugins


def discover_plugins_safe(namespace: str) -> dict[str, type]:
    """Discover plugins without raising exceptions for failed loads.

    Like discover_plugins, but catches and logs all errors without
    re-raising them. Use this when you want to load all available
    plugins even if some fail.

    Args:
        namespace: The entry point group to scan.

    Returns:
        Dictionary mapping successfully loaded plugin names to classes.

    Example:
        >>> plugins = discover_plugins_safe("ragnarok_ai.adapters.llm")
        >>> print(f"Loaded {len(plugins)} plugins")
        Loaded 3 plugins
    """
    plugins: dict[str, type] = {}

    for ep in _get_entry_points(namespace):
        try:
            plugin_class = ep.load()
            plugins[ep.name] = plugin_class
            logger.debug(f"Loaded plugin '{ep.name}' from {namespace}")
        except Exception as e:
            logger.warning(f"Failed to load plugin '{ep.name}' from {namespace}: {e}")
            # Continue loading other plugins

    return plugins


def list_entry_points(namespace: str) -> list[str]:
    """List entry point names registered for a namespace without loading them.

    This is faster than discover_plugins when you only need the names.

    Args:
        namespace: The entry point group to scan.

    Returns:
        List of registered plugin names.

    Example:
        >>> names = list_entry_points("ragnarok_ai.adapters.llm")
        >>> print(names)
        ['groq', 'together', 'fireworks']
    """
    return [ep.name for ep in _get_entry_points(namespace)]
