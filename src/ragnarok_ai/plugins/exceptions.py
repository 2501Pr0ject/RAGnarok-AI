"""Plugin-specific exceptions for ragnarok-ai."""

from __future__ import annotations


class PluginError(Exception):
    """Base exception for plugin-related errors."""


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load.

    This typically occurs when:
    - The entry point is malformed
    - The module cannot be imported
    - The class doesn't implement the required protocol
    """

    def __init__(self, plugin_name: str, reason: str) -> None:
        """Initialize PluginLoadError.

        Args:
            plugin_name: Name of the plugin that failed to load.
            reason: Description of why the plugin failed to load.
        """
        self.plugin_name = plugin_name
        self.reason = reason
        super().__init__(f"Failed to load plugin '{plugin_name}': {reason}")


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin is not found.

    This occurs when:
    - The plugin name doesn't exist in the registry
    - The plugin was never installed or registered
    """

    def __init__(self, plugin_name: str, adapter_type: str | None = None) -> None:
        """Initialize PluginNotFoundError.

        Args:
            plugin_name: Name of the plugin that was not found.
            adapter_type: Optional adapter type for more specific error messages.
        """
        self.plugin_name = plugin_name
        self.adapter_type = adapter_type
        if adapter_type:
            msg = f"Plugin '{plugin_name}' not found in {adapter_type} adapters"
        else:
            msg = f"Plugin '{plugin_name}' not found"
        super().__init__(msg)


class PluginUnavailableError(PluginError, ImportError):
    """Raised when a registered plugin cannot be used in this environment.

    This occurs when a builtin adapter's optional dependencies are not
    installed (e.g. httpx for the Ollama adapter on a base install).
    """

    def __init__(self, plugin_name: str, extra: str | None = None) -> None:
        """Initialize PluginUnavailableError.

        Args:
            plugin_name: Name of the unavailable plugin.
            extra: Optional extra providing the missing dependencies.
        """
        self.plugin_name = plugin_name
        self.extra = extra
        msg = f"Plugin '{plugin_name}' is not available: missing optional dependencies"
        if extra:
            msg += f" (install with: pip install 'ragnarok-ai[{extra}]')"
        super().__init__(msg)
