"""Tests for plugin system."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from ragnarok_ai.plugins import (
    NAMESPACE_LLM,
    PluginInfo,
    PluginLoadError,
    PluginNotFoundError,
    PluginRegistry,
    discover_plugins_safe,
    get_llm_adapters,
    list_entry_points,
)

# =============================================================================
# PluginInfo Tests
# =============================================================================


class TestPluginInfo:
    """Tests for PluginInfo dataclass."""

    def test_plugin_info_creation(self):
        """Should create PluginInfo with required fields."""

        class MockAdapter:
            pass

        info = PluginInfo(
            name="test",
            adapter_class=MockAdapter,
            adapter_type="llm",
            is_local=True,
        )

        assert info.name == "test"
        assert info.adapter_class is MockAdapter
        assert info.adapter_type == "llm"
        assert info.is_local is True
        assert info.is_builtin is True  # default
        assert info.version is None
        assert info.description is None

    def test_plugin_info_with_optional_fields(self):
        """Should create PluginInfo with optional fields."""

        class MockAdapter:
            pass

        info = PluginInfo(
            name="custom",
            adapter_class=MockAdapter,
            adapter_type="vectorstore",
            is_local=False,
            is_builtin=False,
            version="1.0.0",
            description="A custom adapter",
        )

        assert info.name == "custom"
        assert info.is_builtin is False
        assert info.version == "1.0.0"
        assert info.description == "A custom adapter"

    def test_plugin_info_is_frozen(self):
        """PluginInfo should be immutable."""

        class MockAdapter:
            pass

        info = PluginInfo(
            name="test",
            adapter_class=MockAdapter,
            adapter_type="llm",
            is_local=True,
        )

        with pytest.raises(FrozenInstanceError):
            info.name = "changed"  # type: ignore[misc]


# =============================================================================
# PluginRegistry Tests
# =============================================================================


class TestPluginRegistry:
    """Tests for PluginRegistry class."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry singleton before each test."""
        PluginRegistry.reset()
        yield
        PluginRegistry.reset()

    def test_singleton_pattern(self):
        """Should return same instance on multiple calls."""
        registry1 = PluginRegistry.get()
        registry2 = PluginRegistry.get()

        assert registry1 is registry2

    def test_reset_clears_singleton(self):
        """Reset should clear the singleton instance."""
        registry1 = PluginRegistry.get()
        PluginRegistry.reset()
        registry2 = PluginRegistry.get()

        assert registry1 is not registry2

    def test_discover_registers_builtins(self):
        """Discover should register builtin adapters."""
        registry = PluginRegistry.get()
        registry.discover()

        # Check some known builtins
        assert registry.is_registered("ollama")
        assert registry.is_registered("openai")
        assert registry.is_registered("chroma")

    def test_discover_only_runs_once(self):
        """Discover should not re-run if already done."""
        registry = PluginRegistry.get()

        with patch.object(registry, "_register_builtins") as mock:
            registry.discover()
            registry.discover()

            # Should only be called once
            assert mock.call_count == 1

    def test_discover_force_reruns(self):
        """Discover with force=True should re-run."""
        registry = PluginRegistry.get()
        registry.discover()

        with patch.object(registry, "_register_builtins") as mock:
            registry.discover(force=True)
            assert mock.call_count == 1

    def test_get_adapter_returns_class(self):
        """get_adapter should return the adapter class."""
        from ragnarok_ai.adapters.llm import OllamaLLM

        registry = PluginRegistry.get()
        registry.discover()

        adapter = registry.get_adapter("ollama")
        assert adapter is OllamaLLM

    def test_get_adapter_case_insensitive(self):
        """get_adapter should be case insensitive."""
        registry = PluginRegistry.get()
        registry.discover()

        adapter1 = registry.get_adapter("ollama")
        adapter2 = registry.get_adapter("OLLAMA")
        adapter3 = registry.get_adapter("Ollama")

        assert adapter1 is adapter2 is adapter3

    def test_get_adapter_not_found(self):
        """get_adapter should raise PluginNotFoundError."""
        registry = PluginRegistry.get()
        registry.discover()

        with pytest.raises(PluginNotFoundError) as exc_info:
            registry.get_adapter("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_get_plugin_info(self):
        """get_plugin_info should return PluginInfo."""
        registry = PluginRegistry.get()
        registry.discover()

        info = registry.get_plugin_info("ollama")

        assert info.name == "ollama"
        assert info.adapter_type == "llm"
        assert info.is_local is True
        assert info.is_builtin is True

    def test_list_adapters_all(self):
        """list_adapters should return all adapters."""
        registry = PluginRegistry.get()
        registry.discover()

        adapters = registry.list_adapters()

        assert len(adapters) > 0
        assert any(a.name == "ollama" for a in adapters)
        assert any(a.name == "openai" for a in adapters)

    def test_list_adapters_by_type(self):
        """list_adapters should filter by type."""
        registry = PluginRegistry.get()
        registry.discover()

        llm_adapters = registry.list_adapters(adapter_type="llm")
        vectorstore_adapters = registry.list_adapters(adapter_type="vectorstore")

        assert all(a.adapter_type == "llm" for a in llm_adapters)
        assert all(a.adapter_type == "vectorstore" for a in vectorstore_adapters)

    def test_list_adapters_local_only(self):
        """list_adapters should filter local only."""
        registry = PluginRegistry.get()
        registry.discover()

        local_adapters = registry.list_adapters(local_only=True)

        assert all(a.is_local for a in local_adapters)
        assert any(a.name == "ollama" for a in local_adapters)
        assert not any(a.name == "openai" for a in local_adapters)

    def test_list_adapters_builtin_only(self):
        """list_adapters should filter builtin only."""
        registry = PluginRegistry.get()
        registry.discover()

        builtin_adapters = registry.list_adapters(builtin_only=True)

        assert all(a.is_builtin for a in builtin_adapters)

    def test_list_adapters_sorted_by_name(self):
        """list_adapters should return sorted results."""
        registry = PluginRegistry.get()
        registry.discover()

        adapters = registry.list_adapters()
        names = [a.name for a in adapters]

        assert names == sorted(names)

    def test_is_registered(self):
        """is_registered should check if plugin exists."""
        registry = PluginRegistry.get()
        registry.discover()

        assert registry.is_registered("ollama") is True
        assert registry.is_registered("nonexistent") is False


# =============================================================================
# Discovery Tests
# =============================================================================


class TestDiscovery:
    """Tests for plugin discovery functions."""

    def test_discover_plugins_safe_returns_empty_for_nonexistent_namespace(self):
        """discover_plugins_safe should return empty dict for unknown namespace."""
        plugins = discover_plugins_safe("nonexistent.namespace")
        assert plugins == {}

    def test_list_entry_points_returns_list(self):
        """list_entry_points should return a list of names."""
        # This will return empty for a namespace with no plugins
        names = list_entry_points("ragnarok_ai.test.nonexistent")
        assert isinstance(names, list)
        assert len(names) == 0


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        PluginRegistry.reset()
        yield
        PluginRegistry.reset()

    def test_get_llm_adapters(self):
        """get_llm_adapters should return dict of LLM adapters."""
        adapters = get_llm_adapters()

        assert isinstance(adapters, dict)
        assert "ollama" in adapters
        assert "openai" in adapters

    def test_get_llm_adapters_returns_classes(self):
        """get_llm_adapters should return adapter classes."""
        from ragnarok_ai.adapters.llm import OllamaLLM

        adapters = get_llm_adapters()

        assert adapters["ollama"] is OllamaLLM


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for plugin exceptions."""

    def test_plugin_load_error(self):
        """PluginLoadError should contain plugin name and reason."""
        error = PluginLoadError("myplugin", "import failed")

        assert error.plugin_name == "myplugin"
        assert error.reason == "import failed"
        assert "myplugin" in str(error)
        assert "import failed" in str(error)

    def test_plugin_not_found_error(self):
        """PluginNotFoundError should contain plugin name."""
        error = PluginNotFoundError("myplugin")

        assert error.plugin_name == "myplugin"
        assert "myplugin" in str(error)

    def test_plugin_not_found_error_with_type(self):
        """PluginNotFoundError should include adapter type if provided."""
        error = PluginNotFoundError("myplugin", adapter_type="llm")

        assert error.plugin_name == "myplugin"
        assert error.adapter_type == "llm"
        assert "myplugin" in str(error)
        assert "llm" in str(error)


# =============================================================================
# Mock Entry Point Tests
# =============================================================================


class TestMockEntryPoints:
    """Tests with mocked entry points."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        PluginRegistry.reset()
        yield
        PluginRegistry.reset()

    def test_external_plugin_discovery(self):
        """Should discover external plugins from entry points."""

        class MockGroqLLM:
            is_local = False

        mock_ep = MagicMock()
        mock_ep.name = "groq"
        mock_ep.load.return_value = MockGroqLLM

        with patch(
            "ragnarok_ai.plugins.discovery._get_entry_points",
            return_value=[mock_ep],
        ):
            plugins = discover_plugins_safe(NAMESPACE_LLM)

            assert "groq" in plugins
            assert plugins["groq"] is MockGroqLLM

    def test_external_plugin_in_registry(self):
        """External plugins should appear in registry."""

        class MockGroqLLM:
            is_local = False

        mock_ep = MagicMock()
        mock_ep.name = "groq"
        mock_ep.load.return_value = MockGroqLLM

        with patch(
            "ragnarok_ai.plugins.discovery._get_entry_points",
            return_value=[mock_ep],
        ):
            registry = PluginRegistry.get()
            registry.discover()

            assert registry.is_registered("groq")
            info = registry.get_plugin_info("groq")
            assert info.is_builtin is False
            assert info.is_local is False

    def test_failed_plugin_load_logged(self):
        """Failed plugin loads should be logged and skipped."""
        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = ImportError("module not found")

        with patch(
            "ragnarok_ai.plugins.discovery._get_entry_points",
            return_value=[mock_ep],
        ):
            # Should not raise, just skip the broken plugin
            plugins = discover_plugins_safe(NAMESPACE_LLM)

            assert "broken" not in plugins
