"""Tests for plugin discovery."""

from unittest.mock import patch

import pytest

from ragnarok_ai.plugins.discovery import discover_plugins, discover_plugins_safe
from ragnarok_ai.plugins.exceptions import PluginLoadError


class MockEntryPoint:
    """Mock entry point for testing."""

    def __init__(self, name: str, load_result: type | Exception) -> None:
        self.name = name
        self._load_result = load_result

    def load(self) -> type:
        if isinstance(self._load_result, Exception):
            raise self._load_result
        return self._load_result


class TestDiscoverPlugins:
    """Tests for discover_plugins function."""

    def test_discover_plugins_success(self) -> None:
        """Test successful plugin discovery."""

        class FakePlugin:
            pass

        mock_eps = [MockEntryPoint("fake", FakePlugin)]

        with patch(
            "ragnarok_ai.plugins.discovery._get_entry_points",
            return_value=mock_eps,
        ):
            plugins = discover_plugins("test.namespace")

        assert "fake" in plugins
        assert plugins["fake"] is FakePlugin

    def test_discover_plugins_load_error(self) -> None:
        """Test plugin discovery with load error."""
        mock_eps = [MockEntryPoint("broken", ImportError("Module not found"))]

        with (
            patch(
                "ragnarok_ai.plugins.discovery._get_entry_points",
                return_value=mock_eps,
            ),
            pytest.raises(PluginLoadError) as exc_info,
        ):
            discover_plugins("test.namespace")

        assert "broken" in str(exc_info.value)

    def test_discover_plugins_empty(self) -> None:
        """Test discovery with no plugins."""
        with patch(
            "ragnarok_ai.plugins.discovery._get_entry_points",
            return_value=[],
        ):
            plugins = discover_plugins("test.namespace")

        assert plugins == {}


class TestDiscoverPluginsSafe:
    """Tests for discover_plugins_safe function."""

    def test_discover_plugins_safe_success(self) -> None:
        """Test safe discovery with working plugin."""

        class FakePlugin:
            pass

        mock_eps = [MockEntryPoint("fake", FakePlugin)]

        with patch(
            "ragnarok_ai.plugins.discovery._get_entry_points",
            return_value=mock_eps,
        ):
            plugins = discover_plugins_safe("test.namespace")

        assert "fake" in plugins

    def test_discover_plugins_safe_ignores_errors(self) -> None:
        """Test safe discovery ignores load errors."""

        class GoodPlugin:
            pass

        mock_eps = [
            MockEntryPoint("good", GoodPlugin),
            MockEntryPoint("broken", ImportError("Module not found")),
        ]

        with patch(
            "ragnarok_ai.plugins.discovery._get_entry_points",
            return_value=mock_eps,
        ):
            plugins = discover_plugins_safe("test.namespace")

        assert "good" in plugins
        assert "broken" not in plugins
