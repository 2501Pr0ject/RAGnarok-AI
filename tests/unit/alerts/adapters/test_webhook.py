"""Tests for WebhookAlertAdapter."""

import httpx
import pytest
import respx

from ragnarok_ai.alerts.adapters.webhook import WebhookAlertAdapter
from ragnarok_ai.alerts.protocols import Alert, AlertSeverity


@pytest.fixture
def alert() -> Alert:
    """Create a test alert."""
    return Alert(
        title="Test Alert",
        message="Test message",
        severity=AlertSeverity.WARNING,
        source="test",
        metadata={"key": "value"},
    )


@pytest.fixture
def adapter() -> WebhookAlertAdapter:
    """Create a test adapter."""
    return WebhookAlertAdapter(
        url="https://example.com/webhook",
        headers={"X-Custom": "header"},
        timeout=5.0,
    )


class TestWebhookAlertAdapter:
    """Tests for WebhookAlertAdapter."""

    def test_adapter_name(self, adapter: WebhookAlertAdapter) -> None:
        """Test adapter name property."""
        assert adapter.name == "webhook"

    def test_build_payload(self, adapter: WebhookAlertAdapter, alert: Alert) -> None:
        """Test payload building."""
        payload = adapter._build_payload(alert)

        assert "alert" in payload
        assert payload["alert"]["title"] == "Test Alert"
        assert payload["alert"]["message"] == "Test message"
        assert payload["alert"]["severity"] == "warning"

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_success(self, adapter: WebhookAlertAdapter, alert: Alert) -> None:
        """Test successful alert sending."""
        respx.post("https://example.com/webhook").mock(return_value=httpx.Response(200, text="OK"))

        result = await adapter.send(alert)

        assert result.success is True
        assert result.adapter == "webhook"
        assert result.error is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_http_error(self, adapter: WebhookAlertAdapter, alert: Alert) -> None:
        """Test handling HTTP error responses."""
        respx.post("https://example.com/webhook").mock(return_value=httpx.Response(500, text="Internal Server Error"))

        result = await adapter.send(alert)

        assert result.success is False
        assert result.adapter == "webhook"
        assert "500" in result.error  # type: ignore[operator]

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_connection_error(self, adapter: WebhookAlertAdapter, alert: Alert) -> None:
        """Test handling connection errors."""
        respx.post("https://example.com/webhook").mock(side_effect=httpx.ConnectError("Connection refused"))

        result = await adapter.send(alert)

        assert result.success is False
        assert result.adapter == "webhook"
        assert "Connection" in result.error  # type: ignore[operator]

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_timeout(self, adapter: WebhookAlertAdapter, alert: Alert) -> None:
        """Test handling timeout errors."""
        respx.post("https://example.com/webhook").mock(side_effect=httpx.ReadTimeout("Read timed out"))

        result = await adapter.send(alert)

        assert result.success is False
        assert result.adapter == "webhook"
        assert result.error is not None

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_includes_custom_headers(self, alert: Alert) -> None:
        """Test that custom headers are included."""
        adapter = WebhookAlertAdapter(
            url="https://example.com/webhook",
            headers={"Authorization": "Bearer token123"},
        )

        route = respx.post("https://example.com/webhook").mock(return_value=httpx.Response(200, text="OK"))

        await adapter.send(alert)

        assert route.called
        request = route.calls[0].request
        assert request.headers["Authorization"] == "Bearer token123"
        assert request.headers["Content-Type"] == "application/json"
