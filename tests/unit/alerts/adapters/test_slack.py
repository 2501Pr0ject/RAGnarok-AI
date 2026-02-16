"""Tests for SlackAlertAdapter."""

import json

import httpx
import pytest
import respx

from ragnarok_ai.alerts.adapters.slack import SlackAlertAdapter
from ragnarok_ai.alerts.protocols import Alert, AlertSeverity


@pytest.fixture
def alert() -> Alert:
    """Create a test alert."""
    return Alert(
        title="Test Alert",
        message="Test message",
        severity=AlertSeverity.WARNING,
        source="threshold",
        metadata={"metric": "precision", "value": 0.65},
    )


@pytest.fixture
def adapter() -> SlackAlertAdapter:
    """Create a test adapter."""
    return SlackAlertAdapter(
        webhook_url="https://hooks.slack.com/services/XXX/YYY/ZZZ",
        channel="#alerts",
        username="Test Bot",
        timeout=5.0,
    )


class TestSlackAlertAdapter:
    """Tests for SlackAlertAdapter."""

    def test_adapter_name(self, adapter: SlackAlertAdapter) -> None:
        """Test adapter name property."""
        assert adapter.name == "slack"

    def test_severity_emoji_mapping(self) -> None:
        """Test severity to emoji mapping."""
        assert ":information_source:" in SlackAlertAdapter.SEVERITY_EMOJI[AlertSeverity.INFO]
        assert ":warning:" in SlackAlertAdapter.SEVERITY_EMOJI[AlertSeverity.WARNING]
        assert ":rotating_light:" in SlackAlertAdapter.SEVERITY_EMOJI[AlertSeverity.CRITICAL]

    def test_severity_color_mapping(self) -> None:
        """Test severity to color mapping."""
        assert SlackAlertAdapter.SEVERITY_COLOR[AlertSeverity.INFO] == "#36a64f"
        assert SlackAlertAdapter.SEVERITY_COLOR[AlertSeverity.WARNING] == "#ff9800"
        assert SlackAlertAdapter.SEVERITY_COLOR[AlertSeverity.CRITICAL] == "#dc3545"


class TestSlackAlertAdapterPayload:
    """Tests for SlackAlertAdapter payload building."""

    def test_build_payload_structure(self, adapter: SlackAlertAdapter, alert: Alert) -> None:
        """Test payload has correct structure."""
        payload = adapter._build_payload(alert)

        assert "username" in payload
        assert "blocks" in payload
        assert "attachments" in payload
        assert payload["username"] == "Test Bot"
        assert payload["channel"] == "#alerts"

    def test_build_payload_blocks(self, adapter: SlackAlertAdapter, alert: Alert) -> None:
        """Test payload blocks contain expected content."""
        payload = adapter._build_payload(alert)
        blocks = payload["blocks"]

        # Header block
        header_block = blocks[0]
        assert header_block["type"] == "header"
        assert "Test Alert" in header_block["text"]["text"]

        # Section block with message
        section_block = blocks[1]
        assert section_block["type"] == "section"
        assert section_block["text"]["text"] == "Test message"

        # Context block with severity
        context_block = blocks[2]
        assert context_block["type"] == "context"
        assert "WARNING" in context_block["elements"][0]["text"]

    def test_build_payload_with_metadata(self, adapter: SlackAlertAdapter, alert: Alert) -> None:
        """Test that metadata fields are included."""
        payload = adapter._build_payload(alert)

        # Find metadata section
        metadata_found = False
        for block in payload["blocks"]:
            if block["type"] == "section" and "fields" in block:
                metadata_found = True
                fields_text = json.dumps(block["fields"])
                assert "metric" in fields_text
                assert "value" in fields_text

        assert metadata_found

    def test_build_payload_without_channel(self) -> None:
        """Test payload without channel override."""
        adapter = SlackAlertAdapter(
            webhook_url="https://hooks.slack.com/services/XXX/YYY/ZZZ",
        )
        alert = Alert(
            title="Test",
            message="Message",
            severity=AlertSeverity.INFO,
            source="test",
        )
        payload = adapter._build_payload(alert)

        assert "channel" not in payload


class TestSlackAlertAdapterSend:
    """Tests for SlackAlertAdapter.send()."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_success(self, adapter: SlackAlertAdapter, alert: Alert) -> None:
        """Test successful alert sending."""
        respx.post("https://hooks.slack.com/services/XXX/YYY/ZZZ").mock(return_value=httpx.Response(200, text="ok"))

        result = await adapter.send(alert)

        assert result.success is True
        assert result.adapter == "slack"
        assert result.error is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_slack_error_response(self, adapter: SlackAlertAdapter, alert: Alert) -> None:
        """Test handling Slack error response (200 but not 'ok')."""
        respx.post("https://hooks.slack.com/services/XXX/YYY/ZZZ").mock(
            return_value=httpx.Response(200, text="invalid_token")
        )

        result = await adapter.send(alert)

        assert result.success is False
        assert result.adapter == "slack"
        assert "invalid_token" in result.error  # type: ignore[operator]

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_http_error(self, adapter: SlackAlertAdapter, alert: Alert) -> None:
        """Test handling HTTP error responses."""
        respx.post("https://hooks.slack.com/services/XXX/YYY/ZZZ").mock(
            return_value=httpx.Response(404, text="Not Found")
        )

        result = await adapter.send(alert)

        assert result.success is False
        assert result.adapter == "slack"
        assert "404" in result.error  # type: ignore[operator]

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_connection_error(self, adapter: SlackAlertAdapter, alert: Alert) -> None:
        """Test handling connection errors."""
        respx.post("https://hooks.slack.com/services/XXX/YYY/ZZZ").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        result = await adapter.send(alert)

        assert result.success is False
        assert result.adapter == "slack"
        assert "Connection" in result.error  # type: ignore[operator]

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_request_body(self, adapter: SlackAlertAdapter, alert: Alert) -> None:
        """Test that request body is correctly formatted."""
        route = respx.post("https://hooks.slack.com/services/XXX/YYY/ZZZ").mock(
            return_value=httpx.Response(200, text="ok")
        )

        await adapter.send(alert)

        assert route.called
        request = route.calls[0].request
        body = json.loads(request.content)

        assert body["username"] == "Test Bot"
        assert body["channel"] == "#alerts"
        assert "blocks" in body
