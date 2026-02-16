"""Slack alert adapter.

Sends alerts to Slack via incoming webhook using Block Kit formatting.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import httpx

from ragnarok_ai.alerts.protocols import Alert, AlertResult, AlertSeverity

logger = logging.getLogger(__name__)


class SlackAlertAdapter:
    """Alert adapter that sends alerts to Slack.

    Uses Slack's incoming webhooks with Block Kit formatting for
    rich alert messages.

    Attributes:
        webhook_url: Slack incoming webhook URL.
        channel: Optional channel override.
        username: Bot username for the message.
        timeout: Request timeout in seconds.

    Example:
        >>> adapter = SlackAlertAdapter(
        ...     webhook_url="https://hooks.slack.com/services/...",
        ...     channel="#alerts",
        ... )
        >>> result = await adapter.send(alert)
    """

    # Emoji mapping for severity levels
    SEVERITY_EMOJI: ClassVar[dict[AlertSeverity, str]] = {
        AlertSeverity.INFO: ":information_source:",
        AlertSeverity.WARNING: ":warning:",
        AlertSeverity.CRITICAL: ":rotating_light:",
    }

    # Color mapping for severity levels (Slack attachment colors)
    SEVERITY_COLOR: ClassVar[dict[AlertSeverity, str]] = {
        AlertSeverity.INFO: "#36a64f",  # Green
        AlertSeverity.WARNING: "#ff9800",  # Orange
        AlertSeverity.CRITICAL: "#dc3545",  # Red
    }

    def __init__(
        self,
        webhook_url: str,
        channel: str | None = None,
        username: str = "RAGnarok Alerts",
        timeout: float = 10.0,
    ) -> None:
        """Initialize the Slack adapter.

        Args:
            webhook_url: Slack incoming webhook URL.
            channel: Optional channel override (e.g., "#alerts").
            username: Bot username for the message.
            timeout: Request timeout in seconds.
        """
        self._webhook_url = webhook_url
        self._channel = channel
        self._username = username
        self._timeout = timeout

    @property
    def name(self) -> str:
        """Return the adapter name."""
        return "slack"

    def _build_payload(self, alert: Alert) -> dict[str, Any]:
        """Build the Slack Block Kit payload.

        Args:
            alert: The alert to send.

        Returns:
            Slack message payload with Block Kit blocks.
        """
        emoji = self.SEVERITY_EMOJI.get(alert.severity, ":bell:")
        color = self.SEVERITY_COLOR.get(alert.severity, "#808080")

        # Build blocks
        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {alert.title}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert.message,
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:* {alert.severity.value.upper()} | *Source:* {alert.source}",
                    },
                ],
            },
        ]

        # Add metadata fields if present
        if alert.metadata:
            fields = []
            for key, value in alert.metadata.items():
                if key not in ("rule_name",):  # Skip redundant fields
                    fields.append(
                        {
                            "type": "mrkdwn",
                            "text": f"*{key}:* {value}",
                        }
                    )
            if fields:
                blocks.append(
                    {
                        "type": "section",
                        "fields": fields[:10],  # Slack limits to 10 fields
                    }
                )

        # Add timestamp divider
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"<!date^{int(alert.timestamp.timestamp())}^{{date_short_pretty}} at {{time}}|{alert.timestamp.isoformat()}>",
                    },
                ],
            }
        )

        payload: dict[str, Any] = {
            "username": self._username,
            "blocks": blocks,
            "attachments": [
                {
                    "color": color,
                    "blocks": [],
                }
            ],
        }

        if self._channel:
            payload["channel"] = self._channel

        return payload

    async def send(self, alert: Alert) -> AlertResult:
        """Send an alert to Slack.

        Args:
            alert: The alert to send.

        Returns:
            Result indicating success or failure.
        """
        payload = self._build_payload(alert)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()

                # Slack returns "ok" for success
                if response.text == "ok":
                    logger.debug(f"Slack alert sent: {alert.title}")
                    return AlertResult(success=True, adapter=self.name)
                else:
                    error_msg = f"Slack error: {response.text}"
                    logger.warning(f"Slack alert may have failed: {error_msg}")
                    return AlertResult(success=False, adapter=self.name, error=error_msg)

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.error(f"Slack alert failed: {error_msg}")
            return AlertResult(success=False, adapter=self.name, error=error_msg)

        except httpx.RequestError as e:
            error_msg = f"Request failed: {e}"
            logger.error(f"Slack alert failed: {error_msg}")
            return AlertResult(success=False, adapter=self.name, error=error_msg)

        except Exception as e:  # pragma: no cover
            error_msg = f"Unexpected error: {e}"
            logger.error(f"Slack alert failed: {error_msg}")
            return AlertResult(success=False, adapter=self.name, error=error_msg)
