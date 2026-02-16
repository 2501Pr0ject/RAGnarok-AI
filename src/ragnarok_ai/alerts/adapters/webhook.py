"""Webhook alert adapter.

Sends alerts via HTTP POST to a configurable webhook URL.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ragnarok_ai.alerts.protocols import Alert, AlertResult

logger = logging.getLogger(__name__)


class WebhookAlertAdapter:
    """Alert adapter that sends alerts via HTTP webhook.

    Sends a JSON POST request to the configured URL with alert data.

    Attributes:
        url: The webhook URL to POST to.
        headers: Optional HTTP headers to include.
        timeout: Request timeout in seconds.

    Example:
        >>> adapter = WebhookAlertAdapter(
        ...     url="https://my-service.com/alerts",
        ...     headers={"Authorization": "Bearer token"},
        ... )
        >>> result = await adapter.send(alert)
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        """Initialize the webhook adapter.

        Args:
            url: The webhook URL to POST to.
            headers: Optional HTTP headers to include.
            timeout: Request timeout in seconds.
        """
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout

    @property
    def name(self) -> str:
        """Return the adapter name."""
        return "webhook"

    def _build_payload(self, alert: Alert) -> dict[str, Any]:
        """Build the JSON payload for the webhook.

        Args:
            alert: The alert to send.

        Returns:
            Dictionary payload for the POST request.
        """
        return {
            "alert": alert.to_dict(),
        }

    async def send(self, alert: Alert) -> AlertResult:
        """Send an alert via webhook.

        Args:
            alert: The alert to send.

        Returns:
            Result indicating success or failure.
        """
        payload = self._build_payload(alert)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        **self._headers,
                    },
                )
                response.raise_for_status()

            logger.debug(f"Webhook alert sent: {alert.title}")
            return AlertResult(success=True, adapter=self.name)

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.error(f"Webhook alert failed: {error_msg}")
            return AlertResult(success=False, adapter=self.name, error=error_msg)

        except httpx.RequestError as e:
            error_msg = f"Request failed: {e}"
            logger.error(f"Webhook alert failed: {error_msg}")
            return AlertResult(success=False, adapter=self.name, error=error_msg)

        except Exception as e:  # pragma: no cover
            error_msg = f"Unexpected error: {e}"
            logger.error(f"Webhook alert failed: {error_msg}")
            return AlertResult(success=False, adapter=self.name, error=error_msg)
