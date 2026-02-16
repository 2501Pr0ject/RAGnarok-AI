"""Alert adapters for various notification channels.

This module provides alert adapters for sending notifications to:
- Webhooks (generic HTTP POST)
- Slack (via incoming webhooks)

Example:
    >>> from ragnarok_ai.alerts.adapters import SlackAlertAdapter, WebhookAlertAdapter
    >>>
    >>> slack = SlackAlertAdapter(webhook_url="https://hooks.slack.com/...")
    >>> webhook = WebhookAlertAdapter(url="https://my-api.com/alerts")
"""

from ragnarok_ai.alerts.adapters.slack import SlackAlertAdapter
from ragnarok_ai.alerts.adapters.webhook import WebhookAlertAdapter

__all__ = [
    "SlackAlertAdapter",
    "WebhookAlertAdapter",
]
