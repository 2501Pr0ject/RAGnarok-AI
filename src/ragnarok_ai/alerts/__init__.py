"""Alerting system for RAGnarok-AI.

This module provides a multi-channel alerting system for monitoring
RAG pipeline metrics and triggering alerts based on configurable rules.

Example:
    >>> from ragnarok_ai.alerts import AlertManager, AlertRule, AlertSeverity
    >>> from ragnarok_ai.alerts.adapters import SlackAlertAdapter, WebhookAlertAdapter
    >>>
    >>> # Create manager and add adapters
    >>> manager = AlertManager()
    >>> manager.add_adapter(SlackAlertAdapter(webhook_url="https://..."))
    >>> manager.add_adapter(WebhookAlertAdapter(url="https://..."))
    >>>
    >>> # Add threshold rules
    >>> manager.add_rule(AlertRule(
    ...     name="Low Precision",
    ...     metric="precision",
    ...     condition="lt",
    ...     threshold=0.7,
    ...     severity=AlertSeverity.WARNING,
    ... ))
    >>>
    >>> # Check metrics and send alerts
    >>> results = await manager.check_and_alert({"precision": 0.65})
"""

from ragnarok_ai.alerts.manager import AlertManager
from ragnarok_ai.alerts.protocols import Alert, AlertAdapter, AlertResult, AlertSeverity
from ragnarok_ai.alerts.rules import AlertRule

__all__ = [
    "Alert",
    "AlertAdapter",
    "AlertManager",
    "AlertResult",
    "AlertRule",
    "AlertSeverity",
]
