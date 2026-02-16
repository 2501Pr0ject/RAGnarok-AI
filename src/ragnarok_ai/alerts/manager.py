"""Alert manager for multi-channel dispatch.

This module provides AlertManager which coordinates sending alerts
to multiple adapters and evaluating rules against metrics.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from ragnarok_ai.alerts.protocols import Alert, AlertResult

if TYPE_CHECKING:
    from ragnarok_ai.alerts.protocols import AlertAdapter
    from ragnarok_ai.alerts.rules import AlertRule

logger = logging.getLogger(__name__)


class AlertManager:
    """Manager for multi-channel alert dispatch.

    Coordinates sending alerts to multiple adapters (webhook, Slack, etc.)
    and evaluates alert rules against metrics.

    Example:
        >>> manager = AlertManager()
        >>> manager.add_adapter(WebhookAlertAdapter(url="https://..."))
        >>> manager.add_adapter(SlackAlertAdapter(webhook_url="https://..."))
        >>> manager.add_rule(AlertRule(
        ...     name="Low Precision",
        ...     metric="precision",
        ...     condition="lt",
        ...     threshold=0.7,
        ... ))
        >>> # Send alert to all adapters
        >>> results = await manager.send(alert)
        >>> # Or check rules and alert automatically
        >>> results = await manager.check_and_alert({"precision": 0.65})
    """

    def __init__(self) -> None:
        """Initialize the AlertManager."""
        self._adapters: list[AlertAdapter] = []
        self._rules: list[AlertRule] = []

    def add_adapter(self, adapter: AlertAdapter) -> None:
        """Add an alert adapter.

        Args:
            adapter: The adapter to add.
        """
        self._adapters.append(adapter)
        logger.debug(f"Added alert adapter: {adapter.name}")

    def remove_adapter(self, name: str) -> bool:
        """Remove an adapter by name.

        Args:
            name: Name of the adapter to remove.

        Returns:
            True if adapter was removed, False if not found.
        """
        for i, adapter in enumerate(self._adapters):
            if adapter.name == name:
                self._adapters.pop(i)
                logger.debug(f"Removed alert adapter: {name}")
                return True
        return False

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: The rule to add.
        """
        self._rules.append(rule)
        logger.debug(f"Added alert rule: {rule.name}")

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name.

        Args:
            name: Name of the rule to remove.

        Returns:
            True if rule was removed, False if not found.
        """
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                self._rules.pop(i)
                logger.debug(f"Removed alert rule: {name}")
                return True
        return False

    def list_adapters(self) -> list[str]:
        """List registered adapter names.

        Returns:
            List of adapter names.
        """
        return [adapter.name for adapter in self._adapters]

    def list_rules(self) -> list[str]:
        """List registered rule names.

        Returns:
            List of rule names.
        """
        return [rule.name for rule in self._rules]

    async def send(self, alert: Alert) -> list[AlertResult]:
        """Send an alert to all registered adapters.

        Args:
            alert: The alert to send.

        Returns:
            List of results from each adapter.
        """
        if not self._adapters:
            logger.warning("No alert adapters configured")
            return []

        # Send to all adapters concurrently
        tasks = [adapter.send(alert) for adapter in self._adapters]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed: list[AlertResult] = []
        for i, result in enumerate(results):
            adapter_name = self._adapters[i].name
            if isinstance(result, BaseException):
                logger.error(f"Alert adapter {adapter_name} failed: {result}")
                processed.append(
                    AlertResult(
                        success=False,
                        adapter=adapter_name,
                        error=str(result),
                    )
                )
            else:
                alert_result: AlertResult = result
                processed.append(alert_result)
                if alert_result.success:
                    logger.info(f"Alert sent via {adapter_name}: {alert.title}")
                else:
                    logger.warning(f"Alert failed via {adapter_name}: {alert_result.error}")

        return processed

    async def check_and_alert(
        self,
        metrics: dict[str, float],
    ) -> list[AlertResult]:
        """Check all rules against metrics and send alerts if triggered.

        Args:
            metrics: Dictionary of metric names to values.

        Returns:
            List of results for all alerts sent.
        """
        all_results: list[AlertResult] = []

        for rule in self._rules:
            if rule.metric not in metrics:
                continue

            value = metrics[rule.metric]
            alert = rule.check(value)

            if alert is not None:
                results = await self.send(alert)
                all_results.extend(results)

        return all_results

    @property
    def adapter_count(self) -> int:
        """Return number of registered adapters."""
        return len(self._adapters)

    @property
    def rule_count(self) -> int:
        """Return number of registered rules."""
        return len(self._rules)
