"""Alert rules for threshold-based alerting.

This module provides AlertRule for defining when alerts should be triggered
based on metric thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

from ragnarok_ai.alerts.protocols import Alert, AlertSeverity

Condition = Literal["gt", "lt", "gte", "lte", "eq", "neq"]


@dataclass
class AlertRule:
    """A rule that triggers alerts based on metric thresholds.

    Attributes:
        name: Human-readable name for the rule.
        metric: Name of the metric to monitor.
        condition: Comparison operator.
        threshold: Threshold value for the condition.
        severity: Severity of alerts triggered by this rule.
        cooldown: Minimum time between alerts for this rule.
        message_template: Optional custom message template.

    Example:
        >>> rule = AlertRule(
        ...     name="Low Precision",
        ...     metric="precision",
        ...     condition="lt",
        ...     threshold=0.7,
        ...     severity=AlertSeverity.WARNING,
        ... )
        >>> rule.evaluate(0.65)
        True
        >>> rule.evaluate(0.85)
        False
    """

    name: str
    metric: str
    condition: Condition
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    message_template: str | None = None

    _last_triggered: datetime | None = field(default=None, repr=False, compare=False)

    def evaluate(self, value: float) -> bool:
        """Evaluate if the rule should trigger.

        Args:
            value: Current metric value.

        Returns:
            True if the condition is met, False otherwise.
        """
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "gte":
            return value >= self.threshold
        elif self.condition == "lte":
            return value <= self.threshold
        elif self.condition == "eq":
            return value == self.threshold
        elif self.condition == "neq":
            return value != self.threshold
        return False  # pragma: no cover

    def is_in_cooldown(self) -> bool:
        """Check if the rule is in cooldown period.

        Returns:
            True if rule was recently triggered and is in cooldown.
        """
        if self._last_triggered is None:
            return False
        elapsed = datetime.now(timezone.utc) - self._last_triggered
        return elapsed < self.cooldown

    def create_alert(self, value: float) -> Alert:
        """Create an alert for this rule.

        Args:
            value: The metric value that triggered the rule.

        Returns:
            An Alert object for this rule.
        """
        if self.message_template:
            message = self.message_template.format(
                metric=self.metric,
                value=value,
                threshold=self.threshold,
                condition=self.condition,
            )
        else:
            message = self._default_message(value)

        return Alert(
            title=self.name,
            message=message,
            severity=self.severity,
            source="threshold",
            metadata={
                "rule_name": self.name,
                "metric": self.metric,
                "value": value,
                "threshold": self.threshold,
                "condition": self.condition,
            },
        )

    def _default_message(self, value: float) -> str:
        """Generate default alert message.

        Args:
            value: The metric value that triggered the rule.

        Returns:
            Default message string.
        """
        condition_text = {
            "gt": "exceeded",
            "lt": "dropped below",
            "gte": "reached or exceeded",
            "lte": "reached or dropped below",
            "eq": "equals",
            "neq": "changed from",
        }
        verb = condition_text.get(self.condition, "triggered at")
        return f"{self.metric} {verb} {self.threshold} (current: {value:.4f})"

    def mark_triggered(self) -> None:
        """Mark the rule as triggered (updates cooldown timer)."""
        object.__setattr__(self, "_last_triggered", datetime.now(timezone.utc))

    def check(self, value: float) -> Alert | None:
        """Check if rule triggers and return alert if so.

        This is a convenience method that combines evaluate(),
        is_in_cooldown(), create_alert(), and mark_triggered().

        Args:
            value: Current metric value.

        Returns:
            Alert if rule triggers, None otherwise.
        """
        if self.is_in_cooldown():
            return None

        if not self.evaluate(value):
            return None

        self.mark_triggered()
        return self.create_alert(value)
