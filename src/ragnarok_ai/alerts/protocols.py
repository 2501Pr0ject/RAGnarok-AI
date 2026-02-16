"""Alert protocols and core data types.

This module defines the core types for the alerting system:
- AlertSeverity: Enum for alert severity levels
- Alert: Dataclass representing an alert
- AlertResult: Result of sending an alert
- AlertAdapter: Protocol for alert adapters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol


class AlertSeverity(str, Enum):
    """Alert severity level.

    Attributes:
        INFO: Informational alert (no action needed).
        WARNING: Warning alert (attention needed).
        CRITICAL: Critical alert (immediate action needed).
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    """An alert to be sent.

    Attributes:
        title: Short title for the alert.
        message: Detailed alert message.
        severity: Alert severity level.
        source: Source of the alert (e.g., "threshold", "drift").
        metadata: Additional context data.
        timestamp: When the alert was created.

    Example:
        >>> alert = Alert(
        ...     title="Low Precision",
        ...     message="Precision dropped to 0.65",
        ...     severity=AlertSeverity.WARNING,
        ...     source="threshold",
        ... )
    """

    title: str
    message: str
    severity: AlertSeverity
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary.

        Returns:
            Dictionary representation of the alert.
        """
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AlertResult:
    """Result of sending an alert.

    Attributes:
        success: Whether the alert was sent successfully.
        adapter: Name of the adapter that sent the alert.
        error: Error message if sending failed.
    """

    success: bool
    adapter: str
    error: str | None = None


class AlertAdapter(Protocol):
    """Protocol for alert adapters.

    Alert adapters are responsible for sending alerts to external
    services (webhooks, Slack, etc.).

    Example:
        >>> class MyAdapter:
        ...     @property
        ...     def name(self) -> str:
        ...         return "my-adapter"
        ...
        ...     async def send(self, alert: Alert) -> AlertResult:
        ...         # Send alert to service
        ...         return AlertResult(success=True, adapter=self.name)
    """

    @property
    def name(self) -> str:
        """Return the adapter name."""
        ...  # pragma: no cover

    async def send(self, alert: Alert) -> AlertResult:
        """Send an alert.

        Args:
            alert: The alert to send.

        Returns:
            Result indicating success or failure.
        """
        ...  # pragma: no cover
