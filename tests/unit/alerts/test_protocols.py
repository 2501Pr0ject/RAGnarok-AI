"""Tests for alert protocols and core types."""

from datetime import datetime, timezone

import pytest

from ragnarok_ai.alerts.protocols import Alert, AlertResult, AlertSeverity


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_values(self) -> None:
        """Test severity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_severity_is_string(self) -> None:
        """Test that severity values are strings."""
        assert isinstance(AlertSeverity.INFO, str)
        assert AlertSeverity.WARNING == "warning"


class TestAlert:
    """Tests for Alert dataclass."""

    def test_create_alert_minimal(self) -> None:
        """Test creating alert with minimal args."""
        alert = Alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.INFO,
            source="test",
        )
        assert alert.title == "Test Alert"
        assert alert.message == "Test message"
        assert alert.severity == AlertSeverity.INFO
        assert alert.source == "test"
        assert alert.metadata == {}
        assert isinstance(alert.timestamp, datetime)

    def test_create_alert_with_metadata(self) -> None:
        """Test creating alert with metadata."""
        alert = Alert(
            title="Test",
            message="Message",
            severity=AlertSeverity.WARNING,
            source="threshold",
            metadata={"metric": "precision", "value": 0.65},
        )
        assert alert.metadata == {"metric": "precision", "value": 0.65}

    def test_create_alert_with_timestamp(self) -> None:
        """Test creating alert with custom timestamp."""
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        alert = Alert(
            title="Test",
            message="Message",
            severity=AlertSeverity.CRITICAL,
            source="test",
            timestamp=ts,
        )
        assert alert.timestamp == ts

    def test_alert_is_frozen(self) -> None:
        """Test that alert is immutable."""
        alert = Alert(
            title="Test",
            message="Message",
            severity=AlertSeverity.INFO,
            source="test",
        )
        with pytest.raises(AttributeError):
            alert.title = "New Title"  # type: ignore[misc]

    def test_alert_to_dict(self) -> None:
        """Test converting alert to dictionary."""
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        alert = Alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.WARNING,
            source="threshold",
            metadata={"key": "value"},
            timestamp=ts,
        )
        d = alert.to_dict()

        assert d["title"] == "Test Alert"
        assert d["message"] == "Test message"
        assert d["severity"] == "warning"
        assert d["source"] == "threshold"
        assert d["metadata"] == {"key": "value"}
        assert d["timestamp"] == "2024-01-15T10:30:00+00:00"


class TestAlertResult:
    """Tests for AlertResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating successful result."""
        result = AlertResult(success=True, adapter="webhook")
        assert result.success is True
        assert result.adapter == "webhook"
        assert result.error is None

    def test_create_failure_result(self) -> None:
        """Test creating failure result."""
        result = AlertResult(
            success=False,
            adapter="slack",
            error="Connection timeout",
        )
        assert result.success is False
        assert result.adapter == "slack"
        assert result.error == "Connection timeout"
