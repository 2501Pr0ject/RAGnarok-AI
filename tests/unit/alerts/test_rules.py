"""Tests for alert rules."""

from datetime import timedelta

import pytest

from ragnarok_ai.alerts.protocols import AlertSeverity
from ragnarok_ai.alerts.rules import AlertRule


class TestAlertRule:
    """Tests for AlertRule class."""

    def test_create_rule_minimal(self) -> None:
        """Test creating rule with minimal args."""
        rule = AlertRule(
            name="Test Rule",
            metric="precision",
            condition="lt",
            threshold=0.7,
        )
        assert rule.name == "Test Rule"
        assert rule.metric == "precision"
        assert rule.condition == "lt"
        assert rule.threshold == 0.7
        assert rule.severity == AlertSeverity.WARNING
        assert rule.cooldown == timedelta(minutes=5)

    def test_create_rule_full(self) -> None:
        """Test creating rule with all args."""
        rule = AlertRule(
            name="Critical Rule",
            metric="latency_p99",
            condition="gt",
            threshold=1000.0,
            severity=AlertSeverity.CRITICAL,
            cooldown=timedelta(minutes=10),
            message_template="Latency {metric} is {value}ms (threshold: {threshold}ms)",
        )
        assert rule.severity == AlertSeverity.CRITICAL
        assert rule.cooldown == timedelta(minutes=10)
        assert rule.message_template is not None


class TestAlertRuleEvaluate:
    """Tests for AlertRule.evaluate()."""

    @pytest.mark.parametrize(
        ("condition", "threshold", "value", "expected"),
        [
            ("gt", 0.7, 0.8, True),
            ("gt", 0.7, 0.7, False),
            ("gt", 0.7, 0.6, False),
            ("lt", 0.7, 0.6, True),
            ("lt", 0.7, 0.7, False),
            ("lt", 0.7, 0.8, False),
            ("gte", 0.7, 0.8, True),
            ("gte", 0.7, 0.7, True),
            ("gte", 0.7, 0.6, False),
            ("lte", 0.7, 0.6, True),
            ("lte", 0.7, 0.7, True),
            ("lte", 0.7, 0.8, False),
            ("eq", 0.7, 0.7, True),
            ("eq", 0.7, 0.8, False),
            ("neq", 0.7, 0.8, True),
            ("neq", 0.7, 0.7, False),
        ],
    )
    def test_evaluate_conditions(
        self,
        condition: str,
        threshold: float,
        value: float,
        expected: bool,
    ) -> None:
        """Test all condition operators."""
        rule = AlertRule(
            name="Test",
            metric="test",
            condition=condition,  # type: ignore[arg-type]
            threshold=threshold,
        )
        assert rule.evaluate(value) == expected


class TestAlertRuleCooldown:
    """Tests for AlertRule cooldown functionality."""

    def test_not_in_cooldown_initially(self) -> None:
        """Test that rule is not in cooldown initially."""
        rule = AlertRule(
            name="Test",
            metric="test",
            condition="lt",
            threshold=0.7,
        )
        assert rule.is_in_cooldown() is False

    def test_in_cooldown_after_trigger(self) -> None:
        """Test that rule is in cooldown after being triggered."""
        rule = AlertRule(
            name="Test",
            metric="test",
            condition="lt",
            threshold=0.7,
            cooldown=timedelta(minutes=5),
        )
        rule.mark_triggered()
        assert rule.is_in_cooldown() is True


class TestAlertRuleCreateAlert:
    """Tests for AlertRule.create_alert()."""

    def test_create_alert_default_message(self) -> None:
        """Test creating alert with default message."""
        rule = AlertRule(
            name="Low Precision",
            metric="precision",
            condition="lt",
            threshold=0.7,
            severity=AlertSeverity.WARNING,
        )
        alert = rule.create_alert(0.65)

        assert alert.title == "Low Precision"
        assert "precision" in alert.message
        assert "0.7" in alert.message
        assert "0.65" in alert.message
        assert alert.severity == AlertSeverity.WARNING
        assert alert.source == "threshold"
        assert alert.metadata["rule_name"] == "Low Precision"
        assert alert.metadata["metric"] == "precision"
        assert alert.metadata["value"] == 0.65
        assert alert.metadata["threshold"] == 0.7

    def test_create_alert_custom_message(self) -> None:
        """Test creating alert with custom message template."""
        rule = AlertRule(
            name="Custom Rule",
            metric="latency",
            condition="gt",
            threshold=100,
            message_template="{metric} is {value}ms, threshold is {threshold}ms",
        )
        alert = rule.create_alert(150.0)

        assert alert.message == "latency is 150.0ms, threshold is 100ms"


class TestAlertRuleCheck:
    """Tests for AlertRule.check() convenience method."""

    def test_check_returns_alert_when_triggered(self) -> None:
        """Test check returns alert when condition is met."""
        rule = AlertRule(
            name="Test",
            metric="test",
            condition="lt",
            threshold=0.7,
        )
        alert = rule.check(0.65)

        assert alert is not None
        assert alert.title == "Test"

    def test_check_returns_none_when_not_triggered(self) -> None:
        """Test check returns None when condition is not met."""
        rule = AlertRule(
            name="Test",
            metric="test",
            condition="lt",
            threshold=0.7,
        )
        alert = rule.check(0.85)

        assert alert is None

    def test_check_returns_none_during_cooldown(self) -> None:
        """Test check returns None during cooldown."""
        rule = AlertRule(
            name="Test",
            metric="test",
            condition="lt",
            threshold=0.7,
        )
        # First check triggers
        alert1 = rule.check(0.65)
        assert alert1 is not None

        # Second check during cooldown returns None
        alert2 = rule.check(0.65)
        assert alert2 is None

    def test_check_marks_triggered(self) -> None:
        """Test that check marks rule as triggered."""
        rule = AlertRule(
            name="Test",
            metric="test",
            condition="lt",
            threshold=0.7,
        )
        assert rule.is_in_cooldown() is False

        rule.check(0.65)

        assert rule.is_in_cooldown() is True
