"""Tests for AlertManager."""

from datetime import timedelta

import pytest

from ragnarok_ai.alerts.manager import AlertManager
from ragnarok_ai.alerts.protocols import Alert, AlertResult, AlertSeverity
from ragnarok_ai.alerts.rules import AlertRule


class MockAdapter:
    """Mock adapter for testing."""

    def __init__(self, name: str = "mock", should_fail: bool = False) -> None:
        self._name = name
        self._should_fail = should_fail
        self.sent_alerts: list[Alert] = []

    @property
    def name(self) -> str:
        return self._name

    async def send(self, alert: Alert) -> AlertResult:
        if self._should_fail:
            return AlertResult(success=False, adapter=self._name, error="Mock failure")
        self.sent_alerts.append(alert)
        return AlertResult(success=True, adapter=self._name)


class ExceptionAdapter:
    """Mock adapter that raises an exception."""

    def __init__(self, name: str = "exception") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def send(self, alert: Alert) -> AlertResult:  # noqa: ARG002
        raise RuntimeError("Adapter crashed")


class TestAlertManagerAdapters:
    """Tests for AlertManager adapter management."""

    def test_add_adapter(self) -> None:
        """Test adding an adapter."""
        manager = AlertManager()
        adapter = MockAdapter("test")

        manager.add_adapter(adapter)

        assert manager.adapter_count == 1
        assert "test" in manager.list_adapters()

    def test_add_multiple_adapters(self) -> None:
        """Test adding multiple adapters."""
        manager = AlertManager()
        manager.add_adapter(MockAdapter("adapter1"))
        manager.add_adapter(MockAdapter("adapter2"))

        assert manager.adapter_count == 2
        assert manager.list_adapters() == ["adapter1", "adapter2"]

    def test_remove_adapter(self) -> None:
        """Test removing an adapter."""
        manager = AlertManager()
        manager.add_adapter(MockAdapter("test"))

        removed = manager.remove_adapter("test")

        assert removed is True
        assert manager.adapter_count == 0

    def test_remove_nonexistent_adapter(self) -> None:
        """Test removing adapter that doesn't exist."""
        manager = AlertManager()

        removed = manager.remove_adapter("nonexistent")

        assert removed is False


class TestAlertManagerRules:
    """Tests for AlertManager rule management."""

    def test_add_rule(self) -> None:
        """Test adding a rule."""
        manager = AlertManager()
        rule = AlertRule(
            name="Test Rule",
            metric="precision",
            condition="lt",
            threshold=0.7,
        )

        manager.add_rule(rule)

        assert manager.rule_count == 1
        assert "Test Rule" in manager.list_rules()

    def test_remove_rule(self) -> None:
        """Test removing a rule."""
        manager = AlertManager()
        manager.add_rule(
            AlertRule(
                name="Test Rule",
                metric="precision",
                condition="lt",
                threshold=0.7,
            )
        )

        removed = manager.remove_rule("Test Rule")

        assert removed is True
        assert manager.rule_count == 0

    def test_remove_nonexistent_rule(self) -> None:
        """Test removing rule that doesn't exist."""
        manager = AlertManager()

        removed = manager.remove_rule("nonexistent")

        assert removed is False


class TestAlertManagerSend:
    """Tests for AlertManager.send()."""

    @pytest.fixture
    def alert(self) -> Alert:
        """Create a test alert."""
        return Alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.WARNING,
            source="test",
        )

    @pytest.mark.asyncio
    async def test_send_no_adapters(self, alert: Alert) -> None:
        """Test sending with no adapters configured."""
        manager = AlertManager()

        results = await manager.send(alert)

        assert results == []

    @pytest.mark.asyncio
    async def test_send_single_adapter(self, alert: Alert) -> None:
        """Test sending to single adapter."""
        manager = AlertManager()
        adapter = MockAdapter("test")
        manager.add_adapter(adapter)

        results = await manager.send(alert)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].adapter == "test"
        assert len(adapter.sent_alerts) == 1

    @pytest.mark.asyncio
    async def test_send_multiple_adapters(self, alert: Alert) -> None:
        """Test sending to multiple adapters."""
        manager = AlertManager()
        adapter1 = MockAdapter("adapter1")
        adapter2 = MockAdapter("adapter2")
        manager.add_adapter(adapter1)
        manager.add_adapter(adapter2)

        results = await manager.send(alert)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert len(adapter1.sent_alerts) == 1
        assert len(adapter2.sent_alerts) == 1

    @pytest.mark.asyncio
    async def test_send_with_failing_adapter(self, alert: Alert) -> None:
        """Test sending when one adapter fails."""
        manager = AlertManager()
        adapter1 = MockAdapter("success")
        adapter2 = MockAdapter("failing", should_fail=True)
        manager.add_adapter(adapter1)
        manager.add_adapter(adapter2)

        results = await manager.send(alert)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error == "Mock failure"

    @pytest.mark.asyncio
    async def test_send_with_exception_adapter(self, alert: Alert) -> None:
        """Test sending when adapter raises an exception."""
        manager = AlertManager()
        adapter1 = MockAdapter("success")
        adapter2 = ExceptionAdapter("crashing")
        manager.add_adapter(adapter1)
        manager.add_adapter(adapter2)

        results = await manager.send(alert)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert "Adapter crashed" in results[1].error  # type: ignore[operator]


class TestAlertManagerCheckAndAlert:
    """Tests for AlertManager.check_and_alert()."""

    @pytest.mark.asyncio
    async def test_check_and_alert_no_rules(self) -> None:
        """Test check_and_alert with no rules."""
        manager = AlertManager()
        manager.add_adapter(MockAdapter("test"))

        results = await manager.check_and_alert({"precision": 0.65})

        assert results == []

    @pytest.mark.asyncio
    async def test_check_and_alert_rule_not_triggered(self) -> None:
        """Test check_and_alert when rule is not triggered."""
        manager = AlertManager()
        adapter = MockAdapter("test")
        manager.add_adapter(adapter)
        manager.add_rule(
            AlertRule(
                name="Low Precision",
                metric="precision",
                condition="lt",
                threshold=0.7,
            )
        )

        results = await manager.check_and_alert({"precision": 0.85})

        assert results == []
        assert len(adapter.sent_alerts) == 0

    @pytest.mark.asyncio
    async def test_check_and_alert_rule_triggered(self) -> None:
        """Test check_and_alert when rule is triggered."""
        manager = AlertManager()
        adapter = MockAdapter("test")
        manager.add_adapter(adapter)
        manager.add_rule(
            AlertRule(
                name="Low Precision",
                metric="precision",
                condition="lt",
                threshold=0.7,
            )
        )

        results = await manager.check_and_alert({"precision": 0.65})

        assert len(results) == 1
        assert results[0].success is True
        assert len(adapter.sent_alerts) == 1
        assert adapter.sent_alerts[0].title == "Low Precision"

    @pytest.mark.asyncio
    async def test_check_and_alert_metric_not_present(self) -> None:
        """Test check_and_alert when metric is not in data."""
        manager = AlertManager()
        adapter = MockAdapter("test")
        manager.add_adapter(adapter)
        manager.add_rule(
            AlertRule(
                name="Low Precision",
                metric="precision",
                condition="lt",
                threshold=0.7,
            )
        )

        results = await manager.check_and_alert({"recall": 0.65})

        assert results == []

    @pytest.mark.asyncio
    async def test_check_and_alert_multiple_rules(self) -> None:
        """Test check_and_alert with multiple rules triggering."""
        manager = AlertManager()
        adapter = MockAdapter("test")
        manager.add_adapter(adapter)
        manager.add_rule(
            AlertRule(
                name="Low Precision",
                metric="precision",
                condition="lt",
                threshold=0.7,
                cooldown=timedelta(seconds=0),
            )
        )
        manager.add_rule(
            AlertRule(
                name="Low Recall",
                metric="recall",
                condition="lt",
                threshold=0.6,
                cooldown=timedelta(seconds=0),
            )
        )

        results = await manager.check_and_alert({"precision": 0.65, "recall": 0.55})

        assert len(results) == 2
        assert len(adapter.sent_alerts) == 2

    @pytest.mark.asyncio
    async def test_check_and_alert_respects_cooldown(self) -> None:
        """Test that check_and_alert respects rule cooldown."""
        manager = AlertManager()
        adapter = MockAdapter("test")
        manager.add_adapter(adapter)
        manager.add_rule(
            AlertRule(
                name="Low Precision",
                metric="precision",
                condition="lt",
                threshold=0.7,
                cooldown=timedelta(minutes=5),
            )
        )

        # First check triggers
        results1 = await manager.check_and_alert({"precision": 0.65})
        assert len(results1) == 1

        # Second check during cooldown doesn't trigger
        results2 = await manager.check_and_alert({"precision": 0.65})
        assert len(results2) == 0
