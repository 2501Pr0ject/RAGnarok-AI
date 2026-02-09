"""Tests for cost tracker module."""

import pytest

from ragnarok_ai.cost.tracker import (
    CostSummary,
    CostTracker,
    ProviderUsage,
    cost_tracking,
    get_active_tracker,
    track_usage,
)


class TestProviderUsage:
    """Tests for ProviderUsage dataclass."""

    def test_total_tokens(self):
        """Test total_tokens property."""
        usage = ProviderUsage(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        assert usage.total_tokens == 150

    def test_is_local_ollama(self):
        """Test is_local for Ollama."""
        usage = ProviderUsage(provider="ollama", model="llama2")
        assert usage.is_local is True

    def test_is_local_openai(self):
        """Test is_local for OpenAI."""
        usage = ProviderUsage(provider="openai", model="gpt-4o")
        assert usage.is_local is False


class TestCostSummary:
    """Tests for CostSummary dataclass."""

    def test_total_tokens(self):
        """Test total_tokens property."""
        summary = CostSummary(total_input_tokens=100, total_output_tokens=50)
        assert summary.total_tokens == 150

    def test_to_dict(self):
        """Test to_dict method."""
        summary = CostSummary(
            total_input_tokens=1000,
            total_output_tokens=500,
            total_cost=0.01,
        )
        result = summary.to_dict()
        assert result["total_input_tokens"] == 1000
        assert result["total_output_tokens"] == 500
        assert result["total_tokens"] == 1500
        assert result["total_cost"] == 0.01
        assert result["total_cost_formatted"] == "$0.01"

    def test_summary_empty(self):
        """Test summary with no providers."""
        summary = CostSummary()
        assert "No usage tracked" in summary.summary()

    def test_summary_with_providers(self):
        """Test summary with providers."""
        usage = ProviderUsage(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cost=0.01,
        )
        summary = CostSummary(
            total_input_tokens=1000,
            total_output_tokens=500,
            total_cost=0.01,
            by_provider={"openai:gpt-4o": usage},
        )
        text = summary.summary()
        assert "openai" in text
        assert "$0.01" in text


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_track_single_call(self):
        """Test tracking a single call."""
        tracker = CostTracker()
        cost = tracker.track(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost > 0
        assert tracker.total_cost > 0
        assert tracker.total_tokens == 1500

    def test_track_multiple_calls(self):
        """Test tracking multiple calls."""
        tracker = CostTracker()
        tracker.track("openai", "gpt-4o", input_tokens=1000, output_tokens=500)
        tracker.track("openai", "gpt-4o", input_tokens=2000, output_tokens=1000)

        summary = tracker.get_summary()
        assert summary.total_input_tokens == 3000
        assert summary.total_output_tokens == 1500

    def test_track_local_is_free(self):
        """Test that local providers are free."""
        tracker = CostTracker()
        cost = tracker.track(
            provider="ollama",
            model="llama2",
            input_tokens=10000,
            output_tokens=5000,
        )
        assert cost == 0.0
        assert tracker.total_cost == 0.0

    def test_track_custom_pricing(self):
        """Test tracking with custom pricing."""
        custom = {"my-model": {"input": 1.0, "output": 2.0}}
        tracker = CostTracker(custom_pricing=custom)
        cost = tracker.track(
            provider="custom",
            model="my-model",
            input_tokens=1_000_000,
            output_tokens=500_000,
        )
        expected = 1.0 + 1.0  # 1M * $1 + 0.5M * $2
        assert cost == pytest.approx(expected)

    def test_reset(self):
        """Test resetting tracker."""
        tracker = CostTracker()
        tracker.track("openai", "gpt-4o", input_tokens=1000, output_tokens=500)
        assert tracker.total_tokens > 0

        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0

    def test_get_summary(self):
        """Test getting summary."""
        tracker = CostTracker()
        tracker.track("openai", "gpt-4o", input_tokens=1000, output_tokens=500)
        tracker.track("ollama", "llama2", input_tokens=2000, output_tokens=1000)

        summary = tracker.get_summary()
        assert len(summary.by_provider) == 2
        assert "openai:gpt-4o" in summary.by_provider
        assert "ollama:llama2" in summary.by_provider

    def test_to_dict(self):
        """Test to_dict method."""
        tracker = CostTracker()
        tracker.track("openai", "gpt-4o", input_tokens=1000, output_tokens=500)
        result = tracker.to_dict()
        assert "total_input_tokens" in result
        assert "by_provider" in result


class TestCostTrackingContext:
    """Tests for cost_tracking context manager."""

    def test_cost_tracking_context(self):
        """Test cost_tracking context manager."""
        with cost_tracking() as tracker:
            # Track some usage directly
            tracker.track("openai", "gpt-4o", input_tokens=1000, output_tokens=500)

        assert tracker.total_tokens == 1500

    def test_get_active_tracker_inside_context(self):
        """Test get_active_tracker inside context."""
        assert get_active_tracker() is None

        with cost_tracking() as tracker:
            active = get_active_tracker()
            assert active is tracker

        assert get_active_tracker() is None

    def test_track_usage_inside_context(self):
        """Test track_usage function inside context."""
        with cost_tracking() as tracker:
            track_usage("openai", "gpt-4o", input_tokens=1000, output_tokens=500)

        assert tracker.total_tokens == 1500

    def test_track_usage_outside_context(self):
        """Test track_usage returns 0 outside context."""
        cost = track_usage("openai", "gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_nested_contexts(self):
        """Test nested cost_tracking contexts."""
        with cost_tracking() as outer:
            outer.track("openai", "gpt-4o", input_tokens=1000, output_tokens=500)

            with cost_tracking() as inner:
                inner.track("anthropic", "claude-3-sonnet", input_tokens=2000, output_tokens=1000)
                assert inner.total_tokens == 3000

            # Inner context should not affect outer
            assert get_active_tracker() is outer

        assert outer.total_tokens == 1500
