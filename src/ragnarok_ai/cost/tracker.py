"""Cost tracking for LLM API usage.

This module provides the CostTracker class for tracking token usage
and calculating costs across multiple LLM calls.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ragnarok_ai.cost.pricing import LOCAL_PROVIDERS, ModelPricing, calculate_cost

if TYPE_CHECKING:
    from collections.abc import Iterator

# Context variable for the active cost tracker
_active_tracker: ContextVar[CostTracker | None] = ContextVar("active_cost_tracker", default=None)


@dataclass
class ProviderUsage:
    """Token usage for a single provider/model."""

    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    call_count: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def is_local(self) -> bool:
        """Check if this is a local (free) provider."""
        return self.provider.lower() in LOCAL_PROVIDERS


@dataclass
class CostSummary:
    """Summary of costs across all providers.

    Attributes:
        total_input_tokens: Total input tokens across all providers.
        total_output_tokens: Total output tokens across all providers.
        total_cost: Total cost in USD.
        by_provider: Breakdown by provider/model.
    """

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    by_provider: dict[str, ProviderUsage] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the cost summary.
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "total_cost_formatted": f"${self.total_cost:.2f}",
            "by_provider": {
                key: {
                    "provider": usage.provider,
                    "model": usage.model,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "cost": round(usage.cost, 6),
                    "cost_formatted": f"${usage.cost:.2f}",
                    "call_count": usage.call_count,
                    "is_local": usage.is_local,
                }
                for key, usage in self.by_provider.items()
            },
        }

    def summary(self) -> str:
        """Generate a formatted summary table.

        Returns:
            Formatted string with cost breakdown.
        """
        if not self.by_provider:
            return "No usage tracked."

        lines = []
        lines.append("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 10 + "+")
        lines.append(f"| {'Provider':<18} | {'Tokens':>10} | {'Cost':>8} |")
        lines.append("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 10 + "+")

        for usage in sorted(self.by_provider.values(), key=lambda x: x.cost, reverse=True):
            provider_display = usage.provider
            if usage.is_local:
                provider_display += " (local)"
            # Truncate if too long
            if len(provider_display) > 18:
                provider_display = provider_display[:15] + "..."

            tokens_str = f"{usage.total_tokens:,}"
            cost_str = f"${usage.cost:.2f}"

            lines.append(f"| {provider_display:<18} | {tokens_str:>10} | {cost_str:>8} |")

        lines.append("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 10 + "+")

        # Total row
        total_tokens_str = f"{self.total_tokens:,}"
        total_cost_str = f"${self.total_cost:.2f}"
        lines.append(f"| {'Total':<18} | {total_tokens_str:>10} | {total_cost_str:>8} |")
        lines.append("+" + "-" * 20 + "+" + "-" * 12 + "+" + "-" * 10 + "+")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation."""
        return self.summary()


class CostTracker:
    """Track token usage and costs across LLM calls.

    Example:
        >>> tracker = CostTracker()
        >>> tracker.track("openai", "gpt-4o", input_tokens=100, output_tokens=50)
        >>> tracker.track("ollama", "llama2", input_tokens=200, output_tokens=100)
        >>> print(tracker.summary())
        +--------------------+------------+----------+
        | Provider           |     Tokens |     Cost |
        +--------------------+------------+----------+
        | openai             |        150 |    $0.00 |
        | ollama (local)     |        300 |    $0.00 |
        +--------------------+------------+----------+
        | Total              |        450 |    $0.00 |
        +--------------------+------------+----------+
    """

    def __init__(
        self,
        custom_pricing: dict[str, ModelPricing] | None = None,
    ) -> None:
        """Initialize CostTracker.

        Args:
            custom_pricing: Optional custom pricing override.
                Keys are model names, values are {"input": float, "output": float}.
        """
        self._custom_pricing = custom_pricing or {}
        self._usage: dict[str, ProviderUsage] = {}

    def track(
        self,
        provider: str,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> float:
        """Track token usage for a single API call.

        Args:
            provider: Provider name (e.g., "openai", "anthropic", "ollama").
            model: Model name (e.g., "gpt-4o", "claude-3-sonnet").
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost for this call in USD.
        """
        # Calculate cost for this call
        cost = calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider=provider,
            custom_pricing=self._custom_pricing,
        )

        # Update or create usage entry
        key = f"{provider}:{model}"
        if key not in self._usage:
            self._usage[key] = ProviderUsage(provider=provider, model=model)

        usage = self._usage[key]
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.cost += cost
        usage.call_count += 1

        return cost

    def get_summary(self) -> CostSummary:
        """Get a summary of all tracked usage.

        Returns:
            CostSummary with total and per-provider breakdown.
        """
        total_input = sum(u.input_tokens for u in self._usage.values())
        total_output = sum(u.output_tokens for u in self._usage.values())
        total_cost = sum(u.cost for u in self._usage.values())

        return CostSummary(
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cost=total_cost,
            by_provider=dict(self._usage),
        )

    def summary(self) -> str:
        """Get formatted summary string.

        Returns:
            Formatted table string.
        """
        return self.get_summary().summary()

    def to_dict(self) -> dict[str, object]:
        """Get summary as dictionary.

        Returns:
            Dictionary representation.
        """
        return self.get_summary().to_dict()

    def reset(self) -> None:
        """Reset all tracked usage."""
        self._usage.clear()

    @property
    def total_cost(self) -> float:
        """Get total cost across all providers."""
        return sum(u.cost for u in self._usage.values())

    @property
    def total_tokens(self) -> int:
        """Get total tokens across all providers."""
        return sum(u.total_tokens for u in self._usage.values())


# Global tracker instance for convenience
_global_tracker: CostTracker | None = None


def get_global_tracker() -> CostTracker:
    """Get the global cost tracker instance.

    Creates one if it doesn't exist.

    Returns:
        Global CostTracker instance.
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_global_tracker() -> None:
    """Reset the global cost tracker."""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.reset()


def get_active_tracker() -> CostTracker | None:
    """Get the currently active cost tracker (from context).

    Returns:
        The active CostTracker, or None if no tracker is active.
    """
    return _active_tracker.get()


def track_usage(
    provider: str,
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    """Track token usage if a tracker is active.

    This is the main function LLM adapters should call to report usage.
    If no tracker is active, this is a no-op.

    Args:
        provider: Provider name (e.g., "openai", "anthropic", "ollama").
        model: Model name (e.g., "gpt-4o", "claude-3-sonnet").
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Cost for this call in USD (0.0 if no tracker active).
    """
    tracker = _active_tracker.get()
    if tracker is None:
        return 0.0
    return tracker.track(provider, model, input_tokens=input_tokens, output_tokens=output_tokens)


@contextmanager
def cost_tracking(
    custom_pricing: dict[str, ModelPricing] | None = None,
) -> Iterator[CostTracker]:
    """Context manager to enable cost tracking.

    All LLM calls within this context will be tracked automatically
    if the adapters call track_usage().

    Args:
        custom_pricing: Optional custom pricing override.

    Yields:
        The CostTracker instance for this context.

    Example:
        >>> from ragnarok_ai.cost import cost_tracking
        >>>
        >>> with cost_tracking() as tracker:
        ...     # LLM calls here will be tracked
        ...     response = await llm.generate("Hello")
        ...
        >>> print(tracker.summary())
    """
    tracker = CostTracker(custom_pricing=custom_pricing)
    token = _active_tracker.set(tracker)
    try:
        yield tracker
    finally:
        _active_tracker.reset(token)
