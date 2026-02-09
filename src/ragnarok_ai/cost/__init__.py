"""Cost tracking module for RAGnarok-AI.

This module provides cost tracking and estimation for LLM API usage,
helping users understand the cost of their evaluations.

Example:
    >>> from ragnarok_ai.cost import CostTracker, CostSummary
    >>>
    >>> tracker = CostTracker()
    >>> tracker.track("openai", "gpt-4o", input_tokens=1000, output_tokens=500)
    >>> tracker.track("ollama", "llama2", input_tokens=2000, output_tokens=1000)
    >>>
    >>> print(tracker.summary())
    +--------------------+------------+----------+
    | Provider           |     Tokens |     Cost |
    +--------------------+------------+----------+
    | openai             |      1,500 |    $0.01 |
    | ollama (local)     |      3,000 |    $0.00 |
    +--------------------+------------+----------+
    | Total              |      4,500 |    $0.01 |
    +--------------------+------------+----------+
"""

from ragnarok_ai.cost.pricing import (
    LOCAL_PROVIDERS,
    PRICING,
    ModelPricing,
    calculate_cost,
    get_pricing,
)
from ragnarok_ai.cost.tokenizer import (
    count_tokens,
    estimate_tokens,
    is_tiktoken_available,
)
from ragnarok_ai.cost.tracker import (
    CostSummary,
    CostTracker,
    ProviderUsage,
    cost_tracking,
    get_active_tracker,
    get_global_tracker,
    reset_global_tracker,
    track_usage,
)

__all__ = [
    "LOCAL_PROVIDERS",
    "PRICING",
    "CostSummary",
    "CostTracker",
    "ModelPricing",
    "ProviderUsage",
    "calculate_cost",
    "cost_tracking",
    "count_tokens",
    "estimate_tokens",
    "get_active_tracker",
    "get_global_tracker",
    "get_pricing",
    "is_tiktoken_available",
    "reset_global_tracker",
    "track_usage",
]
