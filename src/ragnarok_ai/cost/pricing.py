"""Pricing table for LLM providers.

This module contains pricing information for various LLM providers,
allowing cost estimation based on token usage.

Prices are in USD per 1 million tokens.
"""

from __future__ import annotations

from typing import TypedDict


class ModelPricing(TypedDict):
    """Pricing for a single model."""

    input: float  # USD per 1M input tokens
    output: float  # USD per 1M output tokens


# Pricing table (USD per 1M tokens)
# Last updated: 2024-01
PRICING: dict[str, ModelPricing] = {
    # =========================================================================
    # OpenAI
    # =========================================================================
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
    # =========================================================================
    # Anthropic
    # =========================================================================
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    # Aliases
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3.5-haiku": {"input": 1.00, "output": 5.00},
    # =========================================================================
    # Groq (when implemented)
    # =========================================================================
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.2-90b-vision-preview": {"input": 0.90, "output": 0.90},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
    # =========================================================================
    # Mistral (when implemented)
    # =========================================================================
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
    "mistral-medium-latest": {"input": 2.70, "output": 8.10},
    "mistral-small-latest": {"input": 0.20, "output": 0.60},
    "open-mistral-7b": {"input": 0.25, "output": 0.25},
    "open-mixtral-8x7b": {"input": 0.70, "output": 0.70},
    "open-mixtral-8x22b": {"input": 2.00, "output": 6.00},
    # =========================================================================
    # Together AI (when implemented)
    # =========================================================================
    "meta-llama/Llama-3-70b-chat-hf": {"input": 0.90, "output": 0.90},
    "meta-llama/Llama-3-8b-chat-hf": {"input": 0.20, "output": 0.20},
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {"input": 0.60, "output": 0.60},
}

# Local models are always free
LOCAL_PROVIDERS = frozenset({"ollama", "vllm", "local"})


def get_pricing(model: str, provider: str | None = None) -> ModelPricing:
    """Get pricing for a model.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-sonnet").
        provider: Optional provider name for local detection.

    Returns:
        ModelPricing with input and output costs per 1M tokens.
        Returns {"input": 0.0, "output": 0.0} for local models.
    """
    # Local providers are always free
    if provider and provider.lower() in LOCAL_PROVIDERS:
        return {"input": 0.0, "output": 0.0}

    # Check if model is in pricing table
    if model in PRICING:
        return PRICING[model]

    # Try lowercase
    model_lower = model.lower()
    if model_lower in PRICING:
        return PRICING[model_lower]

    # Unknown model - return zero (safe default)
    return {"input": 0.0, "output": 0.0}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: str | None = None,
    custom_pricing: dict[str, ModelPricing] | None = None,
) -> float:
    """Calculate cost for a single API call.

    Args:
        model: Model name.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        provider: Optional provider name for local detection.
        custom_pricing: Optional custom pricing override.

    Returns:
        Cost in USD.
    """
    # Check custom pricing first
    pricing = custom_pricing[model] if custom_pricing and model in custom_pricing else get_pricing(model, provider)

    # Calculate cost (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost
