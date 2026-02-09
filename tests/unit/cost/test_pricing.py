"""Tests for pricing module."""

import pytest

from ragnarok_ai.cost.pricing import (
    LOCAL_PROVIDERS,
    PRICING,
    calculate_cost,
    get_pricing,
)


class TestGetPricing:
    """Tests for get_pricing function."""

    def test_get_openai_pricing(self):
        """Test getting pricing for OpenAI models."""
        pricing = get_pricing("gpt-4o")
        assert pricing["input"] == 2.50
        assert pricing["output"] == 10.00

    def test_get_anthropic_pricing(self):
        """Test getting pricing for Anthropic models."""
        pricing = get_pricing("claude-3-sonnet")
        assert pricing["input"] == 3.00
        assert pricing["output"] == 15.00

    def test_get_unknown_model_returns_zero(self):
        """Test that unknown models return zero pricing."""
        pricing = get_pricing("unknown-model-xyz")
        assert pricing["input"] == 0.0
        assert pricing["output"] == 0.0

    def test_local_provider_returns_zero(self):
        """Test that local providers return zero pricing."""
        pricing = get_pricing("gpt-4o", provider="ollama")
        assert pricing["input"] == 0.0
        assert pricing["output"] == 0.0

    def test_vllm_provider_returns_zero(self):
        """Test that vLLM provider returns zero pricing."""
        pricing = get_pricing("mistral-7b", provider="vllm")
        assert pricing["input"] == 0.0
        assert pricing["output"] == 0.0


class TestCalculateCost:
    """Tests for calculate_cost function."""

    def test_calculate_openai_cost(self):
        """Test calculating cost for OpenAI model."""
        # gpt-4o: $2.50/1M input, $10.00/1M output
        cost = calculate_cost(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected)

    def test_calculate_local_cost_is_zero(self):
        """Test that local provider cost is zero."""
        cost = calculate_cost(
            model="llama2",
            input_tokens=10000,
            output_tokens=5000,
            provider="ollama",
        )
        assert cost == 0.0

    def test_calculate_cost_with_custom_pricing(self):
        """Test calculating cost with custom pricing override."""
        custom = {"my-model": {"input": 1.0, "output": 2.0}}
        cost = calculate_cost(
            model="my-model",
            input_tokens=1_000_000,
            output_tokens=500_000,
            custom_pricing=custom,
        )
        expected = 1.0 + 1.0  # 1M * $1 + 0.5M * $2
        assert cost == pytest.approx(expected)

    def test_calculate_cost_zero_tokens(self):
        """Test calculating cost with zero tokens."""
        cost = calculate_cost(model="gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0.0


class TestPricingTable:
    """Tests for pricing table contents."""

    def test_pricing_table_has_openai_models(self):
        """Test that pricing table has OpenAI models."""
        assert "gpt-4o" in PRICING
        assert "gpt-4o-mini" in PRICING
        assert "gpt-4-turbo" in PRICING

    def test_pricing_table_has_anthropic_models(self):
        """Test that pricing table has Anthropic models."""
        assert "claude-3-opus" in PRICING
        assert "claude-3-sonnet" in PRICING
        assert "claude-3-haiku" in PRICING

    def test_pricing_table_has_embedding_models(self):
        """Test that pricing table has embedding models."""
        assert "text-embedding-3-small" in PRICING
        assert "text-embedding-3-large" in PRICING

    def test_local_providers_set(self):
        """Test local providers set."""
        assert "ollama" in LOCAL_PROVIDERS
        assert "vllm" in LOCAL_PROVIDERS
        assert "local" in LOCAL_PROVIDERS
