"""Tests for tokenizer module."""

from ragnarok_ai.cost.tokenizer import count_tokens, estimate_tokens, is_tiktoken_available


class TestTokenizer:
    """Tests for tokenizer functions."""

    def test_count_tokens_empty_string(self):
        """Test counting tokens for empty string."""
        assert count_tokens("") == 0

    def test_count_tokens_simple_text(self):
        """Test counting tokens for simple text."""
        # Should return > 0 tokens
        count = count_tokens("Hello, world!")
        assert count > 0

    def test_estimate_tokens_empty_string(self):
        """Test estimating tokens for empty string."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_approximation(self):
        """Test that estimate_tokens uses ~4 chars per token."""
        text = "a" * 100  # 100 characters
        estimate = estimate_tokens(text)
        assert estimate == 25  # 100 / 4

    def test_is_tiktoken_available_returns_bool(self):
        """Test that is_tiktoken_available returns bool."""
        result = is_tiktoken_available()
        assert isinstance(result, bool)

    def test_count_tokens_consistency(self):
        """Test that count_tokens is consistent."""
        text = "The quick brown fox jumps over the lazy dog."
        count1 = count_tokens(text)
        count2 = count_tokens(text)
        assert count1 == count2
