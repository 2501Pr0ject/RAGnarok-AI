"""Tests for tokenizer module."""

from unittest.mock import patch

from ragnarok_ai.cost.tokenizer import (
    _get_encoding,
    count_tokens,
    estimate_tokens,
    is_tiktoken_available,
)


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

    def test_count_tokens_fallback_when_no_tiktoken(self):
        """Test count_tokens falls back to estimation when tiktoken unavailable."""
        import ragnarok_ai.cost.tokenizer as tokenizer_module

        # Temporarily disable tiktoken
        original_available = tokenizer_module._tiktoken_available
        tokenizer_module._tiktoken_available = False

        # Clear the encoding cache
        _get_encoding.cache_clear()

        try:
            text = "a" * 100  # 100 characters
            count = count_tokens(text)
            # Should use fallback estimation (len // 4)
            assert count == 25
        finally:
            tokenizer_module._tiktoken_available = original_available
            _get_encoding.cache_clear()

    def test_get_encoding_returns_none_when_unavailable(self):
        """Test _get_encoding returns None when tiktoken unavailable."""
        import ragnarok_ai.cost.tokenizer as tokenizer_module

        original_available = tokenizer_module._tiktoken_available
        tokenizer_module._tiktoken_available = False
        _get_encoding.cache_clear()

        try:
            encoding = _get_encoding("gpt-4")
            assert encoding is None
        finally:
            tokenizer_module._tiktoken_available = original_available
            _get_encoding.cache_clear()

    def test_estimate_tokens_short_text(self):
        """Test estimate_tokens with short text."""
        # Text shorter than 4 characters
        assert estimate_tokens("ab") == 0
        assert estimate_tokens("abc") == 0
        assert estimate_tokens("abcd") == 1

    def test_estimate_tokens_unicode(self):
        """Test estimate_tokens with unicode characters."""
        # Unicode characters count as multiple bytes but still use char count
        text = "Hello 世界"  # 8 characters
        estimate = estimate_tokens(text)
        assert estimate == 2  # 8 // 4

    def test_count_tokens_with_different_model(self):
        """Test count_tokens with different model name."""
        # Should work with any model name
        count = count_tokens("Hello, world!", model="gpt-3.5-turbo")
        assert count > 0

    def test_count_tokens_with_unknown_model(self):
        """Test count_tokens with unknown model falls back gracefully."""
        # Should still work even with unknown model
        count = count_tokens("Hello, world!", model="unknown-model-xyz")
        assert count > 0
