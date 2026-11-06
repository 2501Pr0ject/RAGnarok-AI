"""Tests for the generic LLMResponse shape."""

from __future__ import annotations

import pytest

from ragnarok_ai.core import LLMResponse


def test_defaults_are_unknown() -> None:
    r = LLMResponse("text")
    assert (r.prompt_tokens, r.completion_tokens, r.model, r.options, r.latency_ms) == (None, None, None, {}, None)


def test_is_frozen() -> None:
    r = LLMResponse("text", prompt_tokens=3, completion_tokens=1, model="m", options={"seed": 42}, latency_ms=1.5)
    with pytest.raises(AttributeError):
        r.text = "other"  # type: ignore[misc]
