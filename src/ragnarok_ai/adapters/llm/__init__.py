"""LLM adapters for ragnarok-ai.

This module provides adapters for various LLM providers.

Adapters are imported on first access: they depend on httpx, an optional
dependency (``pip install ragnarok-ai[ollama]``), and importing this package
must not fail on a base install.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragnarok_ai.adapters.llm.anthropic import AnthropicLLM
    from ragnarok_ai.adapters.llm.groq import GroqLLM
    from ragnarok_ai.adapters.llm.mistral import MistralLLM
    from ragnarok_ai.adapters.llm.ollama import OllamaLLM
    from ragnarok_ai.adapters.llm.openai import OpenAILLM
    from ragnarok_ai.adapters.llm.together import TogetherLLM
    from ragnarok_ai.adapters.llm.vllm import VLLMAdapter

# Public name -> module defining it
_ADAPTER_MODULES = {
    "AnthropicLLM": "ragnarok_ai.adapters.llm.anthropic",
    "GroqLLM": "ragnarok_ai.adapters.llm.groq",
    "MistralLLM": "ragnarok_ai.adapters.llm.mistral",
    "OllamaLLM": "ragnarok_ai.adapters.llm.ollama",
    "OpenAILLM": "ragnarok_ai.adapters.llm.openai",
    "TogetherLLM": "ragnarok_ai.adapters.llm.together",
    "VLLMAdapter": "ragnarok_ai.adapters.llm.vllm",
}


def __getattr__(name: str) -> Any:
    module = _ADAPTER_MODULES.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # cache: later lookups bypass __getattr__
    return value


__all__ = [
    "AnthropicLLM",
    "GroqLLM",
    "MistralLLM",
    "OllamaLLM",
    "OpenAILLM",
    "TogetherLLM",
    "VLLMAdapter",
]
