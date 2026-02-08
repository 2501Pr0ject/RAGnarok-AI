"""Mock LLM plugin for testing RAGnarok-AI plugin system."""

from __future__ import annotations


class MockLLM:
    """A mock LLM adapter for testing plugin discovery."""

    is_local: bool = True

    def __init__(self, model: str = "mock-model") -> None:
        self.model = model

    async def generate(self, prompt: str) -> str:
        """Return a mock response."""
        return f"[MockLLM] Response to: {prompt[:50]}"

    async def embed(self, text: str) -> list[float]:
        """Return mock embeddings."""
        _ = text  # Used to satisfy protocol
        return [0.1, 0.2, 0.3, 0.4, 0.5]


__all__ = ["MockLLM"]
