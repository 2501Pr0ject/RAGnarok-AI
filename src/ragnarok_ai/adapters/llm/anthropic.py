"""Anthropic LLM adapter for ragnarok-ai.

This module provides an async client for the Anthropic API,
implementing the LLMProtocol for use in evaluation pipelines.

Note: Anthropic does not provide an embedding API. The embed() method
will raise NotImplementedError. Use a different provider for embeddings.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx

from ragnarok_ai.core.exceptions import LLMConnectionError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

# Default configuration
DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 1024
API_VERSION = "2023-06-01"


class AnthropicLLM:
    """Async client for Anthropic API.

    Implements LLMProtocol for text generation. Note that Anthropic does not
    provide an embedding API, so embed() raises NotImplementedError.

    Can be used as a context manager for optimal performance (reuses connections),
    or standalone for simpler use cases.

    Attributes:
        is_local: Always False - Anthropic is a cloud service.
        base_url: Base URL for Anthropic API.
        model: Model name for text generation.
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.

    Example:
        Context manager (recommended for multiple calls):
            >>> async with AnthropicLLM(api_key="sk-ant-...") as llm:
            ...     response = await llm.generate("What is RAG?")

        Standalone (simpler, creates new connection per call):
            >>> llm = AnthropicLLM(api_key="sk-ant-...")
            >>> response = await llm.generate("What is RAG?")

        Using environment variable:
            >>> # Set ANTHROPIC_API_KEY in environment
            >>> llm = AnthropicLLM()  # API key read from env
    """

    is_local: bool = False

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize AnthropicLLM client.

        Args:
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
            base_url: Base URL for Anthropic API. Defaults to api.anthropic.com.
            model: Model name for text generation. Defaults to "claude-sonnet-4-20250514".
            max_tokens: Maximum tokens in response. Defaults to 1024.
            timeout: Request timeout in seconds. Defaults to 60.0.

        Raises:
            ValueError: If no API key is provided and ANTHROPIC_API_KEY is not set.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            msg = "Anthropic API key required. Pass api_key or set ANTHROPIC_API_KEY environment variable."
            raise ValueError(msg)

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> AnthropicLLM:
        """Enter async context manager, creating a reusable HTTP client."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager, closing the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": API_VERSION,
            "Content-Type": "application/json",
        }

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[httpx.AsyncClient]:
        """Get an HTTP client for making requests.

        If used as a context manager, returns the managed client.
        Otherwise, creates a temporary client for this request.

        Yields:
            An httpx.AsyncClient instance.
        """
        if self._client is not None:
            # Reuse managed client (context manager mode)
            yield self._client
        else:
            # Create temporary client (standalone mode)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                yield client

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt using Anthropic.

        Args:
            prompt: The input prompt for text generation.

        Returns:
            The generated text response.

        Raises:
            LLMConnectionError: If connection to Anthropic fails or request errors.

        Example:
            >>> response = await llm.generate("Explain RAG in one sentence.")
            >>> print(response)
            'RAG combines retrieval and generation for accurate responses.'
        """
        url = f"{self.base_url}/v1/messages"
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=payload, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()
                content = data.get("content", [])
                if content and isinstance(content, list):
                    # Extract text from content blocks
                    texts = [block.get("text", "") for block in content if block.get("type") == "text"]
                    return "".join(texts)
                return ""
        except httpx.ConnectError as e:
            msg = f"Failed to connect to Anthropic at {self.base_url}: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request to Anthropic timed out after {self.timeout}s: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = f"Anthropic API error: {e.response.status_code} - {e.response.text}"
            raise LLMConnectionError(msg) from e
        except Exception as e:
            msg = f"Unexpected error calling Anthropic: {e}"
            raise LLMConnectionError(msg) from e

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Note: Anthropic does not provide an embedding API.
        Use OpenAI, Ollama, or another provider for embeddings.

        Args:
            text: The text to embed (unused).

        Raises:
            NotImplementedError: Always, as Anthropic doesn't support embeddings.
        """
        msg = "Anthropic does not provide an embedding API. Use OpenAI, Ollama, or another provider for embeddings."
        raise NotImplementedError(msg)

    async def is_available(self) -> bool:
        """Check if Anthropic API is available and responding.

        Returns:
            True if Anthropic is reachable and API key is valid, False otherwise.

        Example:
            >>> if await llm.is_available():
            ...     print("Anthropic is ready")
        """
        # Anthropic doesn't have a simple health endpoint, so we make a minimal request
        url = f"{self.base_url}/v1/messages"
        payload = {
            "model": self.model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "Hi"}],
        }

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=payload, headers=self._get_headers())
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
