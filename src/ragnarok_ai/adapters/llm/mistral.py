"""Mistral AI LLM adapter for ragnarok-ai.

This module provides an async client for the Mistral AI API,
implementing the LLMProtocol for use in evaluation pipelines.

Mistral AI provides high-quality models via OpenAI-compatible API.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, ClassVar

import httpx

from ragnarok_ai.core.exceptions import LLMConnectionError
from ragnarok_ai.cost.tracker import track_usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

# Default configuration
DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_EMBED_MODEL = "mistral-embed"
DEFAULT_TIMEOUT = 60.0


class MistralLLM:
    """Async client for Mistral AI API.

    Implements LLMProtocol for text generation and embedding.
    Uses httpx for async HTTP requests with connection pooling.

    The API is OpenAI-compatible.

    Attributes:
        is_local: Always False - Mistral AI is a cloud service.
        base_url: Base URL for Mistral API.
        model: Model name for text generation.
        embed_model: Model name for embeddings.
        timeout: Request timeout in seconds.

    Example:
        Context manager (recommended for multiple calls):
            >>> async with MistralLLM(api_key="...") as llm:
            ...     response = await llm.generate("What is RAG?")
            ...     embedding = await llm.embed("Hello world")

        Standalone:
            >>> llm = MistralLLM(api_key="...")
            >>> response = await llm.generate("What is RAG?")

        Using environment variable:
            >>> # Set MISTRAL_API_KEY in environment
            >>> llm = MistralLLM()
    """

    is_local: ClassVar[bool] = False

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize MistralLLM client.

        Args:
            api_key: Mistral API key. If not provided, reads from MISTRAL_API_KEY env var.
            base_url: Base URL for Mistral API. Defaults to api.mistral.ai.
            model: Model name for text generation. Defaults to "mistral-small-latest".
            embed_model: Model name for embeddings. Defaults to "mistral-embed".
            timeout: Request timeout in seconds. Defaults to 60.0.

        Raises:
            ValueError: If no API key is provided and MISTRAL_API_KEY is not set.
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            msg = "Mistral API key required. Pass api_key or set MISTRAL_API_KEY environment variable."
            raise ValueError(msg)

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> MistralLLM:
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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[httpx.AsyncClient]:
        """Get an HTTP client for making requests."""
        if self._client is not None:
            yield self._client
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                yield client

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt using Mistral.

        Args:
            prompt: The input prompt for text generation.

        Returns:
            The generated text response.

        Raises:
            LLMConnectionError: If connection to Mistral fails or request errors.

        Example:
            >>> response = await llm.generate("Explain RAG in one sentence.")
            >>> print(response)
            'RAG combines retrieval and generation for accurate responses.'
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=payload, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()

                # Track token usage if cost tracking is active
                usage = data.get("usage", {})
                if usage:
                    track_usage(
                        provider="mistral",
                        model=self.model,
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                    )

                choices = data.get("choices", [])
                if choices:
                    return str(choices[0].get("message", {}).get("content", ""))
                return ""
        except httpx.ConnectError as e:
            msg = f"Failed to connect to Mistral at {self.base_url}: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request to Mistral timed out after {self.timeout}s: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = f"Mistral API error: {e.response.status_code} - {e.response.text}"
            raise LLMConnectionError(msg) from e
        except Exception as e:
            msg = f"Unexpected error calling Mistral: {e}"
            raise LLMConnectionError(msg) from e

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            LLMConnectionError: If connection to Mistral fails or request errors.

        Example:
            >>> embedding = await llm.embed("Hello world")
            >>> print(len(embedding))
            1024
        """
        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.embed_model,
            "input": [text],
        }

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=payload, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()

                # Track token usage if cost tracking is active
                usage = data.get("usage", {})
                if usage:
                    track_usage(
                        provider="mistral",
                        model=self.embed_model,
                        input_tokens=usage.get("total_tokens", 0),
                        output_tokens=0,
                    )

                embeddings = data.get("data", [])
                if embeddings:
                    return [float(x) for x in embeddings[0].get("embedding", [])]
                return []
        except httpx.ConnectError as e:
            msg = f"Failed to connect to Mistral at {self.base_url}: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request to Mistral timed out after {self.timeout}s: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = f"Mistral API error: {e.response.status_code} - {e.response.text}"
            raise LLMConnectionError(msg) from e
        except Exception as e:
            msg = f"Unexpected error calling Mistral: {e}"
            raise LLMConnectionError(msg) from e

    async def is_available(self) -> bool:
        """Check if Mistral API is available and responding.

        Returns:
            True if Mistral is reachable and API key is valid, False otherwise.
        """
        url = f"{self.base_url}/models"

        try:
            async with self._get_client() as client:
                response = await client.get(url, headers=self._get_headers())
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
