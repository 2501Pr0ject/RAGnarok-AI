"""OpenAI LLM adapter for ragnarok-ai.

This module provides an async client for the OpenAI API,
implementing the LLMProtocol for use in evaluation pipelines.
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
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_TIMEOUT = 60.0


class OpenAILLM:
    """Async client for OpenAI API.

    Implements LLMProtocol for text generation and embedding.
    Uses httpx for async HTTP requests with connection pooling.

    Can be used as a context manager for optimal performance (reuses connections),
    or standalone for simpler use cases.

    Attributes:
        is_local: Always False - OpenAI is a cloud service.
        base_url: Base URL for OpenAI API.
        model: Model name for text generation.
        embed_model: Model name for embeddings.
        timeout: Request timeout in seconds.

    Example:
        Context manager (recommended for multiple calls):
            >>> async with OpenAILLM(api_key="sk-...") as llm:
            ...     response = await llm.generate("What is RAG?")
            ...     embedding = await llm.embed("Hello world")

        Standalone (simpler, creates new connection per call):
            >>> llm = OpenAILLM(api_key="sk-...")
            >>> response = await llm.generate("What is RAG?")

        Using environment variable:
            >>> # Set OPENAI_API_KEY in environment
            >>> llm = OpenAILLM()  # API key read from env
    """

    is_local: bool = False

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize OpenAILLM client.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            base_url: Base URL for OpenAI API. Defaults to api.openai.com.
            model: Model name for text generation. Defaults to "gpt-4o-mini".
            embed_model: Model name for embeddings. Defaults to "text-embedding-3-small".
            timeout: Request timeout in seconds. Defaults to 60.0.

        Raises:
            ValueError: If no API key is provided and OPENAI_API_KEY is not set.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            msg = "OpenAI API key required. Pass api_key or set OPENAI_API_KEY environment variable."
            raise ValueError(msg)

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OpenAILLM:
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
        """Generate text from a prompt using OpenAI.

        Args:
            prompt: The input prompt for text generation.

        Returns:
            The generated text response.

        Raises:
            LLMConnectionError: If connection to OpenAI fails or request errors.

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
                choices = data.get("choices", [])
                if choices:
                    return str(choices[0].get("message", {}).get("content", ""))
                return ""
        except httpx.ConnectError as e:
            msg = f"Failed to connect to OpenAI at {self.base_url}: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request to OpenAI timed out after {self.timeout}s: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = f"OpenAI API error: {e.response.status_code} - {e.response.text}"
            raise LLMConnectionError(msg) from e
        except Exception as e:
            msg = f"Unexpected error calling OpenAI: {e}"
            raise LLMConnectionError(msg) from e

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            LLMConnectionError: If connection to OpenAI fails or request errors.

        Example:
            >>> embedding = await llm.embed("Hello world")
            >>> print(len(embedding))
            1536
        """
        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.embed_model,
            "input": text,
        }

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=payload, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("data", [])
                if embeddings:
                    return [float(x) for x in embeddings[0].get("embedding", [])]
                return []
        except httpx.ConnectError as e:
            msg = f"Failed to connect to OpenAI at {self.base_url}: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request to OpenAI timed out after {self.timeout}s: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = f"OpenAI API error: {e.response.status_code} - {e.response.text}"
            raise LLMConnectionError(msg) from e
        except Exception as e:
            msg = f"Unexpected error calling OpenAI: {e}"
            raise LLMConnectionError(msg) from e

    async def is_available(self) -> bool:
        """Check if OpenAI API is available and responding.

        Returns:
            True if OpenAI is reachable and API key is valid, False otherwise.

        Example:
            >>> if await llm.is_available():
            ...     print("OpenAI is ready")
        """
        url = f"{self.base_url}/models"

        try:
            async with self._get_client() as client:
                response = await client.get(url, headers=self._get_headers())
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
