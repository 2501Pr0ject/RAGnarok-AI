"""Ollama LLM adapter for ragnarok-ai.

This module provides an async client for the Ollama API,
implementing the LLMProtocol for use in evaluation pipelines.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx

from ragnarok_ai.core.exceptions import LLMConnectionError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

# Default configuration
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"
DEFAULT_TIMEOUT = 60.0
DEFAULT_EMBED_MODEL = "nomic-embed-text"


class OllamaLLM:
    """Async client for Ollama LLM API.

    Implements LLMProtocol for text generation and embedding.
    Uses httpx for async HTTP requests with connection pooling.

    Can be used as a context manager for optimal performance (reuses connections),
    or standalone for simpler use cases.

    Attributes:
        base_url: Base URL for Ollama API.
        model: Model name for text generation.
        embed_model: Model name for embeddings.
        timeout: Request timeout in seconds.

    Example:
        Context manager (recommended for multiple calls):
            >>> async with OllamaLLM(model="mistral") as llm:
            ...     response = await llm.generate("What is RAG?")
            ...     embedding = await llm.embed("Hello world")

        Standalone (simpler, creates new connection per call):
            >>> llm = OllamaLLM(model="mistral")
            >>> response = await llm.generate("What is RAG?")
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize OllamaLLM client.

        Args:
            base_url: Base URL for Ollama API. Defaults to localhost:11434.
            model: Model name for text generation. Defaults to "mistral".
            embed_model: Model name for embeddings. Defaults to "nomic-embed-text".
            timeout: Request timeout in seconds. Defaults to 60.0.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OllamaLLM:
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
        """Generate text from a prompt using Ollama.

        Args:
            prompt: The input prompt for text generation.

        Returns:
            The generated text response.

        Raises:
            LLMConnectionError: If connection to Ollama fails or request errors.

        Example:
            >>> response = await llm.generate("Explain RAG in one sentence.")
            >>> print(response)
            'RAG combines retrieval and generation for accurate responses.'
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return str(data.get("response", ""))
        except httpx.ConnectError as e:
            msg = f"Failed to connect to Ollama at {self.base_url}: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request to Ollama timed out after {self.timeout}s: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = f"Ollama API error: {e.response.status_code} - {e.response.text}"
            raise LLMConnectionError(msg) from e
        except Exception as e:
            msg = f"Unexpected error calling Ollama: {e}"
            raise LLMConnectionError(msg) from e

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            LLMConnectionError: If connection to Ollama fails or request errors.

        Example:
            >>> embedding = await llm.embed("Hello world")
            >>> print(len(embedding))
            768
        """
        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.embed_model,
            "input": text,
        }

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                # Ollama returns embeddings in "embeddings" field (list of lists)
                embeddings = data.get("embeddings", [])
                if embeddings and isinstance(embeddings[0], list):
                    return [float(x) for x in embeddings[0]]
                # Fallback for older API format
                embedding = data.get("embedding", [])
                return [float(x) for x in embedding]
        except httpx.ConnectError as e:
            msg = f"Failed to connect to Ollama at {self.base_url}: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request to Ollama timed out after {self.timeout}s: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = f"Ollama API error: {e.response.status_code} - {e.response.text}"
            raise LLMConnectionError(msg) from e
        except Exception as e:
            msg = f"Unexpected error calling Ollama: {e}"
            raise LLMConnectionError(msg) from e

    async def is_available(self) -> bool:
        """Check if Ollama is available and responding.

        Returns:
            True if Ollama is reachable, False otherwise.

        Example:
            >>> if await llm.is_available():
            ...     print("Ollama is ready")
        """
        url = f"{self.base_url}/api/tags"

        try:
            async with self._get_client() as client:
                response = await client.get(url)
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
