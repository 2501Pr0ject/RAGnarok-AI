"""vLLM adapter for ragnarok-ai.

This module provides an async client for vLLM's OpenAI-compatible API,
implementing the LLMProtocol for use in evaluation pipelines.

vLLM is a high-performance local inference server that exposes an
OpenAI-compatible API, allowing local model serving with high throughput.
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
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "default"
DEFAULT_TIMEOUT = 120.0  # vLLM may need more time for large models


class VLLMAdapter:
    """Async client for vLLM's OpenAI-compatible API.

    Implements LLMProtocol for text generation and embedding.
    vLLM provides high-performance local inference with an OpenAI-compatible API.

    Can be used as a context manager for optimal performance (reuses connections),
    or standalone for simpler use cases.

    Attributes:
        is_local: Always True - vLLM runs entirely locally.
        base_url: Base URL for vLLM server.
        api_key: Optional API key if vLLM is configured with auth.
        model: Model name for text generation.
        embed_model: Model name for embeddings (if supported).
        timeout: Request timeout in seconds.

    Example:
        Context manager (recommended for multiple calls):
            >>> async with VLLMAdapter(model="mistral-7b") as llm:
            ...     response = await llm.generate("What is RAG?")
            ...     embedding = await llm.embed("Hello world")

        Standalone (simpler, creates new connection per call):
            >>> llm = VLLMAdapter(model="mistral-7b")
            >>> response = await llm.generate("What is RAG?")

        Custom server URL:
            >>> llm = VLLMAdapter(base_url="http://gpu-server:8000/v1")
    """

    is_local: bool = True

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        embed_model: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize VLLMAdapter client.

        Args:
            base_url: Base URL for vLLM server. Defaults to localhost:8000/v1.
            model: Model name for text generation. Defaults to "default".
            embed_model: Model name for embeddings. Defaults to same as model.
            api_key: Optional API key if vLLM server requires authentication.
            timeout: Request timeout in seconds. Defaults to 120.0.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model or model
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def __aenter__(self) -> VLLMAdapter:
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
            yield self._client
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                yield client

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt using vLLM.

        Args:
            prompt: The input prompt for text generation.

        Returns:
            The generated text response.

        Raises:
            LLMConnectionError: If connection to vLLM fails or request errors.

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
            msg = f"Failed to connect to vLLM at {self.base_url}: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request to vLLM timed out after {self.timeout}s: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = f"vLLM API error: {e.response.status_code} - {e.response.text}"
            raise LLMConnectionError(msg) from e
        except Exception as e:
            msg = f"Unexpected error calling vLLM: {e}"
            raise LLMConnectionError(msg) from e

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Note: Embedding support depends on the model loaded in vLLM.
        Not all models support embeddings.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            LLMConnectionError: If connection to vLLM fails or request errors.

        Example:
            >>> embedding = await llm.embed("Hello world")
            >>> print(len(embedding))
            4096
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
            msg = f"Failed to connect to vLLM at {self.base_url}: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request to vLLM timed out after {self.timeout}s: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = f"vLLM API error: {e.response.status_code} - {e.response.text}"
            raise LLMConnectionError(msg) from e
        except Exception as e:
            msg = f"Unexpected error calling vLLM: {e}"
            raise LLMConnectionError(msg) from e

    async def is_available(self) -> bool:
        """Check if vLLM server is available and responding.

        Returns:
            True if vLLM is reachable, False otherwise.

        Example:
            >>> if await llm.is_available():
            ...     print("vLLM is ready")
        """
        url = f"{self.base_url}/models"

        try:
            async with self._get_client() as client:
                response = await client.get(url, headers=self._get_headers())
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
