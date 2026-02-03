"""Tests for Ollama LLM adapter."""

from __future__ import annotations

import httpx
import pytest
import respx
from httpx import Response

from ragnarok_ai.adapters.llm.ollama import (
    DEFAULT_BASE_URL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    OllamaLLM,
)
from ragnarok_ai.core.exceptions import LLMConnectionError


class TestOllamaLLMInit:
    """Tests for OllamaLLM initialization."""

    def test_default_values(self) -> None:
        """Default values are set correctly."""
        llm = OllamaLLM()

        assert llm.base_url == DEFAULT_BASE_URL
        assert llm.model == DEFAULT_MODEL
        assert llm.embed_model == DEFAULT_EMBED_MODEL
        assert llm.timeout == DEFAULT_TIMEOUT

    def test_custom_values(self) -> None:
        """Custom values are set correctly."""
        llm = OllamaLLM(
            base_url="http://custom:8080",
            model="llama2",
            embed_model="mxbai-embed-large",
            timeout=120.0,
        )

        assert llm.base_url == "http://custom:8080"
        assert llm.model == "llama2"
        assert llm.embed_model == "mxbai-embed-large"
        assert llm.timeout == 120.0

    def test_trailing_slash_removed(self) -> None:
        """Trailing slash is removed from base_url."""
        llm = OllamaLLM(base_url="http://localhost:11434/")

        assert llm.base_url == "http://localhost:11434"


class TestOllamaLLMGenerate:
    """Tests for OllamaLLM.generate method."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """Successful generation returns response text."""
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=Response(
                200,
                json={"response": "RAG is retrieval-augmented generation."},
            )
        )

        llm = OllamaLLM()
        result = await llm.generate("What is RAG?")

        assert result == "RAG is retrieval-augmented generation."

    @respx.mock
    @pytest.mark.asyncio
    async def test_generate_empty_response(self) -> None:
        """Empty response field returns empty string."""
        respx.post("http://localhost:11434/api/generate").mock(return_value=Response(200, json={}))

        llm = OllamaLLM()
        result = await llm.generate("prompt")

        assert result == ""

    @respx.mock
    @pytest.mark.asyncio
    async def test_generate_connection_error(self) -> None:
        """Connection error raises LLMConnectionError."""
        respx.post("http://localhost:11434/api/generate").mock(side_effect=httpx.ConnectError("Connection refused"))

        llm = OllamaLLM()

        with pytest.raises(LLMConnectionError, match="Failed to connect"):
            await llm.generate("prompt")

    @respx.mock
    @pytest.mark.asyncio
    async def test_generate_http_error(self) -> None:
        """HTTP error raises LLMConnectionError."""
        respx.post("http://localhost:11434/api/generate").mock(return_value=Response(500, text="Internal Server Error"))

        llm = OllamaLLM()

        with pytest.raises(LLMConnectionError, match="Ollama API error: 500"):
            await llm.generate("prompt")

    @respx.mock
    @pytest.mark.asyncio
    async def test_generate_uses_correct_model(self) -> None:
        """Generate uses the configured model."""
        route = respx.post("http://localhost:11434/api/generate").mock(
            return_value=Response(200, json={"response": "ok"})
        )

        llm = OllamaLLM(model="llama2")
        await llm.generate("test")

        assert route.called
        import json

        request_data = json.loads(route.calls[0].request.content)
        assert request_data["model"] == "llama2"


class TestOllamaLLMEmbed:
    """Tests for OllamaLLM.embed method."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_embed_success(self) -> None:
        """Successful embedding returns vector."""
        respx.post("http://localhost:11434/api/embed").mock(
            return_value=Response(
                200,
                json={"embeddings": [[0.1, 0.2, 0.3, 0.4]]},
            )
        )

        llm = OllamaLLM()
        result = await llm.embed("Hello world")

        assert result == [0.1, 0.2, 0.3, 0.4]

    @respx.mock
    @pytest.mark.asyncio
    async def test_embed_legacy_format(self) -> None:
        """Legacy embedding format is supported."""
        respx.post("http://localhost:11434/api/embed").mock(
            return_value=Response(
                200,
                json={"embedding": [0.5, 0.6, 0.7]},
            )
        )

        llm = OllamaLLM()
        result = await llm.embed("Hello")

        assert result == [0.5, 0.6, 0.7]

    @respx.mock
    @pytest.mark.asyncio
    async def test_embed_empty_response(self) -> None:
        """Empty response returns empty list."""
        respx.post("http://localhost:11434/api/embed").mock(return_value=Response(200, json={"embeddings": []}))

        llm = OllamaLLM()
        result = await llm.embed("text")

        assert result == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_embed_connection_error(self) -> None:
        """Connection error raises LLMConnectionError."""
        respx.post("http://localhost:11434/api/embed").mock(side_effect=httpx.ConnectError("Connection refused"))

        llm = OllamaLLM()

        with pytest.raises(LLMConnectionError, match="Failed to connect"):
            await llm.embed("text")

    @respx.mock
    @pytest.mark.asyncio
    async def test_embed_http_error(self) -> None:
        """HTTP error raises LLMConnectionError."""
        respx.post("http://localhost:11434/api/embed").mock(return_value=Response(404, text="Model not found"))

        llm = OllamaLLM()

        with pytest.raises(LLMConnectionError, match="Ollama API error: 404"):
            await llm.embed("text")

    @respx.mock
    @pytest.mark.asyncio
    async def test_embed_uses_embed_model(self) -> None:
        """Embed uses the configured embed_model."""
        route = respx.post("http://localhost:11434/api/embed").mock(
            return_value=Response(200, json={"embeddings": [[0.1]]})
        )

        llm = OllamaLLM(embed_model="mxbai-embed-large")
        await llm.embed("test")

        assert route.called
        import json

        request_data = json.loads(route.calls[0].request.content)
        assert request_data["model"] == "mxbai-embed-large"


class TestOllamaLLMIsAvailable:
    """Tests for OllamaLLM.is_available method."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        """Returns True when Ollama responds with 200."""
        respx.get("http://localhost:11434/api/tags").mock(return_value=Response(200, json={"models": []}))

        llm = OllamaLLM()
        result = await llm.is_available()

        assert result is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_is_available_false_on_error(self) -> None:
        """Returns False when Ollama responds with error."""
        respx.get("http://localhost:11434/api/tags").mock(return_value=Response(500))

        llm = OllamaLLM()
        result = await llm.is_available()

        assert result is False

    @respx.mock
    @pytest.mark.asyncio
    async def test_is_available_false_on_connection_error(self) -> None:
        """Returns False when connection fails."""
        respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("Connection refused"))

        llm = OllamaLLM()
        result = await llm.is_available()

        assert result is False


class TestOllamaLLMProtocolCompliance:
    """Tests for LLMProtocol compliance."""

    def test_implements_protocol(self) -> None:
        """OllamaLLM implements LLMProtocol."""
        from ragnarok_ai.core.protocols import LLMProtocol

        llm = OllamaLLM()

        # Protocol check using isinstance (runtime_checkable)
        assert isinstance(llm, LLMProtocol)

    def test_has_generate_method(self) -> None:
        """OllamaLLM has generate method."""
        llm = OllamaLLM()

        assert hasattr(llm, "generate")
        assert callable(llm.generate)

    def test_has_embed_method(self) -> None:
        """OllamaLLM has embed method."""
        llm = OllamaLLM()

        assert hasattr(llm, "embed")
        assert callable(llm.embed)


class TestOllamaLLMContextManager:
    """Tests for OllamaLLM context manager functionality."""

    def test_client_is_none_initially(self) -> None:
        """Client is None before entering context manager."""
        llm = OllamaLLM()

        assert llm._client is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_context_manager_creates_client(self) -> None:
        """Entering context manager creates HTTP client."""
        respx.post("http://localhost:11434/api/generate").mock(return_value=Response(200, json={"response": "ok"}))

        llm = OllamaLLM()

        async with llm:
            assert llm._client is not None
            assert isinstance(llm._client, httpx.AsyncClient)

    @respx.mock
    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self) -> None:
        """Exiting context manager closes HTTP client."""
        respx.post("http://localhost:11434/api/generate").mock(return_value=Response(200, json={"response": "ok"}))

        llm = OllamaLLM()

        async with llm:
            pass

        assert llm._client is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_context_manager_reuses_client(self) -> None:
        """Multiple calls in context manager reuse the same client."""
        respx.post("http://localhost:11434/api/generate").mock(return_value=Response(200, json={"response": "ok"}))
        respx.post("http://localhost:11434/api/embed").mock(return_value=Response(200, json={"embeddings": [[0.1]]}))

        llm = OllamaLLM()

        async with llm:
            client_before = llm._client
            await llm.generate("test1")
            await llm.embed("test2")
            await llm.generate("test3")
            client_after = llm._client

            # Same client instance throughout
            assert client_before is client_after

    @respx.mock
    @pytest.mark.asyncio
    async def test_standalone_mode_works(self) -> None:
        """Standalone mode (without context manager) still works."""
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=Response(200, json={"response": "standalone works"})
        )

        llm = OllamaLLM()
        result = await llm.generate("test")

        assert result == "standalone works"
        assert llm._client is None  # No persistent client

    @respx.mock
    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self) -> None:
        """Context manager returns the OllamaLLM instance."""
        respx.post("http://localhost:11434/api/generate").mock(return_value=Response(200, json={"response": "ok"}))

        llm = OllamaLLM()

        async with llm as entered_llm:
            assert entered_llm is llm
