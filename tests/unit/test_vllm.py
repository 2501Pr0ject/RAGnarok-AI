"""Unit tests for vLLM adapter."""

from __future__ import annotations

import httpx
import pytest
import respx

from ragnarok_ai.adapters.llm.vllm import VLLMAdapter
from ragnarok_ai.core.exceptions import LLMConnectionError
from ragnarok_ai.core.protocols import LLMProtocol

# ============================================================================
# Initialization Tests
# ============================================================================


class TestVLLMAdapterInit:
    """Tests for VLLMAdapter initialization."""

    def test_init_defaults(self) -> None:
        """Test initialization with default values."""
        llm = VLLMAdapter()
        assert llm.base_url == "http://localhost:8000/v1"
        assert llm.model == "default"
        assert llm.embed_model == "default"
        assert llm.api_key is None
        assert llm.timeout == 120.0
        assert llm.is_local is True

    def test_init_with_api_key(self) -> None:
        """Test initialization with optional API key."""
        llm = VLLMAdapter(api_key="vllm-secret-key")
        assert llm.api_key == "vllm-secret-key"

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        llm = VLLMAdapter(
            base_url="http://gpu-server:8000/v1",
            model="mistral-7b",
            embed_model="bge-large",
            timeout=60.0,
        )
        assert llm.base_url == "http://gpu-server:8000/v1"
        assert llm.model == "mistral-7b"
        assert llm.embed_model == "bge-large"
        assert llm.timeout == 60.0

    def test_init_embed_model_defaults_to_model(self) -> None:
        """Test that embed_model defaults to model if not specified."""
        llm = VLLMAdapter(model="mistral-7b")
        assert llm.embed_model == "mistral-7b"

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from base_url."""
        llm = VLLMAdapter(base_url="http://localhost:8000/v1/")
        assert llm.base_url == "http://localhost:8000/v1"


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestVLLMAdapterProtocol:
    """Tests for LLMProtocol compliance."""

    def test_implements_llm_protocol(self) -> None:
        """Test that VLLMAdapter implements LLMProtocol."""
        llm = VLLMAdapter()
        assert isinstance(llm, LLMProtocol)

    def test_is_local_true(self) -> None:
        """Test that is_local is True for local adapter."""
        llm = VLLMAdapter()
        assert llm.is_local is True


# ============================================================================
# Generate Tests
# ============================================================================


class TestVLLMAdapterGenerate:
    """Tests for text generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self) -> None:
        """Test successful text generation."""
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "RAG combines retrieval and generation."}}]},
            )
        )

        llm = VLLMAdapter()
        response = await llm.generate("What is RAG?")

        assert response == "RAG combines retrieval and generation."

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_empty_choices(self) -> None:
        """Test generation with empty choices."""
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )

        llm = VLLMAdapter()
        response = await llm.generate("What is RAG?")

        assert response == ""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_connection_error(self) -> None:
        """Test generation with connection error."""
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        llm = VLLMAdapter()
        with pytest.raises(LLMConnectionError, match="Failed to connect"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_timeout_error(self) -> None:
        """Test generation with timeout."""
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        llm = VLLMAdapter()
        with pytest.raises(LLMConnectionError, match="timed out"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_api_error(self) -> None:
        """Test generation with API error."""
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "Internal server error"})
        )

        llm = VLLMAdapter()
        with pytest.raises(LLMConnectionError, match="API error: 500"):
            await llm.generate("What is RAG?")


# ============================================================================
# Embed Tests
# ============================================================================


class TestVLLMAdapterEmbed:
    """Tests for embedding generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_success(self) -> None:
        """Test successful embedding."""
        respx.post("http://localhost:8000/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}]},
            )
        )

        llm = VLLMAdapter()
        embedding = await llm.embed("Hello world")

        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_empty_data(self) -> None:
        """Test embedding with empty data."""
        respx.post("http://localhost:8000/v1/embeddings").mock(return_value=httpx.Response(200, json={"data": []}))

        llm = VLLMAdapter()
        embedding = await llm.embed("Hello world")

        assert embedding == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_connection_error(self) -> None:
        """Test embedding with connection error."""
        respx.post("http://localhost:8000/v1/embeddings").mock(side_effect=httpx.ConnectError("Connection refused"))

        llm = VLLMAdapter()
        with pytest.raises(LLMConnectionError, match="Failed to connect"):
            await llm.embed("Hello world")

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_api_error(self) -> None:
        """Test embedding with API error."""
        respx.post("http://localhost:8000/v1/embeddings").mock(
            return_value=httpx.Response(400, json={"error": "Model not found"})
        )

        llm = VLLMAdapter()
        with pytest.raises(LLMConnectionError, match="API error: 400"):
            await llm.embed("Hello world")


# ============================================================================
# is_available Tests
# ============================================================================


class TestVLLMAdapterIsAvailable:
    """Tests for availability check."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_true(self) -> None:
        """Test availability when server is reachable."""
        respx.get("http://localhost:8000/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))

        llm = VLLMAdapter()
        assert await llm.is_available() is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_error(self) -> None:
        """Test availability when server returns error."""
        respx.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(500, json={"error": "Server error"})
        )

        llm = VLLMAdapter()
        assert await llm.is_available() is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_connection_error(self) -> None:
        """Test availability when connection fails."""
        respx.get("http://localhost:8000/v1/models").mock(side_effect=httpx.ConnectError("Connection refused"))

        llm = VLLMAdapter()
        assert await llm.is_available() is False


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestVLLMAdapterContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_context_manager_reuses_client(self) -> None:
        """Test that context manager reuses HTTP client."""
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Response"}}]})
        )

        async with VLLMAdapter() as llm:
            assert llm._client is not None
            await llm.generate("Test 1")
            await llm.generate("Test 2")
            assert llm._client is not None

        assert llm._client is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_standalone_creates_new_client(self) -> None:
        """Test that standalone usage creates new client per call."""
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Response"}}]})
        )

        llm = VLLMAdapter()
        assert llm._client is None
        await llm.generate("Test")
        assert llm._client is None


# ============================================================================
# Headers Tests
# ============================================================================


class TestVLLMAdapterHeaders:
    """Tests for request headers."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_headers_without_api_key(self) -> None:
        """Test headers without API key."""
        route = respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Response"}}]})
        )

        llm = VLLMAdapter()
        await llm.generate("Test")

        assert route.called
        request = route.calls[0].request
        assert "Authorization" not in request.headers
        assert request.headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    @respx.mock
    async def test_headers_with_api_key(self) -> None:
        """Test headers with API key."""
        route = respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Response"}}]})
        )

        llm = VLLMAdapter(api_key="vllm-secret-key")
        await llm.generate("Test")

        assert route.called
        request = route.calls[0].request
        assert request.headers["Authorization"] == "Bearer vllm-secret-key"
