"""Unit tests for Together AI LLM adapter."""

from __future__ import annotations

import os
from unittest.mock import patch

import httpx
import pytest
import respx

from ragnarok_ai.adapters.llm.together import TogetherLLM
from ragnarok_ai.core.exceptions import LLMConnectionError

# ============================================================================
# Initialization Tests
# ============================================================================


class TestTogetherLLMInit:
    """Tests for TogetherLLM initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        llm = TogetherLLM(api_key="test_key")
        assert llm.api_key == "test_key"
        assert llm.model == "meta-llama/Llama-3-70b-chat-hf"
        assert llm.embed_model == "togethercomputer/m2-bert-80M-8k-retrieval"
        assert llm.is_local is False

    def test_init_with_env_var(self) -> None:
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"TOGETHER_API_KEY": "env_key"}):
            llm = TogetherLLM()
            assert llm.api_key == "env_key"

    def test_init_without_api_key_raises(self) -> None:
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TOGETHER_API_KEY", None)
            with pytest.raises(ValueError, match="API key required"):
                TogetherLLM()

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        llm = TogetherLLM(
            api_key="test_key",
            base_url="https://custom.together.xyz/v1",
            model="meta-llama/Llama-3-8b-chat-hf",
            embed_model="custom-embed",
            timeout=120.0,
        )
        assert llm.base_url == "https://custom.together.xyz/v1"
        assert llm.model == "meta-llama/Llama-3-8b-chat-hf"
        assert llm.embed_model == "custom-embed"
        assert llm.timeout == 120.0

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from base_url."""
        llm = TogetherLLM(api_key="test_key", base_url="https://api.together.xyz/v1/")
        assert llm.base_url == "https://api.together.xyz/v1"


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestTogetherLLMProtocol:
    """Tests for protocol compliance."""

    def test_is_local_false(self) -> None:
        """Test that is_local is False for cloud adapter."""
        llm = TogetherLLM(api_key="test_key")
        assert llm.is_local is False


# ============================================================================
# Generate Tests
# ============================================================================


class TestTogetherLLMGenerate:
    """Tests for text generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self) -> None:
        """Test successful text generation."""
        respx.post("https://api.together.xyz/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "RAG combines retrieval and generation."}}]},
            )
        )

        llm = TogetherLLM(api_key="test_key")
        response = await llm.generate("What is RAG?")

        assert response == "RAG combines retrieval and generation."

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_empty_choices(self) -> None:
        """Test generation with empty choices."""
        respx.post("https://api.together.xyz/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )

        llm = TogetherLLM(api_key="test_key")
        response = await llm.generate("What is RAG?")

        assert response == ""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_connection_error(self) -> None:
        """Test generation with connection error."""
        respx.post("https://api.together.xyz/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        llm = TogetherLLM(api_key="test_key")
        with pytest.raises(LLMConnectionError, match="Failed to connect"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_timeout_error(self) -> None:
        """Test generation with timeout."""
        respx.post("https://api.together.xyz/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        llm = TogetherLLM(api_key="test_key")
        with pytest.raises(LLMConnectionError, match="timed out"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_api_error(self) -> None:
        """Test generation with API error."""
        respx.post("https://api.together.xyz/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
        )

        llm = TogetherLLM(api_key="invalid_key")
        with pytest.raises(LLMConnectionError, match="API error: 401"):
            await llm.generate("What is RAG?")


# ============================================================================
# Embed Tests
# ============================================================================


class TestTogetherLLMEmbed:
    """Tests for embedding generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_success(self) -> None:
        """Test successful embedding."""
        respx.post("https://api.together.xyz/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}]},
            )
        )

        llm = TogetherLLM(api_key="test_key")
        embedding = await llm.embed("Hello world")

        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_empty_data(self) -> None:
        """Test embedding with empty data."""
        respx.post("https://api.together.xyz/v1/embeddings").mock(return_value=httpx.Response(200, json={"data": []}))

        llm = TogetherLLM(api_key="test_key")
        embedding = await llm.embed("Hello world")

        assert embedding == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_connection_error(self) -> None:
        """Test embedding with connection error."""
        respx.post("https://api.together.xyz/v1/embeddings").mock(side_effect=httpx.ConnectError("Connection refused"))

        llm = TogetherLLM(api_key="test_key")
        with pytest.raises(LLMConnectionError, match="Failed to connect"):
            await llm.embed("Hello world")

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_api_error(self) -> None:
        """Test embedding with API error."""
        respx.post("https://api.together.xyz/v1/embeddings").mock(
            return_value=httpx.Response(400, json={"error": {"message": "Invalid input"}})
        )

        llm = TogetherLLM(api_key="test_key")
        with pytest.raises(LLMConnectionError, match="API error: 400"):
            await llm.embed("Hello world")


# ============================================================================
# is_available Tests
# ============================================================================


class TestTogetherLLMIsAvailable:
    """Tests for availability check."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_true(self) -> None:
        """Test availability when API is reachable."""
        respx.get("https://api.together.xyz/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))

        llm = TogetherLLM(api_key="test_key")
        assert await llm.is_available() is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_error(self) -> None:
        """Test availability when API returns error."""
        respx.get("https://api.together.xyz/v1/models").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )

        llm = TogetherLLM(api_key="invalid_key")
        assert await llm.is_available() is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_connection_error(self) -> None:
        """Test availability when connection fails."""
        respx.get("https://api.together.xyz/v1/models").mock(side_effect=httpx.ConnectError("Connection refused"))

        llm = TogetherLLM(api_key="test_key")
        assert await llm.is_available() is False


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestTogetherLLMContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_context_manager_reuses_client(self) -> None:
        """Test that context manager reuses HTTP client."""
        respx.post("https://api.together.xyz/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Response"}}]})
        )

        async with TogetherLLM(api_key="test_key") as llm:
            assert llm._client is not None
            await llm.generate("Test 1")
            await llm.generate("Test 2")
            assert llm._client is not None

        assert llm._client is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_standalone_creates_new_client(self) -> None:
        """Test that standalone usage creates new client per call."""
        respx.post("https://api.together.xyz/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Response"}}]})
        )

        llm = TogetherLLM(api_key="test_key")
        assert llm._client is None
        await llm.generate("Test")
        assert llm._client is None
