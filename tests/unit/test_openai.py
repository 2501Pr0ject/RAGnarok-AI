"""Unit tests for OpenAI LLM adapter."""

from __future__ import annotations

import os
from unittest.mock import patch

import httpx
import pytest
import respx

from ragnarok_ai.adapters.llm.openai import OpenAILLM
from ragnarok_ai.core.exceptions import LLMConnectionError
from ragnarok_ai.core.protocols import LLMProtocol

# ============================================================================
# Initialization Tests
# ============================================================================


class TestOpenAILLMInit:
    """Tests for OpenAILLM initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        llm = OpenAILLM(api_key="sk-test-key")
        assert llm.api_key == "sk-test-key"
        assert llm.model == "gpt-4o-mini"
        assert llm.embed_model == "text-embedding-3-small"
        assert llm.is_local is False

    def test_init_with_env_var(self) -> None:
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-key"}):
            llm = OpenAILLM()
            assert llm.api_key == "sk-env-key"

    def test_init_without_api_key_raises(self) -> None:
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove OPENAI_API_KEY if it exists
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="API key required"):
                OpenAILLM()

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        llm = OpenAILLM(
            api_key="sk-test",
            base_url="https://custom.openai.com/v1",
            model="gpt-4",
            embed_model="text-embedding-ada-002",
            timeout=120.0,
        )
        assert llm.base_url == "https://custom.openai.com/v1"
        assert llm.model == "gpt-4"
        assert llm.embed_model == "text-embedding-ada-002"
        assert llm.timeout == 120.0

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from base_url."""
        llm = OpenAILLM(api_key="sk-test", base_url="https://api.openai.com/v1/")
        assert llm.base_url == "https://api.openai.com/v1"


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestOpenAILLMProtocol:
    """Tests for LLMProtocol compliance."""

    def test_implements_llm_protocol(self) -> None:
        """Test that OpenAILLM implements LLMProtocol."""
        llm = OpenAILLM(api_key="sk-test")
        assert isinstance(llm, LLMProtocol)

    def test_is_local_false(self) -> None:
        """Test that is_local is False for cloud adapter."""
        llm = OpenAILLM(api_key="sk-test")
        assert llm.is_local is False


# ============================================================================
# Generate Tests
# ============================================================================


class TestOpenAILLMGenerate:
    """Tests for text generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self) -> None:
        """Test successful text generation."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "RAG combines retrieval and generation."}}]},
            )
        )

        llm = OpenAILLM(api_key="sk-test")
        response = await llm.generate("What is RAG?")

        assert response == "RAG combines retrieval and generation."

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_empty_choices(self) -> None:
        """Test generation with empty choices."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )

        llm = OpenAILLM(api_key="sk-test")
        response = await llm.generate("What is RAG?")

        assert response == ""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_connection_error(self) -> None:
        """Test generation with connection error."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        llm = OpenAILLM(api_key="sk-test")
        with pytest.raises(LLMConnectionError, match="Failed to connect"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_timeout_error(self) -> None:
        """Test generation with timeout."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        llm = OpenAILLM(api_key="sk-test")
        with pytest.raises(LLMConnectionError, match="timed out"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_api_error(self) -> None:
        """Test generation with API error."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
        )

        llm = OpenAILLM(api_key="sk-invalid")
        with pytest.raises(LLMConnectionError, match="API error: 401"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_rate_limit(self) -> None:
        """Test generation with rate limit error."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
        )

        llm = OpenAILLM(api_key="sk-test")
        with pytest.raises(LLMConnectionError, match="API error: 429"):
            await llm.generate("What is RAG?")


# ============================================================================
# Embed Tests
# ============================================================================


class TestOpenAILLMEmbed:
    """Tests for embedding generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_success(self) -> None:
        """Test successful embedding."""
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}]},
            )
        )

        llm = OpenAILLM(api_key="sk-test")
        embedding = await llm.embed("Hello world")

        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_empty_data(self) -> None:
        """Test embedding with empty data."""
        respx.post("https://api.openai.com/v1/embeddings").mock(return_value=httpx.Response(200, json={"data": []}))

        llm = OpenAILLM(api_key="sk-test")
        embedding = await llm.embed("Hello world")

        assert embedding == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_connection_error(self) -> None:
        """Test embedding with connection error."""
        respx.post("https://api.openai.com/v1/embeddings").mock(side_effect=httpx.ConnectError("Connection refused"))

        llm = OpenAILLM(api_key="sk-test")
        with pytest.raises(LLMConnectionError, match="Failed to connect"):
            await llm.embed("Hello world")

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_api_error(self) -> None:
        """Test embedding with API error."""
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(400, json={"error": {"message": "Invalid input"}})
        )

        llm = OpenAILLM(api_key="sk-test")
        with pytest.raises(LLMConnectionError, match="API error: 400"):
            await llm.embed("Hello world")


# ============================================================================
# is_available Tests
# ============================================================================


class TestOpenAILLMIsAvailable:
    """Tests for availability check."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_true(self) -> None:
        """Test availability when API is reachable."""
        respx.get("https://api.openai.com/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))

        llm = OpenAILLM(api_key="sk-test")
        assert await llm.is_available() is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_error(self) -> None:
        """Test availability when API returns error."""
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )

        llm = OpenAILLM(api_key="sk-invalid")
        assert await llm.is_available() is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_connection_error(self) -> None:
        """Test availability when connection fails."""
        respx.get("https://api.openai.com/v1/models").mock(side_effect=httpx.ConnectError("Connection refused"))

        llm = OpenAILLM(api_key="sk-test")
        assert await llm.is_available() is False


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestOpenAILLMContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_context_manager_reuses_client(self) -> None:
        """Test that context manager reuses HTTP client."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Response"}}]})
        )

        async with OpenAILLM(api_key="sk-test") as llm:
            assert llm._client is not None
            await llm.generate("Test 1")
            await llm.generate("Test 2")
            # Client should still be the same
            assert llm._client is not None

        # Client should be closed after exiting
        assert llm._client is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_standalone_creates_new_client(self) -> None:
        """Test that standalone usage creates new client per call."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Response"}}]})
        )

        llm = OpenAILLM(api_key="sk-test")
        assert llm._client is None
        await llm.generate("Test")
        # Client should still be None (temporary client was used)
        assert llm._client is None


# ============================================================================
# Headers Tests
# ============================================================================


class TestOpenAILLMHeaders:
    """Tests for request headers."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_authorization_header(self) -> None:
        """Test that Authorization header is set correctly."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Response"}}]})
        )

        llm = OpenAILLM(api_key="sk-test-key-123")
        await llm.generate("Test")

        assert route.called
        request = route.calls[0].request
        assert request.headers["Authorization"] == "Bearer sk-test-key-123"
        assert request.headers["Content-Type"] == "application/json"
