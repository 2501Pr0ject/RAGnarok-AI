"""Unit tests for Groq LLM adapter."""

from __future__ import annotations

import os
from unittest.mock import patch

import httpx
import pytest
import respx

from ragnarok_ai.adapters.llm.groq import GroqLLM
from ragnarok_ai.core.exceptions import LLMConnectionError

# ============================================================================
# Initialization Tests
# ============================================================================


class TestGroqLLMInit:
    """Tests for GroqLLM initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        llm = GroqLLM(api_key="gsk_test_key")
        assert llm.api_key == "gsk_test_key"
        assert llm.model == "llama-3.1-70b-versatile"
        assert llm.is_local is False

    def test_init_with_env_var(self) -> None:
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_env_key"}):
            llm = GroqLLM()
            assert llm.api_key == "gsk_env_key"

    def test_init_without_api_key_raises(self) -> None:
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GROQ_API_KEY", None)
            with pytest.raises(ValueError, match="API key required"):
                GroqLLM()

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        llm = GroqLLM(
            api_key="gsk_test",
            base_url="https://custom.groq.com/openai/v1",
            model="mixtral-8x7b-32768",
            timeout=120.0,
        )
        assert llm.base_url == "https://custom.groq.com/openai/v1"
        assert llm.model == "mixtral-8x7b-32768"
        assert llm.timeout == 120.0

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from base_url."""
        llm = GroqLLM(api_key="gsk_test", base_url="https://api.groq.com/openai/v1/")
        assert llm.base_url == "https://api.groq.com/openai/v1"


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestGroqLLMProtocol:
    """Tests for protocol compliance."""

    def test_is_local_false(self) -> None:
        """Test that is_local is False for cloud adapter."""
        llm = GroqLLM(api_key="gsk_test")
        assert llm.is_local is False


# ============================================================================
# Generate Tests
# ============================================================================


class TestGroqLLMGenerate:
    """Tests for text generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self) -> None:
        """Test successful text generation."""
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "RAG combines retrieval and generation."}}]},
            )
        )

        llm = GroqLLM(api_key="gsk_test")
        response = await llm.generate("What is RAG?")

        assert response == "RAG combines retrieval and generation."

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_empty_choices(self) -> None:
        """Test generation with empty choices."""
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )

        llm = GroqLLM(api_key="gsk_test")
        response = await llm.generate("What is RAG?")

        assert response == ""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_connection_error(self) -> None:
        """Test generation with connection error."""
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        llm = GroqLLM(api_key="gsk_test")
        with pytest.raises(LLMConnectionError, match="Failed to connect"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_timeout_error(self) -> None:
        """Test generation with timeout."""
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        llm = GroqLLM(api_key="gsk_test")
        with pytest.raises(LLMConnectionError, match="timed out"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_api_error(self) -> None:
        """Test generation with API error."""
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
        )

        llm = GroqLLM(api_key="gsk_invalid")
        with pytest.raises(LLMConnectionError, match="API error: 401"):
            await llm.generate("What is RAG?")


# ============================================================================
# Embed Tests
# ============================================================================


class TestGroqLLMEmbed:
    """Tests for embedding generation."""

    @pytest.mark.asyncio
    async def test_embed_not_implemented(self) -> None:
        """Test that embed raises NotImplementedError."""
        llm = GroqLLM(api_key="gsk_test")
        with pytest.raises(NotImplementedError, match="does not currently support embeddings"):
            await llm.embed("Hello world")


# ============================================================================
# is_available Tests
# ============================================================================


class TestGroqLLMIsAvailable:
    """Tests for availability check."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_true(self) -> None:
        """Test availability when API is reachable."""
        respx.get("https://api.groq.com/openai/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))

        llm = GroqLLM(api_key="gsk_test")
        assert await llm.is_available() is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_error(self) -> None:
        """Test availability when API returns error."""
        respx.get("https://api.groq.com/openai/v1/models").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )

        llm = GroqLLM(api_key="gsk_invalid")
        assert await llm.is_available() is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_connection_error(self) -> None:
        """Test availability when connection fails."""
        respx.get("https://api.groq.com/openai/v1/models").mock(side_effect=httpx.ConnectError("Connection refused"))

        llm = GroqLLM(api_key="gsk_test")
        assert await llm.is_available() is False


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestGroqLLMContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_context_manager_reuses_client(self) -> None:
        """Test that context manager reuses HTTP client."""
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Response"}}]})
        )

        async with GroqLLM(api_key="gsk_test") as llm:
            assert llm._client is not None
            await llm.generate("Test 1")
            await llm.generate("Test 2")
            assert llm._client is not None

        assert llm._client is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_standalone_creates_new_client(self) -> None:
        """Test that standalone usage creates new client per call."""
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Response"}}]})
        )

        llm = GroqLLM(api_key="gsk_test")
        assert llm._client is None
        await llm.generate("Test")
        assert llm._client is None
