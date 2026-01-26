"""Unit tests for Anthropic LLM adapter."""

from __future__ import annotations

import os
from unittest.mock import patch

import httpx
import pytest
import respx

from ragnarok_ai.adapters.llm.anthropic import AnthropicLLM
from ragnarok_ai.core.exceptions import LLMConnectionError

# ============================================================================
# Initialization Tests
# ============================================================================


class TestAnthropicLLMInit:
    """Tests for AnthropicLLM initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        llm = AnthropicLLM(api_key="sk-ant-test-key")
        assert llm.api_key == "sk-ant-test-key"
        assert llm.model == "claude-sonnet-4-20250514"
        assert llm.max_tokens == 1024
        assert llm.is_local is False

    def test_init_with_env_var(self) -> None:
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env-key"}):
            llm = AnthropicLLM()
            assert llm.api_key == "sk-ant-env-key"

    def test_init_without_api_key_raises(self) -> None:
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="API key required"):
                AnthropicLLM()

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        llm = AnthropicLLM(
            api_key="sk-ant-test",
            base_url="https://custom.anthropic.com",
            model="claude-3-opus-20240229",
            max_tokens=2048,
            timeout=120.0,
        )
        assert llm.base_url == "https://custom.anthropic.com"
        assert llm.model == "claude-3-opus-20240229"
        assert llm.max_tokens == 2048
        assert llm.timeout == 120.0

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from base_url."""
        llm = AnthropicLLM(api_key="sk-ant-test", base_url="https://api.anthropic.com/")
        assert llm.base_url == "https://api.anthropic.com"


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestAnthropicLLMProtocol:
    """Tests for is_local property."""

    def test_is_local_false(self) -> None:
        """Test that is_local is False for cloud adapter."""
        llm = AnthropicLLM(api_key="sk-ant-test")
        assert llm.is_local is False


# ============================================================================
# Generate Tests
# ============================================================================


class TestAnthropicLLMGenerate:
    """Tests for text generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self) -> None:
        """Test successful text generation."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "text", "text": "RAG combines retrieval and generation."}
                    ]
                },
            )
        )

        llm = AnthropicLLM(api_key="sk-ant-test")
        response = await llm.generate("What is RAG?")

        assert response == "RAG combines retrieval and generation."

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_multiple_content_blocks(self) -> None:
        """Test generation with multiple content blocks."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "text", "text": "First part. "},
                        {"type": "text", "text": "Second part."},
                    ]
                },
            )
        )

        llm = AnthropicLLM(api_key="sk-ant-test")
        response = await llm.generate("Test")

        assert response == "First part. Second part."

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_empty_content(self) -> None:
        """Test generation with empty content."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={"content": []})
        )

        llm = AnthropicLLM(api_key="sk-ant-test")
        response = await llm.generate("What is RAG?")

        assert response == ""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_connection_error(self) -> None:
        """Test generation with connection error."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        llm = AnthropicLLM(api_key="sk-ant-test")
        with pytest.raises(LLMConnectionError, match="Failed to connect"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_timeout_error(self) -> None:
        """Test generation with timeout."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        llm = AnthropicLLM(api_key="sk-ant-test")
        with pytest.raises(LLMConnectionError, match="timed out"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_api_error(self) -> None:
        """Test generation with API error."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
        )

        llm = AnthropicLLM(api_key="sk-ant-invalid")
        with pytest.raises(LLMConnectionError, match="API error: 401"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_rate_limit(self) -> None:
        """Test generation with rate limit error."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
        )

        llm = AnthropicLLM(api_key="sk-ant-test")
        with pytest.raises(LLMConnectionError, match="API error: 429"):
            await llm.generate("What is RAG?")


# ============================================================================
# Embed Tests
# ============================================================================


class TestAnthropicLLMEmbed:
    """Tests for embedding (not supported)."""

    @pytest.mark.asyncio
    async def test_embed_raises_not_implemented(self) -> None:
        """Test that embed raises NotImplementedError."""
        llm = AnthropicLLM(api_key="sk-ant-test")
        with pytest.raises(NotImplementedError, match="does not provide an embedding API"):
            await llm.embed("Hello world")


# ============================================================================
# is_available Tests
# ============================================================================


class TestAnthropicLLMIsAvailable:
    """Tests for availability check."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_true(self) -> None:
        """Test availability when API is reachable."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200, json={"content": [{"type": "text", "text": "Hi"}]}
            )
        )

        llm = AnthropicLLM(api_key="sk-ant-test")
        assert await llm.is_available() is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_error(self) -> None:
        """Test availability when API returns error."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )

        llm = AnthropicLLM(api_key="sk-ant-invalid")
        assert await llm.is_available() is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_connection_error(self) -> None:
        """Test availability when connection fails."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        llm = AnthropicLLM(api_key="sk-ant-test")
        assert await llm.is_available() is False


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestAnthropicLLMContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_context_manager_reuses_client(self) -> None:
        """Test that context manager reuses HTTP client."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200, json={"content": [{"type": "text", "text": "Response"}]}
            )
        )

        async with AnthropicLLM(api_key="sk-ant-test") as llm:
            assert llm._client is not None
            await llm.generate("Test 1")
            await llm.generate("Test 2")
            assert llm._client is not None

        assert llm._client is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_standalone_creates_new_client(self) -> None:
        """Test that standalone usage creates new client per call."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200, json={"content": [{"type": "text", "text": "Response"}]}
            )
        )

        llm = AnthropicLLM(api_key="sk-ant-test")
        assert llm._client is None
        await llm.generate("Test")
        assert llm._client is None


# ============================================================================
# Headers Tests
# ============================================================================


class TestAnthropicLLMHeaders:
    """Tests for request headers."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_headers(self) -> None:
        """Test that headers are set correctly."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200, json={"content": [{"type": "text", "text": "Response"}]}
            )
        )

        llm = AnthropicLLM(api_key="sk-ant-test-key-123")
        await llm.generate("Test")

        assert route.called
        request = route.calls[0].request
        assert request.headers["x-api-key"] == "sk-ant-test-key-123"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert request.headers["Content-Type"] == "application/json"
