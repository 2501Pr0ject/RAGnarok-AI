"""Unit tests for Mistral AI LLM adapter."""

from __future__ import annotations

import os
from unittest.mock import patch

import httpx
import pytest
import respx

from ragnarok_ai.adapters.llm.mistral import MistralLLM
from ragnarok_ai.core.exceptions import LLMConnectionError


# ============================================================================
# Initialization Tests
# ============================================================================


class TestMistralLLMInit:
    """Tests for MistralLLM initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        llm = MistralLLM(api_key="test_key")
        assert llm.api_key == "test_key"
        assert llm.model == "mistral-small-latest"
        assert llm.embed_model == "mistral-embed"
        assert llm.is_local is False

    def test_init_with_env_var(self) -> None:
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "env_key"}):
            llm = MistralLLM()
            assert llm.api_key == "env_key"

    def test_init_without_api_key_raises(self) -> None:
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MISTRAL_API_KEY", None)
            with pytest.raises(ValueError, match="API key required"):
                MistralLLM()

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        llm = MistralLLM(
            api_key="test_key",
            base_url="https://custom.mistral.ai/v1",
            model="mistral-large-latest",
            embed_model="custom-embed",
            timeout=120.0,
        )
        assert llm.base_url == "https://custom.mistral.ai/v1"
        assert llm.model == "mistral-large-latest"
        assert llm.embed_model == "custom-embed"
        assert llm.timeout == 120.0

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from base_url."""
        llm = MistralLLM(api_key="test_key", base_url="https://api.mistral.ai/v1/")
        assert llm.base_url == "https://api.mistral.ai/v1"


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestMistralLLMProtocol:
    """Tests for protocol compliance."""

    def test_is_local_false(self) -> None:
        """Test that is_local is False for cloud adapter."""
        llm = MistralLLM(api_key="test_key")
        assert llm.is_local is False


# ============================================================================
# Generate Tests
# ============================================================================


class TestMistralLLMGenerate:
    """Tests for text generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self) -> None:
        """Test successful text generation."""
        respx.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "RAG combines retrieval and generation."}}]},
            )
        )

        llm = MistralLLM(api_key="test_key")
        response = await llm.generate("What is RAG?")

        assert response == "RAG combines retrieval and generation."

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_empty_choices(self) -> None:
        """Test generation with empty choices."""
        respx.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )

        llm = MistralLLM(api_key="test_key")
        response = await llm.generate("What is RAG?")

        assert response == ""

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_connection_error(self) -> None:
        """Test generation with connection error."""
        respx.post("https://api.mistral.ai/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        llm = MistralLLM(api_key="test_key")
        with pytest.raises(LLMConnectionError, match="Failed to connect"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_timeout_error(self) -> None:
        """Test generation with timeout."""
        respx.post("https://api.mistral.ai/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        llm = MistralLLM(api_key="test_key")
        with pytest.raises(LLMConnectionError, match="timed out"):
            await llm.generate("What is RAG?")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_api_error(self) -> None:
        """Test generation with API error."""
        respx.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
        )

        llm = MistralLLM(api_key="invalid_key")
        with pytest.raises(LLMConnectionError, match="API error: 401"):
            await llm.generate("What is RAG?")


# ============================================================================
# Embed Tests
# ============================================================================


class TestMistralLLMEmbed:
    """Tests for embedding generation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_success(self) -> None:
        """Test successful embedding."""
        respx.post("https://api.mistral.ai/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}]},
            )
        )

        llm = MistralLLM(api_key="test_key")
        embedding = await llm.embed("Hello world")

        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_empty_data(self) -> None:
        """Test embedding with empty data."""
        respx.post("https://api.mistral.ai/v1/embeddings").mock(return_value=httpx.Response(200, json={"data": []}))

        llm = MistralLLM(api_key="test_key")
        embedding = await llm.embed("Hello world")

        assert embedding == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_connection_error(self) -> None:
        """Test embedding with connection error."""
        respx.post("https://api.mistral.ai/v1/embeddings").mock(side_effect=httpx.ConnectError("Connection refused"))

        llm = MistralLLM(api_key="test_key")
        with pytest.raises(LLMConnectionError, match="Failed to connect"):
            await llm.embed("Hello world")

    @pytest.mark.asyncio
    @respx.mock
    async def test_embed_api_error(self) -> None:
        """Test embedding with API error."""
        respx.post("https://api.mistral.ai/v1/embeddings").mock(
            return_value=httpx.Response(400, json={"error": {"message": "Invalid input"}})
        )

        llm = MistralLLM(api_key="test_key")
        with pytest.raises(LLMConnectionError, match="API error: 400"):
            await llm.embed("Hello world")


# ============================================================================
# is_available Tests
# ============================================================================


class TestMistralLLMIsAvailable:
    """Tests for availability check."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_true(self) -> None:
        """Test availability when API is reachable."""
        respx.get("https://api.mistral.ai/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))

        llm = MistralLLM(api_key="test_key")
        assert await llm.is_available() is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_error(self) -> None:
        """Test availability when API returns error."""
        respx.get("https://api.mistral.ai/v1/models").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )

        llm = MistralLLM(api_key="invalid_key")
        assert await llm.is_available() is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_is_available_false_on_connection_error(self) -> None:
        """Test availability when connection fails."""
        respx.get("https://api.mistral.ai/v1/models").mock(side_effect=httpx.ConnectError("Connection refused"))

        llm = MistralLLM(api_key="test_key")
        assert await llm.is_available() is False


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestMistralLLMContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_context_manager_reuses_client(self) -> None:
        """Test that context manager reuses HTTP client."""
        respx.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Response"}}]})
        )

        async with MistralLLM(api_key="test_key") as llm:
            assert llm._client is not None
            await llm.generate("Test 1")
            await llm.generate("Test 2")
            assert llm._client is not None

        assert llm._client is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_standalone_creates_new_client(self) -> None:
        """Test that standalone usage creates new client per call."""
        respx.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Response"}}]})
        )

        llm = MistralLLM(api_key="test_key")
        assert llm._client is None
        await llm.generate("Test")
        assert llm._client is None
