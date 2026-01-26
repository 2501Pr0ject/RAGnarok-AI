# Contributing Adapters to ragnarok-ai

This guide explains how to create new adapters for ragnarok-ai. Adapters enable the framework to work with different LLM providers, vector stores, and RAG frameworks.

## Table of Contents

- [Overview](#overview)
- [Protocols](#protocols)
- [Creating an LLM Adapter](#creating-an-llm-adapter)
- [Creating a Vector Store Adapter](#creating-a-vector-store-adapter)
- [Creating a Framework Adapter](#creating-a-framework-adapter)
- [Testing Your Adapter](#testing-your-adapter)
- [Submission Checklist](#submission-checklist)

---

## Overview

ragnarok-ai uses **Protocol-based design** (duck typing). Your adapter doesn't need to inherit from a base class — it just needs to implement the required methods with the correct signatures.

### Adapter Categories

| Category | Protocol | Location |
|----------|----------|----------|
| LLM providers | `LLMProtocol` | `src/ragnarok_ai/adapters/llm/` |
| Vector stores | `VectorStoreProtocol` | `src/ragnarok_ai/adapters/vectorstore/` |
| RAG frameworks | `RAGProtocol` | `src/ragnarok_ai/adapters/frameworks/` |

### Classification

Adapters are classified by where data is processed:

- **LOCAL**: Data stays on your infrastructure (Ollama, FAISS, local models)
- **CLOUD**: Data is sent to external APIs (OpenAI, Anthropic, cloud vector stores)

This classification is exposed via the `is_local` attribute on LLM and VectorStore adapters.

---

## Protocols

All protocols are defined in `src/ragnarok_ai/core/protocols.py` and are `@runtime_checkable`.

### LLMProtocol

```python
@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM providers."""

    is_local: bool

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for text."""
        ...
```

### VectorStoreProtocol

```python
@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector store providers."""

    is_local: bool

    async def search(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents."""
        ...

    async def add(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        ...
```

### RAGProtocol

```python
@runtime_checkable
class RAGProtocol(Protocol):
    """Protocol for RAG pipeline implementations."""

    async def query(self, question: str) -> RAGResponse:
        """Execute RAG pipeline and return response with retrieved docs."""
        ...
```

---

## Creating an LLM Adapter

### Step 1: Create the adapter file

Create `src/ragnarok_ai/adapters/llm/your_provider.py`:

```python
"""YourProvider LLM adapter for ragnarok-ai.

This module provides an async client for the YourProvider API,
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
DEFAULT_BASE_URL = "https://api.yourprovider.com"
DEFAULT_MODEL = "your-default-model"
DEFAULT_TIMEOUT = 60.0


class YourProviderLLM:
    """Async client for YourProvider LLM API.

    Implements LLMProtocol for text generation and embedding.

    Attributes:
        is_local: False - data is sent to external API.
        base_url: Base URL for the API.
        model: Model name for text generation.
        timeout: Request timeout in seconds.

    Example:
        >>> async with YourProviderLLM(api_key="sk-...") as llm:
        ...     response = await llm.generate("What is RAG?")
        ...     embedding = await llm.embed("Hello world")
    """

    is_local: bool = False  # Set True if runs locally

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize YourProviderLLM client.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API.
            model: Model name for text generation.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> YourProviderLLM:
        """Enter async context manager."""
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[httpx.AsyncClient]:
        """Get an HTTP client for making requests."""
        if self._client is not None:
            yield self._client
        else:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
            ) as client:
                yield client

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt.

        Returns:
            Generated text response.

        Raises:
            LLMConnectionError: If API call fails.
        """
        url = f"{self.base_url}/v1/completions"
        payload = {"model": self.model, "prompt": prompt}

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return str(data.get("text", ""))
        except httpx.ConnectError as e:
            msg = f"Failed to connect to YourProvider at {self.base_url}: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = f"YourProvider API error: {e.response.status_code}"
            raise LLMConnectionError(msg) from e

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            LLMConnectionError: If API call fails.
        """
        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": self.model, "input": text}

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return [float(x) for x in data.get("embedding", [])]
        except httpx.ConnectError as e:
            msg = f"Failed to connect to YourProvider at {self.base_url}: {e}"
            raise LLMConnectionError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = f"YourProvider API error: {e.response.status_code}"
            raise LLMConnectionError(msg) from e

    async def is_available(self) -> bool:
        """Check if the API is available."""
        try:
            async with self._get_client() as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
```

### Step 2: Export the adapter

Update `src/ragnarok_ai/adapters/llm/__init__.py`:

```python
from ragnarok_ai.adapters.llm.your_provider import YourProviderLLM

__all__ = [
    # ... existing exports
    "YourProviderLLM",
]
```

Update `src/ragnarok_ai/adapters/__init__.py`:

```python
from ragnarok_ai.adapters.llm import YourProviderLLM

# Add to classification
CLOUD_LLM_ADAPTERS: tuple[type, ...] = (AnthropicLLM, OpenAILLM, YourProviderLLM)

__all__ = [
    # ... existing exports
    "YourProviderLLM",
]
```

### Step 3: Add optional dependency

Update `pyproject.toml`:

```toml
[project.optional-dependencies]
yourprovider = [
    "httpx>=0.27",
]
all = [
    "ragnarok-ai[ollama,openai,anthropic,vllm,yourprovider,...]",
]
```

---

## Creating a Vector Store Adapter

### Step 1: Create the adapter file

Create `src/ragnarok_ai/adapters/vectorstore/your_store.py`:

```python
"""YourStore vector store adapter for ragnarok-ai."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragnarok_ai.core.exceptions import VectorStoreConnectionError
from ragnarok_ai.core.types import Document

if TYPE_CHECKING:
    from types import TracebackType


class YourStoreVectorStore:
    """Adapter for YourStore vector database.

    Implements VectorStoreProtocol for document storage and retrieval.

    Attributes:
        is_local: True if runs locally, False for cloud.

    Example:
        >>> async with YourStoreVectorStore(url="...") as store:
        ...     await store.add(documents)
        ...     results = await store.search(embedding, k=10)
    """

    is_local: bool = True  # Adjust based on your store

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "ragnarok_documents",
        vector_size: int = 768,
    ) -> None:
        """Initialize the vector store.

        Args:
            url: Connection URL.
            collection_name: Name of the collection.
            vector_size: Dimension of embedding vectors.
        """
        self.url = url.rstrip("/")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client = None

    async def __aenter__(self) -> YourStoreVectorStore:
        """Initialize connection."""
        # Initialize your client here
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close connection."""
        if self._client is not None:
            # Close your client here
            self._client = None

    async def search(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents.

        Args:
            query_embedding: Query vector.
            k: Number of results.

        Returns:
            List of (document, score) tuples, sorted by similarity.

        Raises:
            VectorStoreConnectionError: If search fails.
        """
        # Implement search logic
        # Return list of (Document, float) tuples
        pass

    async def add(self, documents: list[Document]) -> None:
        """Add documents to the store.

        Documents must have embeddings in metadata["embedding"].

        Args:
            documents: Documents to add.

        Raises:
            VectorStoreConnectionError: If add fails.
            ValueError: If documents lack embeddings.
        """
        for doc in documents:
            if "embedding" not in doc.metadata:
                msg = f"Document {doc.id} missing embedding in metadata"
                raise ValueError(msg)

        # Implement add logic
        pass

    async def is_available(self) -> bool:
        """Check if the store is available."""
        # Implement health check
        pass
```

---

## Creating a Framework Adapter

Framework adapters wrap external RAG frameworks (LangChain, LlamaIndex, DSPy) to implement `RAGProtocol`.

### Key Pattern: Document Conversion

Each framework has its own document type. Create conversion functions:

```python
from ragnarok_ai.core.types import Document

def _convert_framework_doc(fw_doc: Any) -> Document:
    """Convert a framework document to RAGnarok Document."""
    doc_id = (
        getattr(fw_doc, "id", None) or
        getattr(fw_doc, "node_id", None) or
        str(hash(fw_doc.content))
    )

    return Document(
        id=str(doc_id),
        content=fw_doc.content,  # Adjust field name
        metadata=getattr(fw_doc, "metadata", {}),
    )
```

### Step 1: Create the adapter file

Create `src/ragnarok_ai/adapters/frameworks/your_framework.py`:

```python
"""YourFramework adapter for ragnarok-ai."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ragnarok_ai.core.types import Document, RAGResponse

if TYPE_CHECKING:
    from your_framework import YourRetriever, YourPipeline


def _convert_doc(fw_doc: Any) -> Document:
    """Convert framework document to RAGnarok Document."""
    return Document(
        id=str(getattr(fw_doc, "id", hash(str(fw_doc)))),
        content=str(getattr(fw_doc, "text", fw_doc)),
        metadata=getattr(fw_doc, "metadata", {}),
    )


class YourFrameworkAdapter:
    """Adapter for YourFramework RAG pipelines.

    Implements RAGProtocol for evaluation with ragnarok-ai.

    Example:
        >>> from your_framework import create_pipeline
        >>> pipeline = create_pipeline(...)
        >>> adapter = YourFrameworkAdapter(pipeline)
        >>> results = await evaluate(adapter, testset)
    """

    def __init__(self, pipeline: YourPipeline) -> None:
        """Initialize the adapter.

        Args:
            pipeline: A YourFramework pipeline instance.
        """
        self._pipeline = pipeline

    @property
    def pipeline(self) -> YourPipeline:
        """Get the underlying pipeline."""
        return self._pipeline

    async def query(self, question: str) -> RAGResponse:
        """Execute the pipeline and return response.

        Args:
            question: The question to answer.

        Returns:
            RAGResponse with answer and retrieved documents.
        """
        # Handle sync/async
        if hasattr(self._pipeline, "arun"):
            result = await self._pipeline.arun(question)
        else:
            result = await asyncio.to_thread(self._pipeline.run, question)

        # Extract answer and documents from result
        answer = str(result.answer)
        docs = [_convert_doc(d) for d in result.documents]

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            metadata={"adapter": "your_framework"},
        )


def create_your_framework_adapter(obj: Any) -> YourFrameworkAdapter:
    """Factory function to create adapter.

    Args:
        obj: A YourFramework object.

    Returns:
        Appropriate adapter instance.

    Raises:
        ImportError: If your_framework is not installed.
    """
    try:
        import your_framework
    except ImportError as e:
        msg = "your_framework is required. Install with: pip install your_framework"
        raise ImportError(msg) from e

    return YourFrameworkAdapter(obj)
```

---

## Testing Your Adapter

### Test File Structure

Create `tests/unit/test_your_adapter.py`:

```python
"""Unit tests for YourAdapter."""

from __future__ import annotations

import pytest
import respx  # For HTTP mocking
from httpx import Response

from ragnarok_ai.adapters.llm.your_provider import (
    DEFAULT_BASE_URL,
    YourProviderLLM,
)
from ragnarok_ai.core.exceptions import LLMConnectionError


# ============================================================================
# Mock Classes (for framework adapters)
# ============================================================================

class MockDocument:
    """Mock framework document."""

    def __init__(self, content: str, doc_id: str = "mock") -> None:
        self.content = content
        self.id = doc_id
        self.metadata = {}


# ============================================================================
# Initialization Tests
# ============================================================================

class TestYourAdapterInit:
    """Tests for adapter initialization."""

    def test_default_values(self) -> None:
        """Default values are set correctly."""
        adapter = YourProviderLLM(api_key="test")

        assert adapter.base_url == DEFAULT_BASE_URL
        assert adapter.is_local is False

    def test_custom_values(self) -> None:
        """Custom values are set correctly."""
        adapter = YourProviderLLM(
            api_key="test",
            base_url="http://custom:8080",
            model="custom-model",
        )

        assert adapter.base_url == "http://custom:8080"
        assert adapter.model == "custom-model"

    def test_trailing_slash_removed(self) -> None:
        """Trailing slash is removed from base_url."""
        adapter = YourProviderLLM(api_key="test", base_url="http://api.com/")

        assert adapter.base_url == "http://api.com"


# ============================================================================
# Method Tests
# ============================================================================

class TestYourAdapterGenerate:
    """Tests for generate method."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """Successful generation returns response text."""
        respx.post(f"{DEFAULT_BASE_URL}/v1/completions").mock(
            return_value=Response(200, json={"text": "Generated response"})
        )

        adapter = YourProviderLLM(api_key="test")
        result = await adapter.generate("Test prompt")

        assert result == "Generated response"

    @respx.mock
    @pytest.mark.asyncio
    async def test_generate_connection_error(self) -> None:
        """Connection error raises LLMConnectionError."""
        import httpx
        respx.post(f"{DEFAULT_BASE_URL}/v1/completions").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        adapter = YourProviderLLM(api_key="test")

        with pytest.raises(LLMConnectionError, match="Failed to connect"):
            await adapter.generate("prompt")

    @respx.mock
    @pytest.mark.asyncio
    async def test_generate_http_error(self) -> None:
        """HTTP error raises LLMConnectionError."""
        respx.post(f"{DEFAULT_BASE_URL}/v1/completions").mock(
            return_value=Response(401, text="Unauthorized")
        )

        adapter = YourProviderLLM(api_key="test")

        with pytest.raises(LLMConnectionError, match="API error: 401"):
            await adapter.generate("prompt")


# ============================================================================
# Protocol Compliance Tests
# ============================================================================

class TestYourAdapterProtocolCompliance:
    """Tests for protocol compliance."""

    def test_implements_protocol(self) -> None:
        """Adapter implements the required protocol."""
        from ragnarok_ai.core.protocols import LLMProtocol

        adapter = YourProviderLLM(api_key="test")

        assert isinstance(adapter, LLMProtocol)

    def test_has_required_methods(self) -> None:
        """Adapter has all required methods."""
        adapter = YourProviderLLM(api_key="test")

        assert hasattr(adapter, "generate")
        assert hasattr(adapter, "embed")
        assert callable(adapter.generate)
        assert callable(adapter.embed)

    def test_has_is_local_attribute(self) -> None:
        """Adapter has is_local attribute."""
        adapter = YourProviderLLM(api_key="test")

        assert hasattr(adapter, "is_local")
        assert isinstance(adapter.is_local, bool)


# ============================================================================
# Context Manager Tests
# ============================================================================

class TestYourAdapterContextManager:
    """Tests for context manager functionality."""

    def test_client_is_none_initially(self) -> None:
        """Client is None before entering context."""
        adapter = YourProviderLLM(api_key="test")

        assert adapter._client is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self) -> None:
        """Context manager creates and closes client."""
        respx.post(f"{DEFAULT_BASE_URL}/v1/completions").mock(
            return_value=Response(200, json={"text": "ok"})
        )

        adapter = YourProviderLLM(api_key="test")

        async with adapter:
            assert adapter._client is not None

        assert adapter._client is None
```

### Running Tests

```bash
# Run your adapter tests
pytest tests/unit/test_your_adapter.py -v

# Run with coverage
pytest tests/unit/test_your_adapter.py --cov=ragnarok_ai.adapters -v

# Run all tests to ensure no regressions
pytest
```

### Test Requirements

1. **Initialization tests**: Default values, custom values, edge cases
2. **Method tests**: Success cases, error handling, edge cases
3. **Protocol compliance**: `isinstance` check, method existence
4. **Context manager**: Lifecycle, client reuse
5. **Mock external services**: Use `respx` for HTTP, create mock classes for frameworks

---

## Submission Checklist

Before submitting your adapter PR:

### Code Quality

- [ ] Type hints on all functions and methods
- [ ] Google-style docstrings with Args, Returns, Raises, Example
- [ ] No `Any` types (except where truly necessary)
- [ ] `ruff check .` passes
- [ ] `ruff format .` applied
- [ ] `mypy src/` passes

### Implementation

- [ ] Implements the correct protocol (`LLMProtocol`, `VectorStoreProtocol`, or `RAGProtocol`)
- [ ] Protocol compliance verified with `isinstance()` in tests
- [ ] `is_local` attribute set correctly (LLM/VectorStore only)
- [ ] Context manager support (`__aenter__`/`__aexit__`)
- [ ] Proper error handling with ragnarok-ai exceptions
- [ ] Async-first with sync fallback via `asyncio.to_thread()`

### Tests

- [ ] Tests in `tests/unit/test_your_adapter.py`
- [ ] All external calls mocked (no real API calls in unit tests)
- [ ] Test coverage > 80%
- [ ] Tests pass: `pytest tests/unit/test_your_adapter.py -v`

### Integration

- [ ] Exported from `adapters/llm/__init__.py` (or appropriate submodule)
- [ ] Exported from `adapters/__init__.py`
- [ ] Added to `LOCAL_*_ADAPTERS` or `CLOUD_*_ADAPTERS` tuple
- [ ] Optional dependency added to `pyproject.toml`
- [ ] Added to `[all]` extra in `pyproject.toml`
- [ ] mypy override added if third-party lib lacks types

### Documentation

- [ ] Module docstring explaining the adapter
- [ ] Class docstring with usage example
- [ ] README updated if adding a new category of adapters

---

## Questions?

- Open an [issue](https://github.com/2501Pr0ject/ragnarok-ai/issues) for questions
- Check existing adapters in `src/ragnarok_ai/adapters/` for reference
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for general contribution guidelines
