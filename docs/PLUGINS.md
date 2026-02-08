# RAGnarok-AI Plugin System

RAGnarok-AI supports a plugin architecture based on Python entry points. This allows third-party packages to register adapters for LLM providers, vector stores, RAG frameworks, and evaluators without modifying the core codebase.

## Overview

The plugin system provides:

- **Dynamic Discovery**: Plugins are discovered at runtime via Python entry points
- **Type Classification**: Each adapter is classified as LOCAL or CLOUD
- **Central Registry**: A singleton registry provides access to all available adapters
- **CLI Integration**: The `ragnarok plugins` command lists available plugins

## Using the Plugin Registry

```python
from ragnarok_ai.plugins import PluginRegistry

# Get the singleton registry
registry = PluginRegistry.get()

# Discover all plugins (builtin + external)
registry.discover()

# List all LLM adapters
for info in registry.list_adapters(adapter_type="llm"):
    print(f"{info.name}: local={info.is_local}, builtin={info.is_builtin}")

# Get a specific adapter class
OllamaLLM = registry.get_adapter("ollama")
llm = OllamaLLM(model="mistral")

# Filter adapters
local_llms = registry.list_adapters(adapter_type="llm", local_only=True)
external_plugins = registry.list_adapters(external_only=True)
```

## CLI Commands

```bash
# List all available plugins
ragnarok plugins --list

# Filter by type
ragnarok plugins --list --type llm

# Show only local adapters
ragnarok plugins --list --local

# Get info about a specific plugin
ragnarok plugins --info ollama

# JSON output for scripting
ragnarok --json plugins --list
```

## Creating a Plugin Package

### 1. Package Structure

```
ragnarok-ai-groq/
├── pyproject.toml
├── src/
│   └── ragnarok_ai_groq/
│       ├── __init__.py
│       └── adapter.py
└── tests/
    └── test_adapter.py
```

### 2. Entry Points Configuration

In your `pyproject.toml`:

```toml
[project]
name = "ragnarok-ai-groq"
version = "0.1.0"
dependencies = [
    "ragnarok-ai>=1.2.0",
    "groq>=0.4.0",
]

[project.entry-points."ragnarok_ai.adapters.llm"]
groq = "ragnarok_ai_groq:GroqLLM"
```

### 3. Implementing the Adapter

Your adapter must implement the appropriate protocol. For LLM adapters:

```python
# src/ragnarok_ai_groq/adapter.py
from __future__ import annotations

import groq


class GroqLLM:
    """Groq LLM adapter for RAGnarok-AI."""

    # Required: indicates this is a cloud service
    is_local: bool = False

    def __init__(
        self,
        model: str = "mixtral-8x7b-32768",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.client = groq.Groq(api_key=api_key)

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings (Groq doesn't support this)."""
        raise NotImplementedError("Groq does not support embeddings")
```

```python
# src/ragnarok_ai_groq/__init__.py
from ragnarok_ai_groq.adapter import GroqLLM

__all__ = ["GroqLLM"]
```

### 4. Testing Your Plugin

```python
# tests/test_adapter.py
import pytest
from ragnarok_ai_groq import GroqLLM


def test_groq_is_cloud():
    """Verify adapter is classified as cloud."""
    assert GroqLLM.is_local is False


@pytest.mark.asyncio
async def test_groq_generate():
    """Test generation (requires API key)."""
    llm = GroqLLM()
    response = await llm.generate("Say hello")
    assert isinstance(response, str)
    assert len(response) > 0
```

## Entry Point Namespaces

RAGnarok-AI defines four entry point namespaces:

| Namespace | Protocol | Example |
|-----------|----------|---------|
| `ragnarok_ai.adapters.llm` | `LLMProtocol` | OpenAI, Anthropic, Groq |
| `ragnarok_ai.adapters.vectorstore` | `VectorStoreProtocol` | Pinecone, Weaviate |
| `ragnarok_ai.adapters.framework` | `RAGProtocol` | Custom RAG frameworks |
| `ragnarok_ai.adapters.evaluator` | `EvaluatorProtocol` | Custom evaluators |

## Protocol Requirements

### LLMProtocol

```python
class LLMProtocol(Protocol):
    is_local: bool

    async def generate(self, prompt: str) -> str: ...
    async def embed(self, text: str) -> list[float]: ...
```

### VectorStoreProtocol

```python
class VectorStoreProtocol(Protocol):
    is_local: bool

    async def search(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[tuple[Document, float]]: ...

    async def add(self, documents: list[Document]) -> None: ...
```

### EvaluatorProtocol

```python
class EvaluatorProtocol(Protocol):
    async def evaluate(
        self,
        response: str,
        context: str,
        query: str | None = None,
    ) -> float: ...
```

## Best Practices

1. **Set `is_local` correctly**: This affects privacy classification
2. **Handle missing dependencies gracefully**: Check imports at runtime
3. **Document API requirements**: Specify required API keys, endpoints
4. **Follow semantic versioning**: Pin ragnarok-ai version requirements
5. **Include tests**: Test protocol compliance and basic functionality

## Troubleshooting

### Plugin Not Discovered

- Ensure the package is installed (`pip install ragnarok-ai-groq`)
- Check entry point syntax in `pyproject.toml`
- Run `ragnarok plugins --list` to see what's discovered

### Import Errors

- Check all dependencies are installed
- Verify the module path in the entry point is correct
- Test the import manually: `python -c "from ragnarok_ai_groq import GroqLLM"`

### Protocol Compliance

The registry doesn't validate protocol compliance at discovery time.
Use runtime checks or type hints in your tests to ensure compatibility.
