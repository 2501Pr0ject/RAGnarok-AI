# Adapters

Reference for RAGnarok-AI adapters.

---

## LLM Adapters

### OllamaLLM

Local LLM via Ollama.

```python
from ragnarok_ai.adapters.llm import OllamaLLM

async with OllamaLLM(
    model: str = "mistral",
    base_url: str = "http://localhost:11434",
) as llm:
    response = await llm.generate("What is Python?")
    print(response)
```

**Methods:**

```python
async def generate(self, prompt: str, **kwargs) -> str
async def embed(self, text: str) -> list[float]
async def is_available(self) -> bool
```

**Installation:**

```bash
pip install ragnarok-ai[ollama]
```

---

### OpenAILLM

OpenAI API adapter.

```python
from ragnarok_ai.adapters.llm import OpenAILLM

async with OpenAILLM(
    model: str = "gpt-4",
    api_key: str | None = None,  # Uses OPENAI_API_KEY env var
) as llm:
    response = await llm.generate("What is Python?")
```

**Installation:**

```bash
pip install ragnarok-ai[openai]
```

---

### AnthropicLLM

Anthropic Claude adapter.

```python
from ragnarok_ai.adapters.llm import AnthropicLLM

async with AnthropicLLM(
    model: str = "claude-3-sonnet",
    api_key: str | None = None,  # Uses ANTHROPIC_API_KEY env var
) as llm:
    response = await llm.generate("What is Python?")
```

**Installation:**

```bash
pip install ragnarok-ai[anthropic]
```

---

### vLLM

Local high-performance inference.

```python
from ragnarok_ai.adapters.llm import vLLM

async with vLLM(
    model: str = "mistral-7b",
    base_url: str = "http://localhost:8000",
) as llm:
    response = await llm.generate("What is Python?")
```

**Installation:**

```bash
pip install ragnarok-ai[vllm]
```

---

## Vector Store Adapters

### QdrantVectorStore

Qdrant vector database.

```python
from ragnarok_ai.adapters.vectorstore import QdrantVectorStore

async with QdrantVectorStore(
    url: str = "http://localhost:6333",
    collection_name: str = "documents",
) as store:
    await store.add(documents)
    results = await store.search(query_embedding, top_k=10)
```

**Installation:**

```bash
pip install ragnarok-ai[qdrant]
```

---

### ChromaVectorStore

ChromaDB adapter.

```python
from ragnarok_ai.adapters.vectorstore import ChromaVectorStore

async with ChromaVectorStore(
    collection_name: str = "documents",
    persist_directory: str | None = None,
) as store:
    await store.add(documents)
    results = await store.search(query_embedding, top_k=10)
```

**Installation:**

```bash
pip install ragnarok-ai[chroma]
```

---

### FAISSVectorStore

FAISS local vector store (no server required).

```python
from ragnarok_ai.adapters.vectorstore import FAISSVectorStore

async with FAISSVectorStore(
    dimension: int = 384,
    index_type: str = "flat",  # or "hnsw"
) as store:
    await store.add(documents)
    results = await store.search(query_embedding, top_k=10)
```

**Installation:**

```bash
pip install ragnarok-ai[faiss]
```

---

## Framework Adapters

### LangChainAdapter

Wrap LangChain pipelines.

```python
from ragnarok_ai.adapters.framework import LangChainAdapter
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(...)
adapter = LangChainAdapter(chain)

response = await adapter.query("What is Python?")
```

**Installation:**

```bash
pip install ragnarok-ai[langchain]
```

---

### LangGraphAdapter

Wrap LangGraph agents.

```python
from ragnarok_ai.adapters.framework import LangGraphAdapter
from langgraph.graph import StateGraph

graph = StateGraph(...)
adapter = LangGraphAdapter(graph)

response = await adapter.query("What is Python?")
```

**Installation:**

```bash
pip install ragnarok-ai[langgraph]
```

---

### LlamaIndexAdapter

Wrap LlamaIndex query engines.

```python
from ragnarok_ai.adapters.framework import LlamaIndexAdapter
from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(...)
adapter = LlamaIndexAdapter(index.as_query_engine())

response = await adapter.query("What is Python?")
```

**Installation:**

```bash
pip install ragnarok-ai[llamaindex]
```

---

### DSPyAdapter

Wrap DSPy modules.

```python
from ragnarok_ai.adapters.framework import DSPyAdapter
import dspy

class MyRAG(dspy.Module):
    ...

adapter = DSPyAdapter(MyRAG())
response = await adapter.query("What is Python?")
```

**Installation:**

```bash
pip install ragnarok-ai[dspy]
```

---

## Local vs Cloud

All adapters are classified as local or cloud:

| Adapter | Type | Description |
|---------|------|-------------|
| OllamaLLM | Local | Runs on your machine |
| vLLM | Local | High-performance local inference |
| OpenAILLM | Cloud | Requires API key |
| AnthropicLLM | Cloud | Requires API key |
| QdrantVectorStore | Local | Self-hosted |
| ChromaVectorStore | Local | Local or persistent |
| FAISSVectorStore | Local | Pure local, no server |

List adapters by type:

```bash
ragnarok plugins --list --local
```

---

## Custom Adapters

Implement the protocol for custom adapters:

```python
from ragnarok_ai.core.protocols import LLMProtocol

class MyCustomLLM:
    is_local: ClassVar[bool] = True

    async def generate(self, prompt: str, **kwargs) -> str:
        # Your implementation
        return "response"

    async def embed(self, text: str) -> list[float]:
        # Your implementation
        return [0.1, 0.2, ...]
```

---

## Next Steps

- [Core Types](types.md) — Type reference
- [Evaluators](evaluators.md) — Metric implementations
