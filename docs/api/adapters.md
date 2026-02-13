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

### GroqLLM

Fast inference for open-source models via Groq.

```python
from ragnarok_ai.adapters.llm import GroqLLM

async with GroqLLM(
    api_key: str | None = None,  # Uses GROQ_API_KEY env var
    model: str = "llama-3.1-70b-versatile",
) as llm:
    response = await llm.generate("What is Python?")
```

**Installation:**

```bash
pip install ragnarok-ai[groq]
```

---

### MistralLLM

Mistral AI models with embedding support.

```python
from ragnarok_ai.adapters.llm import MistralLLM

async with MistralLLM(
    api_key: str | None = None,  # Uses MISTRAL_API_KEY env var
    model: str = "mistral-small-latest",
) as llm:
    response = await llm.generate("What is Python?")
    embedding = await llm.embed("Hello world")
```

**Installation:**

```bash
pip install ragnarok-ai[mistral]
```

---

### TogetherLLM

Open-source models via Together AI.

```python
from ragnarok_ai.adapters.llm import TogetherLLM

async with TogetherLLM(
    api_key: str | None = None,  # Uses TOGETHER_API_KEY env var
    model: str = "meta-llama/Llama-3-70b-chat-hf",
) as llm:
    response = await llm.generate("What is Python?")
    embedding = await llm.embed("Hello world")
```

**Installation:**

```bash
pip install ragnarok-ai[together]
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

### PineconeVectorStore

Pinecone cloud vector database.

```python
from ragnarok_ai.adapters.vectorstore import PineconeVectorStore

async with PineconeVectorStore(
    api_key: str | None = None,  # Uses PINECONE_API_KEY env var
    index_name: str = "my-index",
    namespace: str = "",
) as store:
    await store.add(documents)
    results = await store.search(query_embedding, k=10)
```

**Installation:**

```bash
pip install ragnarok-ai[pinecone]
```

---

### WeaviateVectorStore

Weaviate vector database (cloud or self-hosted).

```python
from ragnarok_ai.adapters.vectorstore import WeaviateVectorStore

async with WeaviateVectorStore(
    url: str = "http://localhost:8080",
    api_key: str | None = None,  # Uses WEAVIATE_API_KEY env var
    collection_name: str = "RagnarokDocuments",
) as store:
    await store.add(documents)
    results = await store.search(query_embedding, k=10)
```

**Installation:**

```bash
pip install ragnarok-ai[weaviate]
```

---

### MilvusVectorStore

Milvus vector database (self-hosted).

```python
from ragnarok_ai.adapters.vectorstore import MilvusVectorStore

async with MilvusVectorStore(
    host: str = "localhost",
    port: int = 19530,
    collection_name: str = "ragnarok_documents",
    vector_size: int = 768,
) as store:
    await store.add(documents)
    results = await store.search(query_embedding, k=10)
```

**Installation:**

```bash
pip install ragnarok-ai[milvus]
```

---

### PgvectorVectorStore

PostgreSQL with pgvector extension.

```python
from ragnarok_ai.adapters.vectorstore import PgvectorVectorStore

async with PgvectorVectorStore(
    connection_string: str | None = None,  # Uses DATABASE_URL env var
    table_name: str = "ragnarok_documents",
    vector_size: int = 768,
) as store:
    await store.add(documents)
    results = await store.search(query_embedding, k=10)
```

**Installation:**

```bash
pip install ragnarok-ai[pgvector]
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

### HaystackAdapter

Wrap Haystack 2.x pipelines.

```python
from ragnarok_ai.adapters.frameworks import HaystackAdapter
from haystack import Pipeline

pipeline = Pipeline()
pipeline.add_component("retriever", retriever)
pipeline.add_component("generator", generator)
pipeline.connect("retriever", "generator")

adapter = HaystackAdapter(pipeline)
response = await adapter.query("What is Python?")
```

**Installation:**

```bash
pip install ragnarok-ai[haystack]
```

---

### SemanticKernelAdapter

Wrap Microsoft Semantic Kernel functions.

```python
from ragnarok_ai.adapters.frameworks import SemanticKernelAdapter
from semantic_kernel import Kernel

kernel = Kernel()
kernel.add_plugin(rag_plugin, "rag")

adapter = SemanticKernelAdapter(
    kernel,
    function_name="answer_question",
    plugin_name="rag",
)
response = await adapter.query("What is Python?")
```

**Installation:**

```bash
pip install ragnarok-ai[semantic-kernel]
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
| GroqLLM | Cloud | Fast inference for open-source models |
| MistralLLM | Cloud | Mistral AI models |
| TogetherLLM | Cloud | Open-source models via Together AI |
| QdrantVectorStore | Local | Self-hosted |
| ChromaVectorStore | Local | Local or persistent |
| FAISSVectorStore | Local | Pure local, no server |
| PineconeVectorStore | Cloud | Managed cloud service |
| WeaviateVectorStore | Cloud | Cloud or self-hosted |
| MilvusVectorStore | Local | Self-hosted |
| PgvectorVectorStore | Local | PostgreSQL extension |

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
