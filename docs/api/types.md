# Core Types

Reference for RAGnarok-AI core types.

---

## Import

```python
from ragnarok_ai.core.types import (
    Document,
    Query,
    TestSet,
    RAGResponse,
    RetrievalResult,
)
```

---

## Document

Represents a document in the knowledge base.

```python
from ragnarok_ai.core.types import Document

doc = Document(
    id="doc_001",
    content="Python is a programming language...",
    metadata={"source": "wikipedia", "date": "2024-01-01"},
)
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique document identifier |
| `content` | `str` | Document text content |
| `metadata` | `dict[str, Any]` | Optional metadata |

---

## Query

Represents a test query with ground truth.

```python
from ragnarok_ai.core.types import Query

query = Query(
    text="Who created Python?",
    ground_truth_docs=["doc_001", "doc_002"],
    expected_answer="Guido van Rossum created Python in 1991.",
)
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | The query text |
| `ground_truth_docs` | `list[str]` | IDs of relevant documents |
| `expected_answer` | `str \| None` | Expected answer (optional) |

---

## TestSet

Collection of queries for evaluation.

```python
from ragnarok_ai.core.types import TestSet, Query

testset = TestSet(
    name="my-testset",
    queries=[
        Query(text="Question 1", ground_truth_docs=["doc_1"]),
        Query(text="Question 2", ground_truth_docs=["doc_2", "doc_3"]),
    ],
)

print(len(testset.queries))  # 2
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str \| None` | Optional testset name |
| `queries` | `list[Query]` | List of queries |

---

## RAGResponse

Response from a RAG pipeline.

```python
from ragnarok_ai.core.types import RAGResponse, Document

response = RAGResponse(
    answer="Guido van Rossum created Python.",
    retrieved_docs=[
        Document(id="doc_1", content="Python was created by Guido..."),
        Document(id="doc_2", content="The Python language..."),
    ],
    context="Python was created by Guido van Rossum in 1991.",
)
```

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | Generated answer |
| `retrieved_docs` | `list[Document]` | Retrieved documents |
| `context` | `str \| None` | Concatenated context (optional) |

---

## RetrievalResult

Result of retrieval for evaluation.

```python
from ragnarok_ai.core.types import RetrievalResult, Query, Document

result = RetrievalResult(
    query=Query(text="Who created Python?", ground_truth_docs=["doc_1"]),
    retrieved_docs=[
        Document(id="doc_1", content="..."),
        Document(id="doc_2", content="..."),
    ],
    scores=[0.95, 0.82],
)
```

| Field | Type | Description |
|-------|------|-------------|
| `query` | `Query` | The original query |
| `retrieved_docs` | `list[Document]` | Retrieved documents |
| `scores` | `list[float] \| None` | Relevance scores |

---

## Protocols

### RAGProtocol

Interface for RAG pipelines.

```python
from ragnarok_ai.core.protocols import RAGProtocol
from ragnarok_ai.core.types import RAGResponse

class MyRAG:
    async def query(self, question: str, k: int = 5) -> RAGResponse:
        # Your implementation
        pass

# Type check
rag: RAGProtocol = MyRAG()
```

### LLMProtocol

Interface for LLM adapters.

```python
from ragnarok_ai.core.protocols import LLMProtocol

class MyLLM:
    async def generate(self, prompt: str, **kwargs) -> str:
        # Your implementation
        pass

    async def embed(self, text: str) -> list[float]:
        # Your implementation
        pass
```

### VectorStoreProtocol

Interface for vector stores.

```python
from ragnarok_ai.core.protocols import VectorStoreProtocol
from ragnarok_ai.core.types import Document

class MyVectorStore:
    async def add(self, documents: list[Document]) -> None:
        pass

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[Document]:
        pass

    async def delete(self, ids: list[str]) -> None:
        pass
```

---

## Next Steps

- [Evaluators](evaluators.md) — Metric implementations
- [Adapters](adapters.md) — LLM and vector store adapters
