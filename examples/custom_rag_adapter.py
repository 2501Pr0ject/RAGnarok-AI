"""Example of wrapping a custom RAG implementation for evaluation.

This example demonstrates how to implement the RAGProtocol for both
asynchronous and synchronous RAG pipelines.
"""

import asyncio
from typing import Any

from ragnarok_ai import evaluate
from ragnarok_ai.core.types import Document, Query, RAGResponse, TestSet


# 1. Implementation for an Async RAG Pipeline
class MyAsyncRAG:
    """A mock asynchronous RAG pipeline."""

    async def query(self, question: str) -> RAGResponse:
        # Simulate some async work (e.g., database search, LLM call)
        await asyncio.sleep(0.1)

        # In a real implementation, you would:
        # 1. Retrieve documents from your vector store
        # 2. Generate an answer using an LLM
        return RAGResponse(
            answer=f"The answer to '{question}' is Paris.",
            retrieved_docs=[
                Document(id="doc_1", content="Paris is the capital of France."),
                Document(id="doc_2", content="France is a country in Europe."),
            ],
            metadata={"latency_ms": 105, "model": "gpt-4"},
        )


# 2. Implementation for a Sync RAG Pipeline
# The protocol requires 'async def query', but you can easily wrap sync code.
class MySyncRAG:
    """A mock synchronous RAG pipeline wrapped for ragnarok-ai."""

    def _run_sync_query(self, question: str) -> RAGResponse:
        # Your existing synchronous logic
        return RAGResponse(
            answer="Mars is the red planet.",
            retrieved_docs=[
                Document(id="doc_3", content="Mars is known as the Red Planet."),
            ],
        )

    async def query(self, question: str) -> RAGResponse:
        # Offload sync work to a thread to keep the event loop free
        return await asyncio.to_thread(self._run_sync_query, question)


async def main():
    # Define a test set
    testset = TestSet(
        queries=[
            Query(
                text="What is the capital of France?",
                ground_truth_docs=["doc_1"],
            ),
            Query(
                text="Which planet is red?",
                ground_truth_docs=["doc_3"],
            ),
        ]
    )

    # Evaluate the async RAG
    print("Evaluating Async RAG...")
    async_rag = MyAsyncRAG()
    results = await evaluate(async_rag, testset)

    summary = results.summary()
    print(f"Async Results: {summary}")

    # Evaluate the sync RAG
    print("\nEvaluating Sync RAG...")
    sync_rag = MySyncRAG()
    results = await evaluate(sync_rag, testset)

    summary = results.summary()
    print(f"Sync Results: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
