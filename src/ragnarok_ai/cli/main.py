"""Main CLI entry point for ragnarok-ai.

This module defines the Typer application and all CLI commands.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from ragnarok_ai import __version__

if TYPE_CHECKING:
    from ragnarok_ai.core.types import Document

# Create the main Typer app
app = typer.Typer(
    name="ragnarok",
    help="ragnarok-ai: Local-first RAG evaluation framework for LLM applications.",
    add_completion=False,
    no_args_is_help=True,
)

# Global state for options
state: dict[str, bool] = {
    "json": False,
    "no_color": False,
}


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"ragnarok-ai v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[  # noqa: ARG001
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output results in JSON format.",
        ),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option(
            "--no-color",
            help="Disable colored output.",
        ),
    ] = False,
) -> None:
    """ragnarok-ai: Local-first RAG evaluation framework.

    Evaluate, benchmark, and monitor your RAG pipelines — 100% locally.
    """
    state["json"] = json_output
    state["no_color"] = no_color


@app.command()
def version() -> None:
    """Show the current version."""
    typer.echo(f"ragnarok-ai v{__version__}")


@app.command()
def evaluate(
    demo: Annotated[
        bool,
        typer.Option(
            "--demo",
            help="Run a demo evaluation with the NovaTech example dataset.",
        ),
    ] = False,
    testset: Annotated[
        str | None,
        typer.Option(
            "--testset",
            "-t",
            help="Path to testset JSON file.",
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path for results.",
        ),
    ] = None,
    fail_under: Annotated[
        float | None,
        typer.Option(
            "--fail-under",
            help="Fail if average score is below this threshold (0.0-1.0).",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-n",
            help="Limit number of queries to evaluate (useful for quick tests).",
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for reproducible demo results.",
        ),
    ] = 42,
) -> None:
    """Evaluate a RAG pipeline against a test set.

    Examples:
        ragnarok evaluate --demo
        ragnarok evaluate --demo --limit 5
        ragnarok evaluate --demo --output results.json
        ragnarok evaluate --demo --fail-under 0.7
        ragnarok evaluate --demo --json
    """
    if not demo and not testset:
        typer.echo("Error: Either --demo or --testset is required.", err=True)
        typer.echo("Run 'ragnarok evaluate --help' for usage.", err=True)
        raise typer.Exit(1)

    if demo:
        _run_demo_evaluation(output=output, fail_under=fail_under, limit=limit, seed=seed)
    else:
        typer.echo(f"Evaluating with testset: {testset}")
        typer.echo("Custom testset evaluation coming in next release.")
        typer.echo("For now, use --demo to see the evaluation flow.")


def _run_demo_evaluation(
    output: str | None = None,
    fail_under: float | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> None:
    """Run demo evaluation with NovaTech dataset and realistic MockRAG."""
    from ragnarok_ai.core.types import RAGResponse
    from ragnarok_ai.data import load_example_dataset
    from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

    random.seed(seed)

    # Load dataset
    if not state["json"]:
        typer.echo()
        typer.echo("  RAGnarok-AI Demo Evaluation")
        typer.echo("  " + "=" * 40)
        typer.echo()
        typer.echo("  Loading NovaTech example dataset...")

    dataset = load_example_dataset("novatech")
    testset = dataset.to_testset()

    # Apply limit if specified
    if limit and limit < len(testset.queries):
        testset.queries = testset.queries[:limit]

    if not state["json"]:
        typer.echo(f"  Documents: {len(dataset.documents)}")
        typer.echo(f"  Queries: {len(testset.queries)}")
        typer.echo()
        typer.echo("  Simulating RAG retrieval (realistic noise)...")
        typer.echo()

    class RealisticMockRAG:
        """Mock RAG that simulates realistic retrieval with noise.

        - Recall ~75%: misses some relevant docs
        - Precision ~70%: includes some irrelevant docs
        - Variable performance per query
        """

        def __init__(self, docs: list[Document]) -> None:
            self.docs_by_id = {d.id: d for d in docs}
            self.all_doc_ids = list(self.docs_by_id.keys())

        async def query(self, question: str) -> RAGResponse:
            # Find the query in testset to get ground truth
            for q in testset.queries:
                if q.text == question:
                    ground_truth_ids = q.ground_truth_docs
                    retrieved_ids: list[str] = []

                    # Simulate recall ~75%: keep most relevant docs but miss some
                    for doc_id in ground_truth_ids:
                        if random.random() < 0.75:  # 75% chance to retrieve
                            retrieved_ids.append(doc_id)

                    # Simulate precision degradation: add some irrelevant docs
                    num_noise = random.randint(1, 3)
                    noise_candidates = [
                        d for d in self.all_doc_ids if d not in ground_truth_ids and d not in retrieved_ids
                    ]
                    if noise_candidates:
                        noise_docs = random.sample(noise_candidates, min(num_noise, len(noise_candidates)))
                        retrieved_ids.extend(noise_docs)

                    # Shuffle to simulate ranking imperfection
                    random.shuffle(retrieved_ids)

                    # Build response
                    retrieved_docs = [self.docs_by_id[doc_id] for doc_id in retrieved_ids if doc_id in self.docs_by_id]

                    return RAGResponse(
                        answer=q.expected_answer or "Answer based on retrieved context.",
                        retrieved_docs=retrieved_docs,
                    )

            return RAGResponse(answer="No answer found.", retrieved_docs=[])

    mock_rag = RealisticMockRAG(dataset.documents)

    # Run evaluation
    async def run_eval() -> tuple[list[RetrievalMetrics], list[str]]:
        from ragnarok_ai.evaluators import mrr, ndcg_at_k, precision_at_k, recall_at_k

        metrics_list: list[RetrievalMetrics] = []
        responses: list[str] = []

        for i, query in enumerate(testset.queries):
            response = await mock_rag.query(query.text)
            responses.append(response.answer)

            # Compute retrieval metrics
            retrieved_ids = [d.id for d in response.retrieved_docs]
            relevant_ids = query.ground_truth_docs

            p = precision_at_k(retrieved_ids, relevant_ids, k=10)
            r = recall_at_k(retrieved_ids, relevant_ids, k=10)
            m = mrr(retrieved_ids, relevant_ids)
            n = ndcg_at_k(retrieved_ids, relevant_ids, k=10)

            metrics = RetrievalMetrics(precision=p, recall=r, mrr=m, ndcg=n, k=10)
            metrics_list.append(metrics)

            if not state["json"]:
                # Show status based on precision
                if p >= 0.7:
                    status = "✓"
                elif p >= 0.4:
                    status = "○"
                else:
                    status = "✗"
                typer.echo(f"    [{status}] Query {i + 1:2d}/{len(testset.queries)}: P={p:.2f} R={r:.2f} MRR={m:.2f}")

        return metrics_list, responses

    metrics_list, _responses = asyncio.run(run_eval())

    # Aggregate metrics
    avg_precision = sum(m.precision for m in metrics_list) / len(metrics_list)
    avg_recall = sum(m.recall for m in metrics_list) / len(metrics_list)
    avg_mrr = sum(m.mrr for m in metrics_list) / len(metrics_list)
    avg_ndcg = sum(m.ndcg for m in metrics_list) / len(metrics_list)
    avg_score = (avg_precision + avg_recall + avg_mrr + avg_ndcg) / 4

    results = {
        "dataset": "novatech",
        "queries_evaluated": len(testset.queries),
        "seed": seed,
        "metrics": {
            "precision@10": round(avg_precision, 4),
            "recall@10": round(avg_recall, 4),
            "mrr": round(avg_mrr, 4),
            "ndcg@10": round(avg_ndcg, 4),
            "average": round(avg_score, 4),
        },
        "status": "pass",
    }

    # Check fail_under threshold
    exit_code = 0
    if fail_under is not None and avg_score < fail_under:
        results["status"] = "fail"
        results["fail_reason"] = f"Average score {avg_score:.4f} < threshold {fail_under}"
        exit_code = 1

    # Output results
    if state["json"]:
        typer.echo(json.dumps(results, indent=2))
    else:
        typer.echo()
        typer.echo("  " + "-" * 40)
        typer.echo("  Results Summary")
        typer.echo("  " + "-" * 40)
        typer.echo(f"    Precision@10:  {avg_precision:.4f}")
        typer.echo(f"    Recall@10:     {avg_recall:.4f}")
        typer.echo(f"    MRR:           {avg_mrr:.4f}")
        typer.echo(f"    NDCG@10:       {avg_ndcg:.4f}")
        typer.echo("  " + "-" * 40)
        typer.echo(f"    Average:       {avg_score:.4f}")

        if fail_under is not None:
            if exit_code == 0:
                typer.echo(f"    Threshold:     {fail_under} → PASS ✓")
            else:
                typer.echo(f"    Threshold:     {fail_under} → FAIL ✗")

        typer.echo()

    # Save to file if requested
    if output:
        output_path = Path(output)
        output_path.write_text(json.dumps(results, indent=2))
        if not state["json"]:
            typer.echo(f"  Results saved to: {output}")
            typer.echo()

    sys.exit(exit_code)


@app.command()
def generate(
    docs: Annotated[
        str | None,
        typer.Option(
            "--docs",
            "-d",
            help="Path to documents (JSON file or directory with .txt/.md files).",
        ),
    ] = None,
    demo: Annotated[
        bool,
        typer.Option(
            "--demo",
            help="Use NovaTech example dataset as document source.",
        ),
    ] = False,
    num_questions: Annotated[
        int,
        typer.Option(
            "--num",
            "-n",
            help="Number of questions to generate.",
        ),
    ] = 10,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Output file path for generated test set.",
        ),
    ] = "testset.json",
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Ollama model for generation.",
        ),
    ] = "mistral",
    validate: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help="Validate generated questions for quality.",
        ),
    ] = False,
    ollama_url: Annotated[
        str,
        typer.Option(
            "--ollama-url",
            help="Ollama API base URL.",
        ),
    ] = "http://localhost:11434",
) -> None:
    """Generate a synthetic test set from documents using Ollama.

    Requires Ollama running locally with a model installed.

    Examples:
        ragnarok generate --demo --num 10
        ragnarok generate --docs ./knowledge/ --num 50 --output testset.json
        ragnarok generate --docs docs.json --model llama3 --num 20
    """
    if not docs and not demo:
        typer.echo("Error: Either --docs or --demo is required.", err=True)
        typer.echo("Run 'ragnarok generate --help' for usage.", err=True)
        raise typer.Exit(1)

    _run_generation(
        docs_path=docs,
        demo=demo,
        num_questions=num_questions,
        output=output,
        model=model,
        validate=validate,
        ollama_url=ollama_url,
    )


def _run_generation(
    docs_path: str | None,
    demo: bool,
    num_questions: int,
    output: str,
    model: str,
    validate: bool,
    ollama_url: str,
) -> None:
    """Run test set generation."""
    from ragnarok_ai.generators import SyntheticQuestionGenerator, save_testset

    # Load documents
    if demo:
        from ragnarok_ai.data import load_example_dataset

        if not state["json"]:
            typer.echo()
            typer.echo("  RAGnarok-AI Test Generation")
            typer.echo("  " + "=" * 40)
            typer.echo()
            typer.echo("  Loading NovaTech example dataset...")

        dataset = load_example_dataset("novatech")
        documents = dataset.documents
    else:
        documents = _load_documents(docs_path)

    if not documents:
        typer.echo("Error: No documents found.", err=True)
        raise typer.Exit(1)

    if not state["json"]:
        typer.echo(f"  Documents loaded: {len(documents)}")
        typer.echo(f"  Model: {model}")
        typer.echo(f"  Target questions: {num_questions}")
        typer.echo()
        typer.echo("  Connecting to Ollama...")

    # Check Ollama availability and generate
    async def run_gen() -> None:
        from ragnarok_ai.adapters.llm import OllamaLLM
        from ragnarok_ai.core.exceptions import LLMConnectionError

        try:
            async with OllamaLLM(base_url=ollama_url, model=model) as llm:
                if not await llm.is_available():
                    typer.echo(f"Error: Cannot connect to Ollama at {ollama_url}", err=True)
                    typer.echo("Make sure Ollama is running: ollama serve", err=True)
                    raise typer.Exit(1)

                if not state["json"]:
                    typer.echo("  Ollama connected.")
                    typer.echo()
                    typer.echo("  Generating questions (this may take a while)...")
                    typer.echo()

                generator = SyntheticQuestionGenerator(llm)
                testset = await generator.generate(
                    documents=documents,
                    num_questions=num_questions,
                    validate=validate,
                )

                # Save testset
                save_testset(testset, output)

                # Output results
                result = {
                    "status": "success",
                    "output": output,
                    "questions_generated": len(testset.queries),
                    "source_documents": len(documents),
                    "model": model,
                    "validated": validate,
                }

                if state["json"]:
                    typer.echo(json.dumps(result, indent=2))
                else:
                    typer.echo(f"  Generated {len(testset.queries)} questions")
                    typer.echo(f"  Saved to: {output}")
                    typer.echo()

        except LLMConnectionError as e:
            if state["json"]:
                typer.echo(json.dumps({"status": "error", "error": str(e)}))
            else:
                typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None

    asyncio.run(run_gen())


def _load_documents(path: str | None) -> list[Document]:
    """Load documents from a JSON file or directory."""
    from ragnarok_ai.core.types import Document

    if not path:
        return []

    p = Path(path)

    if not p.exists():
        typer.echo(f"Error: Path not found: {path}", err=True)
        raise typer.Exit(1)

    documents: list[Document] = []

    if p.is_file() and p.suffix == ".json":
        # Load from JSON file
        data = json.loads(p.read_text())
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    doc_id = item.get("id", f"doc_{i}")
                    content = item.get("content", item.get("text", ""))
                    metadata = item.get("metadata", {})
                    if content:
                        documents.append(Document(id=doc_id, content=content, metadata=metadata))
                elif isinstance(item, str):
                    documents.append(Document(id=f"doc_{i}", content=item))
    elif p.is_dir():
        # Load from directory
        for file_path in sorted(p.glob("**/*.txt")) + sorted(p.glob("**/*.md")):
            content = file_path.read_text()
            if content.strip():
                doc_id = file_path.stem
                documents.append(
                    Document(
                        id=doc_id,
                        content=content,
                        metadata={"source": str(file_path)},
                    )
                )
    else:
        typer.echo(f"Error: Unsupported file type: {path}", err=True)
        typer.echo("Use a .json file or a directory with .txt/.md files.", err=True)
        raise typer.Exit(1)

    return documents


@app.command()
def benchmark(
    configs: Annotated[  # noqa: ARG001
        str | None,
        typer.Option(
            "--configs",
            "-c",
            help="Path to benchmark configurations.",
        ),
    ] = None,
    output: Annotated[  # noqa: ARG001
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path for benchmark results.",
        ),
    ] = None,
) -> None:
    """Benchmark multiple RAG configurations.

    Planned for v1.1.

    Example:
        ragnarok benchmark --configs benchmark.yaml --output comparison.html
    """
    if state["json"]:
        typer.echo('{"status": "planned", "version": "v1.1"}')
    else:
        typer.echo()
        typer.echo("  [Planned for v1.1]")
        typer.echo()
        typer.echo("  Use the Python API for benchmarking:")
        typer.echo()
        typer.echo("    from ragnarok_ai import compare")
        typer.echo("    results = await compare([config1, config2], testset)")
        typer.echo()


if __name__ == "__main__":
    app()
