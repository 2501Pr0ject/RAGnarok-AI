"""Main CLI entry point for ragnarok-ai.

This module defines the Typer application and all CLI commands.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from ragnarok_ai import __version__

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1  # Runtime failures, threshold not met
EXIT_BAD_INPUT = 2  # Invalid arguments, missing files

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
state: dict[str, Any] = {
    "json": False,
    "no_color": False,
    "pii_mode": "hash",  # CLI default is hash for safety
}


def json_response(
    command: str,
    status: str,
    data: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> str:
    """Create standardized JSON response envelope.

    All CLI commands should use this format for --json output:
    {
        "command": "evaluate",
        "status": "pass|fail|error|dry_run|success",
        "version": "1.1.0",
        "data": { ... },
        "warnings": [],
        "errors": []
    }
    """
    return json.dumps(
        {
            "command": command,
            "status": status,
            "version": __version__,
            "data": data or {},
            "warnings": warnings or [],
            "errors": errors or [],
        },
        indent=2,
    )


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
    json: Annotated[
        bool,
        typer.Option(
            "--json/--no-json",
            help="Output results in JSON format.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json-output",
            help="(Deprecated) Alias for --json.",
            hidden=True,
        ),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option(
            "--no-color",
            help="Disable colored output.",
        ),
    ] = False,
    pii_mode: Annotated[
        str,
        typer.Option(
            "--pii-mode",
            help="PII handling mode: hash (default), redact, or full.",
            case_sensitive=False,
        ),
    ] = "hash",
) -> None:
    """ragnarok-ai: Local-first RAG evaluation framework.

    Evaluate, benchmark, and monitor your RAG pipelines — 100% locally.
    """
    state["json"] = json or json_output
    state["no_color"] = no_color
    state["pii_mode"] = pii_mode.lower()


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
        if state["json"]:
            typer.echo(json_response("evaluate", "error", errors=["Either --demo or --testset is required."]))
        else:
            typer.echo("Error: Either --demo or --testset is required.", err=True)
            typer.echo("Run 'ragnarok evaluate --help' for usage.", err=True)
        raise typer.Exit(EXIT_BAD_INPUT)

    if demo:
        _run_demo_evaluation(output=output, fail_under=fail_under, limit=limit, seed=seed)
    else:
        if state["json"]:
            typer.echo(
                json_response(
                    "evaluate",
                    "error",
                    errors=["Custom testset evaluation coming in next release. Use --demo for now."],
                )
            )
        else:
            typer.echo(f"Evaluating with testset: {testset}")
            typer.echo("Custom testset evaluation coming in next release.")
            typer.echo("For now, use --demo to see the evaluation flow.")
        raise typer.Exit(1)


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

    data = {
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
    }

    # Check fail_under threshold
    exit_code = 0
    status = "pass"
    if fail_under is not None and avg_score < fail_under:
        status = "fail"
        data["fail_reason"] = f"Average score {avg_score:.4f} < threshold {fail_under}"
        exit_code = 1

    # Output results
    if state["json"]:
        typer.echo(json_response("evaluate", status, data=data))
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

    # Save to file if requested (use flat format for file, envelope for stdout)
    if output:
        output_path = Path(output)
        file_data = {"status": status, **data}
        output_path.write_text(json.dumps(file_data, indent=2))
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
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            "-s",
            help="Random seed for reproducibility (auto-generated if not set).",
        ),
    ] = None,
    validate: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help="Validate generated questions for quality.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be generated without executing.",
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
    Produces testset.json + manifest.json for reproducibility.

    Examples:
        ragnarok generate --demo --num 10
        ragnarok generate --docs ./knowledge/ --num 50 --output testset.json
        ragnarok generate --docs docs.json --model llama3 --seed 42
    """
    if not docs and not demo:
        if state["json"]:
            typer.echo(json_response("generate", "error", errors=["Either --docs or --demo is required."]))
        else:
            typer.echo("Error: Either --docs or --demo is required.", err=True)
            typer.echo("Run 'ragnarok generate --help' for usage.", err=True)
        raise typer.Exit(EXIT_BAD_INPUT)

    _run_generation(
        docs_path=docs,
        demo=demo,
        num_questions=num_questions,
        output=output,
        model=model,
        seed=seed,
        validate=validate,
        dry_run=dry_run,
        ollama_url=ollama_url,
    )


def _run_generation(
    docs_path: str | None,
    demo: bool,
    num_questions: int,
    output: str,
    model: str,
    seed: int | None,
    validate: bool,
    dry_run: bool,
    ollama_url: str,
) -> None:
    """Run test set generation."""
    from datetime import datetime

    from ragnarok_ai.generators import SyntheticQuestionGenerator, save_testset

    # Auto-generate seed if not provided
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)

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
        docs_source = "novatech-demo"
    else:
        documents = _load_documents(docs_path)
        docs_source = docs_path or "unknown"

    if not documents:
        typer.echo("Error: No documents found.", err=True)
        raise typer.Exit(1)

    # Build manifest
    manifest = {
        "ragnarok_version": __version__,
        "generated_at": datetime.now().isoformat(),
        "seed": seed,
        "config": {
            "model": model,
            "ollama_url": ollama_url,
            "num_questions": num_questions,
            "validate": validate,
            "docs_source": docs_source,
            "docs_count": len(documents),
        },
    }

    if not state["json"]:
        typer.echo(f"  Documents loaded: {len(documents)}")
        typer.echo(f"  Model: {model}")
        typer.echo(f"  Target questions: {num_questions}")
        typer.echo(f"  Seed: {seed}")
        typer.echo()

    # Dry run - show what would happen
    if dry_run:
        data = {
            "seed": seed,
            "output": output,
            "manifest_output": output.replace(".json", "_manifest.json"),
            "config": manifest["config"],
        }

        if state["json"]:
            typer.echo(json_response("generate", "dry_run", data=data))
        else:
            typer.echo("  [DRY RUN] Would generate:")
            typer.echo(f"    - {num_questions} questions from {len(documents)} documents")
            typer.echo(f"    - Output: {output}")
            typer.echo(f"    - Manifest: {data['manifest_output']}")
            typer.echo(f"    - Seed: {seed} (use --seed {seed} to reproduce)")
            typer.echo()
        return

    if not state["json"]:
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

                # Save manifest
                manifest_path = output.replace(".json", "_manifest.json")
                manifest["status"] = "success"
                manifest["output"] = output
                manifest["questions_generated"] = len(testset.queries)
                Path(manifest_path).write_text(json.dumps(manifest, indent=2))

                if state["json"]:
                    data = {
                        "seed": seed,
                        "output": output,
                        "manifest_output": manifest_path,
                        "questions_generated": len(testset.queries),
                        "config": manifest["config"],
                    }
                    typer.echo(json_response("generate", "success", data=data))
                else:
                    typer.echo(f"  Generated {len(testset.queries)} questions")
                    typer.echo(f"  Testset: {output}")
                    typer.echo(f"  Manifest: {manifest_path}")
                    typer.echo(f"  Reproduce with: --seed {seed}")
                    typer.echo()

        except LLMConnectionError as e:
            if state["json"]:
                typer.echo(json_response("generate", "error", errors=[str(e)]))
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
        raise typer.Exit(EXIT_BAD_INPUT)

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
        raise typer.Exit(EXIT_BAD_INPUT)

    return documents


@app.command()
def benchmark(
    demo: Annotated[
        bool,
        typer.Option(
            "--demo",
            help="Run a demo showing benchmark tracking and regression detection.",
        ),
    ] = False,
    history: Annotated[
        str | None,
        typer.Option(
            "--history",
            "-H",
            help="Show history for a specific config name.",
        ),
    ] = None,
    list_configs: Annotated[
        bool,
        typer.Option(
            "--list",
            "-l",
            help="List all recorded configurations.",
        ),
    ] = False,
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file for benchmark results (JSON).",
        ),
    ] = None,
    fail_under: Annotated[
        float | None,
        typer.Option(
            "--fail-under",
            help="Fail if average score is below threshold (0.0-1.0).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be benchmarked without executing.",
        ),
    ] = False,
    storage: Annotated[
        str,
        typer.Option(
            "--storage",
            "-s",
            help="Path to benchmark storage file.",
        ),
    ] = ".ragnarok/benchmarks.json",
) -> None:
    """Track benchmark history and detect regressions.

    Examples:
        ragnarok benchmark --demo
        ragnarok benchmark --demo --fail-under 0.7
        ragnarok benchmark --list
        ragnarok benchmark --history my-rag-config
    """
    if not demo and not history and not list_configs:
        if state["json"]:
            typer.echo(json_response("benchmark", "error", errors=["Specify --demo, --history <config>, or --list."]))
        else:
            typer.echo("Error: Specify --demo, --history <config>, or --list.", err=True)
            typer.echo("Run 'ragnarok benchmark --help' for usage.", err=True)
        raise typer.Exit(EXIT_BAD_INPUT)

    if demo:
        _run_benchmark_demo(storage=storage, output=output, fail_under=fail_under, dry_run=dry_run)
    elif list_configs:
        _run_benchmark_list(storage=storage)
    elif history:
        _run_benchmark_history(config_name=history, storage=storage)


def _run_benchmark_demo(
    storage: str,
    output: str | None = None,
    fail_under: float | None = None,
    dry_run: bool = False,
) -> None:
    """Run benchmark demo with simulated runs."""
    from ragnarok_ai.benchmarks import BenchmarkHistory
    from ragnarok_ai.benchmarks.storage import JSONFileStore
    from ragnarok_ai.core.evaluate import EvaluationResult, QueryResult
    from ragnarok_ai.core.types import Query, TestSet
    from ragnarok_ai.evaluators.retrieval import RetrievalMetrics
    from ragnarok_ai.regression import RegressionDetector, RegressionThresholds

    # Dry run - show what would be benchmarked
    if dry_run:
        data = {
            "mode": "demo",
            "configs": [
                {"name": "Run 1 (Baseline)", "description": "Initial baseline metrics"},
                {"name": "Run 2 (Improved)", "description": "Improved configuration"},
                {"name": "Run 3 (Regression)", "description": "Degraded configuration"},
            ],
            "storage": storage,
            "output": output,
            "fail_under": fail_under,
        }
        if state["json"]:
            typer.echo(json_response("benchmark", "dry_run", data=data))
        else:
            typer.echo()
            typer.echo("  [DRY RUN] Would benchmark:")
            typer.echo("    - Run 1 (Baseline): Initial baseline metrics")
            typer.echo("    - Run 2 (Improved): Improved configuration")
            typer.echo("    - Run 3 (Regression): Degraded configuration")
            typer.echo(f"    - Storage: {storage}")
            if output:
                typer.echo(f"    - Output: {output}")
            if fail_under:
                typer.echo(f"    - Fail under: {fail_under}")
            typer.echo()
        return

    if not state["json"]:
        typer.echo()
        typer.echo("  RAGnarok-AI Benchmark Demo")
        typer.echo("  " + "=" * 40)
        typer.echo()
        typer.echo("  Simulating 3 benchmark runs over time...")
        typer.echo()

    # Create test set
    testset = TestSet(
        name="demo-testset",
        queries=[Query(text=f"Question {i}", ground_truth_docs=[f"doc_{i}"]) for i in range(10)],
    )

    # Simulate 3 runs with different metrics
    runs: list[dict[str, str | float]] = [
        {"name": "Run 1 (Baseline)", "precision": 0.72, "recall": 0.68, "mrr": 0.75, "ndcg": 0.70},
        {"name": "Run 2 (Improved)", "precision": 0.78, "recall": 0.74, "mrr": 0.80, "ndcg": 0.76},
        {"name": "Run 3 (Regression)", "precision": 0.65, "recall": 0.60, "mrr": 0.68, "ndcg": 0.62},
    ]

    # Create storage
    store = JSONFileStore(Path(storage))
    history = BenchmarkHistory(store=store)

    async def run_demo() -> dict[str, Any]:
        results_list: list[dict[str, Any]] = []

        for i, run in enumerate(runs):
            # Create mock evaluation result
            p, r, m, n = float(run["precision"]), float(run["recall"]), float(run["mrr"]), float(run["ndcg"])
            metrics = RetrievalMetrics(
                precision=p,
                recall=r,
                mrr=m,
                ndcg=n,
                k=10,
            )
            query_results = [QueryResult(query=q, metric=metrics, answer="", latency_ms=50.0) for q in testset.queries]
            eval_result = EvaluationResult(
                testset=testset,
                metrics=[metrics] * len(testset.queries),
                responses=[""] * len(testset.queries),
                query_results=query_results,
                total_latency_ms=500.0,
            )

            # Record benchmark
            record = await history.record(eval_result, "demo-config", testset)

            if i == 0:
                await history.set_baseline(record.id)

            avg_score = (p + r + m + n) / 4

            if not state["json"]:
                status = "→ Set as baseline" if i == 0 else ""
                typer.echo(f"  {run['name']}")
                typer.echo(f"    Precision: {p:.2f}  Recall: {r:.2f}")
                typer.echo(f"    MRR: {m:.2f}       NDCG: {n:.2f}")
                typer.echo(f"    Average: {avg_score:.2f} {status}")
                typer.echo()

            results_list.append(
                {
                    "run": run["name"],
                    "metrics": run,
                    "average": round(avg_score, 4),
                    "record_id": record.id,
                }
            )

        # Detect regression on last run
        if not state["json"]:
            typer.echo("  " + "-" * 40)
            typer.echo("  Regression Detection (Run 3 vs Baseline)")
            typer.echo("  " + "-" * 40)

        # Create baseline eval result for comparison
        baseline_metrics = RetrievalMetrics(
            precision=float(runs[0]["precision"]),
            recall=float(runs[0]["recall"]),
            mrr=float(runs[0]["mrr"]),
            ndcg=float(runs[0]["ndcg"]),
            k=10,
        )
        baseline_query_results = [
            QueryResult(query=q, metric=baseline_metrics, answer="", latency_ms=50.0) for q in testset.queries
        ]
        baseline_eval = EvaluationResult(
            testset=testset,
            metrics=[baseline_metrics] * len(testset.queries),
            responses=[""] * len(testset.queries),
            query_results=baseline_query_results,
            total_latency_ms=500.0,
        )

        # Create current eval result
        current_metrics = RetrievalMetrics(
            precision=float(runs[2]["precision"]),
            recall=float(runs[2]["recall"]),
            mrr=float(runs[2]["mrr"]),
            ndcg=float(runs[2]["ndcg"]),
            k=10,
        )
        current_query_results = [
            QueryResult(query=q, metric=current_metrics, answer="", latency_ms=50.0) for q in testset.queries
        ]
        current_eval = EvaluationResult(
            testset=testset,
            metrics=[current_metrics] * len(testset.queries),
            responses=[""] * len(testset.queries),
            query_results=current_query_results,
            total_latency_ms=500.0,
        )

        # Detect regression
        detector = RegressionDetector(
            baseline=baseline_eval,
            thresholds=RegressionThresholds(precision_drop=0.05, recall_drop=0.05),
        )
        regression = detector.detect(current_eval)

        regression_data: dict[str, Any] = {
            "has_regression": regression.has_regressions,
            "alerts": [],
        }

        if not state["json"]:
            if regression.has_regressions:
                typer.echo("  ⚠ REGRESSION DETECTED:")
                for alert in regression.alerts:
                    typer.echo(
                        f"    - {alert.metric}: {alert.baseline_value:.2f} → "
                        f"{alert.current_value:.2f} ({alert.change_percent:+.1f}%)"
                    )
                    regression_data["alerts"].append(
                        {
                            "metric": alert.metric,
                            "baseline": alert.baseline_value,
                            "current": alert.current_value,
                            "change_percent": alert.change_percent,
                        }
                    )
            else:
                typer.echo("  ✓ No regression detected")

            # Comparison table
            typer.echo()
            typer.echo("  " + "-" * 40)
            typer.echo("  Comparison Table")
            typer.echo("  " + "-" * 40)
            typer.echo("  Run                  | P     | R     | MRR   | NDCG  | Avg")
            typer.echo("  " + "-" * 60)
            for run_data in results_list:
                m = run_data["metrics"]
                name = str(run_data["run"])[:20].ljust(20)
                typer.echo(
                    f"  {name} | {m['precision']:.3f} | {m['recall']:.3f} | "
                    f"{m['mrr']:.3f} | {m['ndcg']:.3f} | {run_data['average']:.3f}"
                )
            typer.echo()
            typer.echo(f"  Benchmark history saved to: {storage}")
            typer.echo()

        # Calculate best and worst
        best_run = max(results_list, key=lambda x: x["average"])
        worst_run = min(results_list, key=lambda x: x["average"])

        final_data = {
            "storage": storage,
            "runs": results_list,
            "best": {"run": best_run["run"], "average": best_run["average"]},
            "worst": {"run": worst_run["run"], "average": worst_run["average"]},
            "regression": regression_data,
        }

        return final_data

    data = asyncio.run(run_demo())

    # Handle output file (use flat format for backward compatibility)
    if output:
        file_data = {"status": "success", **data}
        Path(output).write_text(json.dumps(file_data, indent=2))
        if not state["json"]:
            typer.echo(f"  Results saved to: {output}")
            typer.echo()

    # Handle fail_under threshold
    exit_code = 0
    status = "success"
    if fail_under is not None:
        worst_data = data["worst"]
        worst_avg = float(worst_data["average"]) if isinstance(worst_data, dict) else 0.0
        if worst_avg < fail_under:
            status = "fail"
            data["fail_reason"] = f"Worst run average {worst_avg:.4f} < threshold {fail_under}"
            exit_code = 1
            if not state["json"]:
                typer.echo(f"  FAIL: Worst run average {worst_avg:.4f} < threshold {fail_under}", err=True)

    if state["json"]:
        typer.echo(json_response("benchmark", status, data=data))

    if exit_code != 0:
        raise typer.Exit(exit_code)


def _run_benchmark_list(storage: str) -> None:
    """List all recorded configurations."""
    storage_path = Path(storage)

    if not storage_path.exists():
        if state["json"]:
            typer.echo(json_response("benchmark", "success", data={"configs": []}))
        else:
            typer.echo()
            typer.echo("  No benchmark history found.")
            typer.echo("  Run 'ragnarok benchmark --demo' to create sample data.")
            typer.echo()
        return

    # Read storage file
    data = json.loads(storage_path.read_text())
    records = data.get("records", [])

    # Group by config name
    configs: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        config_name = record.get("config_name", "unknown")
        if config_name not in configs:
            configs[config_name] = []
        configs[config_name].append(record)

    if state["json"]:
        data = {
            "configs": [
                {"name": name, "runs": len(runs), "latest": runs[0].get("timestamp")} for name, runs in configs.items()
            ],
        }
        typer.echo(json_response("benchmark", "success", data=data))
    else:
        typer.echo()
        typer.echo("  Recorded Configurations")
        typer.echo("  " + "-" * 40)
        if not configs:
            typer.echo("  (none)")
        for name, runs in configs.items():
            latest = runs[0].get("timestamp", "unknown")[:19] if runs else "unknown"
            baseline_count = sum(1 for r in runs if r.get("is_baseline"))
            typer.echo(f"  {name}")
            typer.echo(f"    Runs: {len(runs)}  |  Baseline: {'Yes' if baseline_count else 'No'}  |  Latest: {latest}")
        typer.echo()


def _run_benchmark_history(config_name: str, storage: str) -> None:
    """Show history for a specific config."""
    storage_path = Path(storage)

    if not storage_path.exists():
        if state["json"]:
            typer.echo(json_response("benchmark", "error", errors=[f"No benchmark history found at {storage}"]))
        else:
            typer.echo(f"Error: No benchmark history found at {storage}", err=True)
        raise typer.Exit(EXIT_BAD_INPUT)

    # Read storage file
    data = json.loads(storage_path.read_text())
    records = [r for r in data.get("records", []) if r.get("config_name") == config_name]

    if not records:
        if state["json"]:
            typer.echo(json_response("benchmark", "error", errors=[f"No records for config: {config_name}"]))
        else:
            typer.echo(f"Error: No records found for config '{config_name}'", err=True)
        raise typer.Exit(EXIT_BAD_INPUT)

    if state["json"]:
        typer.echo(json_response("benchmark", "success", data={"config": config_name, "records": records}))
    else:
        typer.echo()
        typer.echo(f"  Benchmark History: {config_name}")
        typer.echo("  " + "-" * 40)
        for record in records:
            timestamp = record.get("timestamp", "unknown")[:19]
            metrics = record.get("metrics", {})
            is_baseline = record.get("is_baseline", False)
            baseline_marker = " [BASELINE]" if is_baseline else ""

            typer.echo(f"  {timestamp}{baseline_marker}")
            typer.echo(
                f"    P={metrics.get('precision', 0):.2f}  R={metrics.get('recall', 0):.2f}  "
                f"MRR={metrics.get('mrr', 0):.2f}  NDCG={metrics.get('ndcg', 0):.2f}"
            )
        typer.echo()


# =============================================================================
# Judge Command
# =============================================================================


@app.command()
def judge(
    context: Annotated[
        str | None,
        typer.Option(
            "--context",
            "-c",
            help="Context text for evaluation.",
        ),
    ] = None,
    question: Annotated[
        str | None,
        typer.Option(
            "--question",
            "-q",
            help="Question to evaluate.",
        ),
    ] = None,
    answer: Annotated[
        str | None,
        typer.Option(
            "--answer",
            "-a",
            help="Answer to evaluate.",
        ),
    ] = None,
    file: Annotated[
        str | None,
        typer.Option(
            "--file",
            "-f",
            help="JSON file with items to evaluate (array of {context, question, answer}).",
        ),
    ] = None,
    criteria: Annotated[
        str,
        typer.Option(
            "--criteria",
            help="Comma-separated criteria: faithfulness,relevance,hallucination,completeness (default: all).",
        ),
    ] = "all",
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Ollama model for evaluation (default: Prometheus 2).",
        ),
    ] = None,
    fail_under: Annotated[
        float | None,
        typer.Option(
            "--fail-under",
            help="Fail if average score is below threshold (0.0-1.0).",
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file for results (JSON).",
        ),
    ] = None,
    ollama_url: Annotated[
        str,
        typer.Option(
            "--ollama-url",
            help="Ollama API base URL.",
        ),
    ] = "http://localhost:11434",
) -> None:
    """Evaluate responses using LLM-as-Judge (Prometheus 2).

    Evaluate a single response:
        ragnarok judge -c "Paris is France's capital." -q "What is France's capital?" -a "Paris"

    Evaluate from file:
        ragnarok judge --file items.json

    Select criteria:
        ragnarok judge --file items.json --criteria faithfulness,relevance
    """
    # Validate input
    has_direct_input = context is not None and question is not None and answer is not None
    has_file_input = file is not None

    if not has_direct_input and not has_file_input:
        if state["json"]:
            typer.echo(
                json_response(
                    "judge",
                    "error",
                    errors=["Provide --context, --question, --answer OR --file."],
                )
            )
        else:
            typer.echo("Error: Provide --context, --question, --answer OR --file.", err=True)
            typer.echo("Run 'ragnarok judge --help' for usage.", err=True)
        raise typer.Exit(EXIT_BAD_INPUT)

    if has_direct_input and has_file_input:
        if state["json"]:
            typer.echo(
                json_response(
                    "judge",
                    "error",
                    errors=["Cannot use both direct input and --file. Choose one."],
                )
            )
        else:
            typer.echo("Error: Cannot use both direct input and --file. Choose one.", err=True)
        raise typer.Exit(EXIT_BAD_INPUT)

    _run_judge(
        context=context,
        question=question,
        answer=answer,
        file=file,
        criteria=criteria,
        model=model,
        fail_under=fail_under,
        output=output,
        ollama_url=ollama_url,
    )


def _run_judge(
    context: str | None,
    question: str | None,
    answer: str | None,
    file: str | None,
    criteria: str,
    model: str | None,
    fail_under: float | None,
    output: str | None,
    ollama_url: str,
) -> None:
    """Run LLM-as-Judge evaluation."""
    from ragnarok_ai.evaluators.judge import LLMJudge

    # Parse criteria
    valid_criteria = {"faithfulness", "relevance", "hallucination", "completeness", "all"}
    if criteria.lower() == "all":
        selected_criteria = ["faithfulness", "relevance", "hallucination", "completeness"]
    else:
        selected_criteria = [c.strip().lower() for c in criteria.split(",")]
        invalid = set(selected_criteria) - valid_criteria
        if invalid:
            if state["json"]:
                typer.echo(
                    json_response(
                        "judge",
                        "error",
                        errors=[f"Invalid criteria: {invalid}. Valid: {valid_criteria - {'all'}}"],
                    )
                )
            else:
                typer.echo(f"Error: Invalid criteria: {invalid}", err=True)
                typer.echo(f"Valid: {', '.join(valid_criteria - {'all'})}", err=True)
            raise typer.Exit(EXIT_BAD_INPUT)

    # Load items to evaluate
    items: list[dict[str, str]] = []
    if file:
        file_path = Path(file)
        if not file_path.exists():
            if state["json"]:
                typer.echo(json_response("judge", "error", errors=[f"File not found: {file}"]))
            else:
                typer.echo(f"Error: File not found: {file}", err=True)
            raise typer.Exit(EXIT_BAD_INPUT)

        try:
            items = json.loads(file_path.read_text())
            if not isinstance(items, list):
                items = [items]
        except json.JSONDecodeError as e:
            if state["json"]:
                typer.echo(json_response("judge", "error", errors=[f"Invalid JSON: {e}"]))
            else:
                typer.echo(f"Error: Invalid JSON in {file}: {e}", err=True)
            raise typer.Exit(EXIT_BAD_INPUT) from None
    else:
        # At this point, direct inputs are validated as non-None
        items = [{"context": context or "", "question": question or "", "answer": answer or ""}]

    if not state["json"]:
        typer.echo()
        typer.echo("  RAGnarok-AI LLM-as-Judge")
        typer.echo("  " + "=" * 40)
        typer.echo()
        typer.echo(f"  Items to evaluate: {len(items)}")
        typer.echo(f"  Criteria: {', '.join(selected_criteria)}")
        typer.echo(f"  Model: {model or 'Prometheus 2 (default)'}")
        typer.echo()
        typer.echo("  Connecting to Ollama...")

    async def run_evaluation() -> dict[str, Any]:
        judge_instance = LLMJudge(model=model, base_url=ollama_url)

        results: list[dict[str, Any]] = []
        total_scores: list[float] = []

        for i, item in enumerate(items):
            ctx = item.get("context", "")
            q = item.get("question", "")
            a = item.get("answer", "")

            if not state["json"]:
                preview = q[:50] + "..." if len(q) > 50 else q
                typer.echo(f"  [{i + 1}/{len(items)}] Evaluating: {preview}")

            item_result: dict[str, Any] = {
                "question": q,
                "answer": a[:100] + "..." if len(a) > 100 else a,
                "criteria": {},
            }

            scores: list[float] = []

            # Evaluate selected criteria
            if "faithfulness" in selected_criteria:
                r = await judge_instance.evaluate_faithfulness(ctx, q, a)
                item_result["criteria"]["faithfulness"] = {
                    "verdict": r.verdict,
                    "score": r.score,
                    "explanation": r.explanation,
                }
                scores.append(r.score)

            if "relevance" in selected_criteria:
                r = await judge_instance.evaluate_relevance(q, a)
                item_result["criteria"]["relevance"] = {
                    "verdict": r.verdict,
                    "score": r.score,
                    "explanation": r.explanation,
                }
                scores.append(r.score)

            if "hallucination" in selected_criteria:
                r = await judge_instance.detect_hallucination(ctx, a)
                item_result["criteria"]["hallucination"] = {
                    "verdict": r.verdict,
                    "score": r.score,
                    "explanation": r.explanation,
                }
                scores.append(r.score)

            if "completeness" in selected_criteria:
                r = await judge_instance.evaluate_completeness(q, a, ctx)
                item_result["criteria"]["completeness"] = {
                    "verdict": r.verdict,
                    "score": r.score,
                    "explanation": r.explanation,
                }
                scores.append(r.score)

            item_avg = sum(scores) / len(scores) if scores else 0
            item_result["average_score"] = round(item_avg, 4)
            total_scores.append(item_avg)
            results.append(item_result)

        overall_avg = sum(total_scores) / len(total_scores) if total_scores else 0

        return {
            "items_evaluated": len(items),
            "criteria": selected_criteria,
            "results": results,
            "overall_average": round(overall_avg, 4),
        }

    try:
        data = asyncio.run(run_evaluation())
    except Exception as e:
        if state["json"]:
            typer.echo(json_response("judge", "error", errors=[str(e)]))
        else:
            typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(EXIT_FAILURE) from None

    # Check threshold
    exit_code = 0
    status = "pass"
    if fail_under is not None and data["overall_average"] < fail_under:
        status = "fail"
        data["fail_reason"] = f"Average score {data['overall_average']:.4f} < threshold {fail_under}"
        exit_code = 1

    # Output results
    if state["json"]:
        typer.echo(json_response("judge", status, data=data))
    else:
        typer.echo()
        typer.echo("  " + "-" * 40)
        typer.echo("  Results")
        typer.echo("  " + "-" * 40)

        for i, result in enumerate(data["results"]):
            typer.echo(f"\n  Item {i + 1}:")
            typer.echo(f"    Question: {result['question'][:60]}...")
            for crit, crit_data in result["criteria"].items():
                verdict_icon = "+" if crit_data["verdict"] == "PASS" else "-" if crit_data["verdict"] == "FAIL" else "~"
                typer.echo(f"    [{verdict_icon}] {crit}: {crit_data['score']:.2f} ({crit_data['verdict']})")
            typer.echo(f"    Average: {result['average_score']:.2f}")

        typer.echo()
        typer.echo("  " + "-" * 40)
        typer.echo(f"  Overall Average: {data['overall_average']:.4f}")

        if fail_under is not None:
            if exit_code == 0:
                typer.echo(f"  Threshold: {fail_under} -> PASS")
            else:
                typer.echo(f"  Threshold: {fail_under} -> FAIL")

        typer.echo()

    # Save to file
    if output:
        file_data = {"status": status, **data}
        Path(output).write_text(json.dumps(file_data, indent=2))
        if not state["json"]:
            typer.echo(f"  Results saved to: {output}")
            typer.echo()

    sys.exit(exit_code)


# =============================================================================
# Plugins Command
# =============================================================================


@app.command()
def plugins(
    list_all: Annotated[
        bool,
        typer.Option(
            "--list",
            "-l",
            help="List all available plugins and adapters.",
        ),
    ] = False,
    adapter_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            "-t",
            help="Filter by adapter type: llm, vectorstore, framework, evaluator.",
        ),
    ] = None,
    local_only: Annotated[
        bool,
        typer.Option(
            "--local",
            help="Only show local adapters (no cloud services).",
        ),
    ] = False,
    info: Annotated[
        str | None,
        typer.Option(
            "--info",
            "-i",
            help="Show detailed info for a specific plugin.",
        ),
    ] = None,
) -> None:
    """Manage and list available plugins.

    Examples:
        ragnarok plugins --list
        ragnarok plugins --list --type llm
        ragnarok plugins --list --local
        ragnarok plugins --info ollama
    """
    if not list_all and not info:
        if state["json"]:
            typer.echo(json_response("plugins", "error", errors=["Specify --list or --info <plugin>."]))
        else:
            typer.echo("Error: Specify --list or --info <plugin>.", err=True)
            typer.echo("Run 'ragnarok plugins --help' for usage.", err=True)
        raise typer.Exit(EXIT_BAD_INPUT)

    if list_all:
        _list_plugins(adapter_type=adapter_type, local_only=local_only)
    elif info:
        _show_plugin_info(plugin_name=info)


def _list_plugins(adapter_type: str | None = None, local_only: bool = False) -> None:
    """List available plugins."""
    from ragnarok_ai.plugins import PluginRegistry

    registry = PluginRegistry.get()
    registry.discover()

    # Validate adapter_type if provided
    valid_types = ["llm", "vectorstore", "framework", "evaluator"]
    if adapter_type and adapter_type.lower() not in valid_types:
        if state["json"]:
            typer.echo(
                json_response(
                    "plugins",
                    "error",
                    errors=[f"Invalid type '{adapter_type}'. Valid: {', '.join(valid_types)}"],
                )
            )
        else:
            typer.echo(f"Error: Invalid type '{adapter_type}'.", err=True)
            typer.echo(f"Valid types: {', '.join(valid_types)}", err=True)
        raise typer.Exit(EXIT_BAD_INPUT)

    # Get plugins with filters
    plugins = registry.list_adapters(
        adapter_type=adapter_type.lower() if adapter_type else None,  # type: ignore[arg-type]
        local_only=local_only,
    )

    if state["json"]:
        data = {
            "plugins": [
                {
                    "name": p.name,
                    "type": p.adapter_type,
                    "is_local": p.is_local,
                    "is_builtin": p.is_builtin,
                }
                for p in plugins
            ],
            "total": len(plugins),
            "filters": {
                "type": adapter_type,
                "local_only": local_only,
            },
        }
        typer.echo(json_response("plugins", "success", data=data))
    else:
        typer.echo()
        title = "Available Plugins"
        if adapter_type:
            title += f" ({adapter_type.upper()})"
        if local_only:
            title += " [Local Only]"
        typer.echo(f"  {title}")
        typer.echo("  " + "-" * 50)

        if not plugins:
            typer.echo("  (none)")
        else:
            # Group by type
            by_type: dict[str, list[Any]] = {}
            for p in plugins:
                if p.adapter_type not in by_type:
                    by_type[p.adapter_type] = []
                by_type[p.adapter_type].append(p)

            for ptype in sorted(by_type.keys()):
                typer.echo(f"\n  {ptype.upper()}:")
                for p in by_type[ptype]:
                    local_tag = "[local]" if p.is_local else "[cloud]"
                    builtin_tag = "" if p.is_builtin else " (plugin)"
                    typer.echo(f"    {p.name:15} {local_tag:8}{builtin_tag}")

        typer.echo()
        typer.echo(f"  Total: {len(plugins)} adapters")
        typer.echo()


def _show_plugin_info(plugin_name: str) -> None:
    """Show detailed info for a plugin."""
    from ragnarok_ai.plugins import PluginNotFoundError, PluginRegistry

    registry = PluginRegistry.get()
    registry.discover()

    try:
        info = registry.get_plugin_info(plugin_name)
    except PluginNotFoundError:
        if state["json"]:
            typer.echo(json_response("plugins", "error", errors=[f"Plugin '{plugin_name}' not found."]))
        else:
            typer.echo(f"Error: Plugin '{plugin_name}' not found.", err=True)
            typer.echo("Run 'ragnarok plugins --list' to see available plugins.", err=True)
        raise typer.Exit(EXIT_BAD_INPUT) from None

    if state["json"]:
        data = {
            "name": info.name,
            "type": info.adapter_type,
            "is_local": info.is_local,
            "is_builtin": info.is_builtin,
            "class": f"{info.adapter_class.__module__}.{info.adapter_class.__name__}",
            "version": info.version,
            "description": info.description,
        }
        typer.echo(json_response("plugins", "success", data=data))
    else:
        typer.echo()
        typer.echo(f"  Plugin: {info.name}")
        typer.echo("  " + "-" * 40)
        typer.echo(f"  Type:     {info.adapter_type}")
        typer.echo(f"  Local:    {'Yes' if info.is_local else 'No'}")
        typer.echo(f"  Builtin:  {'Yes' if info.is_builtin else 'No (external plugin)'}")
        typer.echo(f"  Class:    {info.adapter_class.__module__}.{info.adapter_class.__name__}")
        if info.version:
            typer.echo(f"  Version:  {info.version}")
        if info.description:
            typer.echo(f"  Description: {info.description}")
        typer.echo()


if __name__ == "__main__":
    app()
