#!/usr/bin/env python3
"""RAGnarok-AI Performance Benchmarks.

Measures evaluation performance across different scales:
- Time per query
- Queries per second
- Peak RAM usage
- LLM-as-Judge latency (if Ollama available)
- Checkpointing overhead

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --skip-llm  # Skip LLM-as-Judge benchmarks
    python benchmarks/run_benchmarks.py --output results/my_run.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class SystemInfo:
    """System information for benchmark context."""

    cpu: str
    ram_gb: int
    python_version: str
    os: str
    ragnarok_version: str

    @classmethod
    def collect(cls) -> SystemInfo:
        """Collect current system information."""
        import psutil

        from ragnarok_ai import __version__

        # Get CPU info
        if platform.system() == "Darwin":
            try:
                cpu = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                ).strip()
            except Exception:
                cpu = platform.processor() or "Unknown"
        else:
            cpu = platform.processor() or "Unknown"

        return cls(
            cpu=cpu,
            ram_gb=round(psutil.virtual_memory().total / (1024**3)),
            python_version=platform.python_version(),
            os=f"{platform.system()} {platform.release()}",
            ragnarok_version=__version__,
        )


@dataclass
class EvaluationResult:
    """Results from an evaluation benchmark."""

    queries: int
    time_s: float
    queries_per_sec: float
    ram_peak_mb: float
    ram_avg_mb: float


@dataclass
class LLMJudgeResult:
    """Results from LLM-as-Judge benchmark."""

    criterion: str
    samples: int
    avg_time_s: float
    min_time_s: float
    max_time_s: float
    std_dev_s: float


@dataclass
class CheckpointResult:
    """Results from checkpointing benchmark."""

    queries: int
    time_without_s: float
    time_with_s: float
    overhead_pct: float


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    timestamp: str
    system: SystemInfo
    evaluation: dict[str, EvaluationResult] = field(default_factory=dict)
    llm_judge: list[LLMJudgeResult] = field(default_factory=list)
    checkpointing: CheckpointResult | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "system": asdict(self.system),
            "evaluation": {k: asdict(v) for k, v in self.evaluation.items()},
            "llm_judge": [asdict(r) for r in self.llm_judge],
            "checkpointing": asdict(self.checkpointing) if self.checkpointing else None,
            "errors": self.errors,
        }


def generate_mock_testset(num_queries: int) -> Any:
    """Generate a mock testset for benchmarking."""
    from ragnarok_ai.core.types import Query, TestSet

    queries = []
    for i in range(num_queries):
        queries.append(
            Query(
                text=f"What is the answer to question {i}?",
                ground_truth_docs=[f"doc_{i}", f"doc_{i + 1}"],
                expected_answer=f"The answer to question {i} is {i * 2}.",
            )
        )

    return TestSet(queries=queries)


def generate_mock_responses(num_queries: int) -> list[Any]:
    """Generate mock RAG responses for benchmarking."""
    from ragnarok_ai.core.types import RAGResponse

    responses = []
    for i in range(num_queries):
        responses.append(
            RAGResponse(
                answer=f"The answer to question {i} is {i * 2}.",
                retrieved_docs=[f"doc_{i}", f"doc_{i + 1}", f"doc_{i + 2}"],
                context=f"Document {i} contains information about question {i}.",
            )
        )
    return responses


def benchmark_evaluation(num_queries: int) -> EvaluationResult:
    """Benchmark evaluation performance for given number of queries."""
    from ragnarok_ai.core.types import Document, RetrievalResult
    from ragnarok_ai.evaluators.retrieval import evaluate_retrieval

    print(f"  Benchmarking {num_queries} queries...")

    # Generate test data
    testset = generate_mock_testset(num_queries)

    # Setup memory tracking
    gc.collect()
    tracemalloc.start()

    ram_samples = []

    # Run evaluation
    start_time = time.perf_counter()

    for i, query in enumerate(testset.queries):
        # Create mock documents
        docs = [
            Document(id=f"doc_{i}", content=f"Content for document {i}"),
            Document(id=f"doc_{i + 1}", content=f"Content for document {i + 1}"),
            Document(id=f"doc_{i + 2}", content=f"Content for document {i + 2}"),
        ]

        # Create RetrievalResult with ground truth
        result = RetrievalResult(
            query=query,
            retrieved_docs=docs,
            scores=[0.9, 0.8, 0.7],
        )
        evaluate_retrieval(result, k=10)

        # Sample memory periodically
        current, _ = tracemalloc.get_traced_memory()
        ram_samples.append(current / (1024 * 1024))  # Convert to MB

    end_time = time.perf_counter()

    # Get peak memory
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed = end_time - start_time

    return EvaluationResult(
        queries=num_queries,
        time_s=round(elapsed, 3),
        queries_per_sec=round(num_queries / elapsed, 2),
        ram_peak_mb=round(peak / (1024 * 1024), 2),
        ram_avg_mb=round(statistics.mean(ram_samples), 2) if ram_samples else 0,
    )


def check_ollama_available() -> bool:
    """Check if Ollama is running and has a model available."""
    try:
        import httpx

        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return len(models) > 0
    except Exception:
        pass
    return False


def benchmark_llm_judge(samples: int = 3) -> list[LLMJudgeResult]:
    """Benchmark LLM-as-Judge latency for each criterion."""
    import asyncio

    from ragnarok_ai.evaluators.judge import LLMJudge

    print("  Benchmarking LLM-as-Judge...")

    results = []
    judge = LLMJudge()

    # Test data
    context = "Python is a programming language created by Guido van Rossum in 1991. It emphasizes code readability and simplicity."
    question = "Who created Python and when?"
    answer = "Python was created by Guido van Rossum in 1991."

    criteria = [
        ("faithfulness", lambda: judge.evaluate_faithfulness(context, question, answer)),
        ("relevance", lambda: judge.evaluate_relevance(question, answer)),
        ("hallucination", lambda: judge.detect_hallucination(context, answer)),
        ("completeness", lambda: judge.evaluate_completeness(question, answer, context)),
    ]

    for criterion_name, eval_func in criteria:
        print(f"    {criterion_name}...")
        times = []

        for i in range(samples):
            start = time.perf_counter()
            asyncio.run(eval_func())
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"      Sample {i + 1}: {elapsed:.2f}s")

        results.append(
            LLMJudgeResult(
                criterion=criterion_name,
                samples=samples,
                avg_time_s=round(statistics.mean(times), 2),
                min_time_s=round(min(times), 2),
                max_time_s=round(max(times), 2),
                std_dev_s=round(statistics.stdev(times), 2) if len(times) > 1 else 0,
            )
        )

    return results


def benchmark_checkpointing(num_queries: int = 100) -> CheckpointResult:
    """Benchmark checkpointing overhead."""
    import tempfile

    from ragnarok_ai.core.types import Document, RetrievalResult
    from ragnarok_ai.evaluators.retrieval import evaluate_retrieval

    print(f"  Benchmarking checkpointing ({num_queries} queries)...")

    testset = generate_mock_testset(num_queries)

    # Without checkpointing
    gc.collect()
    start = time.perf_counter()
    for i, query in enumerate(testset.queries):
        docs = [
            Document(id=f"doc_{i}", content=f"Content {i}"),
            Document(id=f"doc_{i + 1}", content=f"Content {i + 1}"),
            Document(id=f"doc_{i + 2}", content=f"Content {i + 2}"),
        ]
        result = RetrievalResult(
            query=query,
            retrieved_docs=docs,
            scores=[0.9, 0.8, 0.7],
        )
        evaluate_retrieval(result, k=10)
    time_without = time.perf_counter() - start

    # With checkpointing (simulated with JSON writes)
    gc.collect()
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "benchmark.ckpt"
        all_results: list[dict[str, Any]] = []

        start = time.perf_counter()
        for i, query in enumerate(testset.queries):
            docs = [
                Document(id=f"doc_{i}", content=f"Content {i}"),
                Document(id=f"doc_{i + 1}", content=f"Content {i + 1}"),
                Document(id=f"doc_{i + 2}", content=f"Content {i + 2}"),
            ]
            result = RetrievalResult(
                query=query,
                retrieved_docs=docs,
                scores=[0.9, 0.8, 0.7],
            )
            metrics = evaluate_retrieval(result, k=10)
            all_results.append({"index": i, "metrics": metrics.model_dump()})

            # Simulate checkpoint save every 10 queries
            if i % 10 == 0:
                checkpoint_path.write_text(json.dumps({"results": all_results, "count": i}))

        time_with = time.perf_counter() - start

    overhead_pct = ((time_with - time_without) / time_without) * 100

    return CheckpointResult(
        queries=num_queries,
        time_without_s=round(time_without, 3),
        time_with_s=round(time_with, 3),
        overhead_pct=round(overhead_pct, 2),
    )


def run_benchmarks(skip_llm: bool = False, output_path: Path | None = None) -> BenchmarkResults:
    """Run all benchmarks and return results."""
    print("=" * 60)
    print("RAGnarok-AI Performance Benchmarks")
    print("=" * 60)

    # Collect system info
    print("\nCollecting system information...")
    try:
        system_info = SystemInfo.collect()
        print(f"  CPU: {system_info.cpu}")
        print(f"  RAM: {system_info.ram_gb}GB")
        print(f"  Python: {system_info.python_version}")
        print(f"  RAGnarok: {system_info.ragnarok_version}")
    except Exception as e:
        print(f"  Warning: Could not collect full system info: {e}")
        system_info = SystemInfo(
            cpu="Unknown",
            ram_gb=0,
            python_version=platform.python_version(),
            os=platform.system(),
            ragnarok_version="unknown",
        )

    results = BenchmarkResults(
        timestamp=datetime.now(timezone.utc).isoformat(),
        system=system_info,
    )

    # Evaluation benchmarks
    print("\n[1/3] Evaluation Performance")
    print("-" * 40)

    for num_queries in [50, 500, 5000]:
        try:
            result = benchmark_evaluation(num_queries)
            results.evaluation[f"queries_{num_queries}"] = result
            print(f"    -> {result.time_s}s, {result.queries_per_sec} q/s, {result.ram_peak_mb}MB peak")
        except Exception as e:
            error_msg = f"Evaluation {num_queries}: {e}"
            results.errors.append(error_msg)
            print(f"    -> Error: {e}")

    # LLM-as-Judge benchmarks
    print("\n[2/3] LLM-as-Judge Latency")
    print("-" * 40)

    if skip_llm:
        print("  Skipped (--skip-llm)")
    elif not check_ollama_available():
        print("  Skipped (Ollama not available)")
        results.errors.append("LLM-as-Judge: Ollama not available")
    else:
        try:
            results.llm_judge = benchmark_llm_judge(samples=3)
            for r in results.llm_judge:
                print(f"    {r.criterion}: {r.avg_time_s}s avg ({r.min_time_s}-{r.max_time_s}s)")
        except Exception as e:
            error_msg = f"LLM-as-Judge: {e}"
            results.errors.append(error_msg)
            print(f"  Error: {e}")

    # Checkpointing benchmarks
    print("\n[3/3] Checkpointing Overhead")
    print("-" * 40)

    try:
        results.checkpointing = benchmark_checkpointing(100)
        print(f"    Without: {results.checkpointing.time_without_s}s")
        print(f"    With: {results.checkpointing.time_with_s}s")
        print(f"    Overhead: {results.checkpointing.overhead_pct}%")
    except Exception as e:
        error_msg = f"Checkpointing: {e}"
        results.errors.append(error_msg)
        print(f"  Error: {e}")

    # Save results
    if output_path is None:
        output_path = Path(__file__).parent / "results" / f"{datetime.now().strftime('%Y-%m-%d')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results.to_dict(), indent=2))
    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nEvaluation Performance:")
    print("| Queries | Time    | Queries/sec | Peak RAM |")
    print("|---------|---------|-------------|----------|")
    for key, r in results.evaluation.items():
        time_str = f"{r.time_s}s" if r.time_s < 60 else f"{r.time_s / 60:.1f}m"
        print(f"| {r.queries:<7} | {time_str:<7} | {r.queries_per_sec:<11} | {r.ram_peak_mb} MB |")

    if results.llm_judge:
        print("\nLLM-as-Judge (Prometheus 2):")
        print("| Criterion     | Avg Time |")
        print("|---------------|----------|")
        for r in results.llm_judge:
            print(f"| {r.criterion:<13} | {r.avg_time_s}s |")

    if results.checkpointing:
        print(f"\nCheckpointing overhead: {results.checkpointing.overhead_pct}%")

    if results.errors:
        print(f"\nWarnings/Errors: {len(results.errors)}")
        for err in results.errors:
            print(f"  - {err}")

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run RAGnarok-AI performance benchmarks")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM-as-Judge benchmarks")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file path")

    args = parser.parse_args()

    try:
        run_benchmarks(skip_llm=args.skip_llm, output_path=args.output)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
