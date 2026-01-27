#!/usr/bin/env python3
"""End-to-end installation and functionality test for ragnarok-ai.

Run this script to verify the package works correctly before PyPI publish.

Usage:
    python tests/e2e/test_install.py
    # or
    uv run python tests/e2e/test_install.py
"""

from __future__ import annotations

import sys

# Track results
results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator to track test results."""

    def decorator(func):
        def wrapper():
            try:
                func()
                results.append((name, True, ""))
            except Exception as e:
                results.append((name, False, str(e)))

        return wrapper

    return decorator


# =============================================================================
# 1. Import Tests
# =============================================================================


@check("Import: core types")
def test_import_core_types():
    from ragnarok_ai.core.types import Document, RAGResponse, TestSet


@check("Import: core protocols")
def test_import_core_protocols():
    from ragnarok_ai.core.protocols import (
        EvaluatorProtocol,
        LLMProtocol,
        RAGProtocol,
        VectorStoreProtocol,
    )


@check("Import: core evaluate")
def test_import_core_evaluate():
    from ragnarok_ai.core.evaluate import EvaluationResult, evaluate


@check("Import: evaluators")
def test_import_evaluators():
    from ragnarok_ai.evaluators import (
        FaithfulnessEvaluator,
        HallucinationDetector,
        RelevanceEvaluator,
        mrr,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )


@check("Import: generators")
def test_import_generators():
    from ragnarok_ai.generators import TestSetGenerator, load_testset, save_testset


@check("Import: reporters")
def test_import_reporters():
    from ragnarok_ai.reporters import ConsoleReporter, HTMLReporter, JSONReporter


@check("Import: cache")
def test_import_cache():
    from ragnarok_ai.cache import DiskCache, MemoryCache


@check("Import: benchmarks")
def test_import_benchmarks():
    from ragnarok_ai.benchmarks import BenchmarkHistory, BenchmarkRecord


@check("Import: baselines compare")
def test_import_baselines_compare():
    from ragnarok_ai.baselines.comparison import compare


@check("Import: regression")
def test_import_regression():
    from ragnarok_ai.regression import RegressionDetector


@check("Import: diff")
def test_import_diff():
    from ragnarok_ai.diff import DiffReport


@check("Import: agents types")
def test_import_agents_types():
    from ragnarok_ai.agents.types import AgentResponse, AgentStep, ToolCall


@check("Import: agents evaluators")
def test_import_agents_evaluators():
    from ragnarok_ai.agents.evaluators import (
        ExpectedToolCall,
        ToolUseMetrics,
        evaluate_tool_use,
    )
    from ragnarok_ai.agents.evaluators.reasoning import (
        GoalProgressEvaluator,
        ReasoningCoherenceEvaluator,
        ReasoningEfficiencyEvaluator,
    )


@check("Import: agents analysis")
def test_import_agents_analysis():
    from ragnarok_ai.agents.analysis import (
        FailurePoint,
        TrajectoryAnalyzer,
        TrajectorySummary,
        format_trajectory_ascii,
        format_trajectory_mermaid,
    )


@check("Import: adapters agents")
def test_import_adapters_agents():
    from ragnarok_ai.adapters.agents import (
        ChainOfThoughtAdapter,
        ReActAdapter,
        ReActParser,
    )


# =============================================================================
# 2. MockRAG Evaluation Test
# =============================================================================


@check("Evaluation: retrieval metrics")
def test_retrieval_metrics():
    from ragnarok_ai.evaluators import mrr, ndcg_at_k, precision_at_k, recall_at_k

    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc1", "doc3", "doc5"]

    p = precision_at_k(retrieved, relevant, k=5)
    r = recall_at_k(retrieved, relevant, k=5)
    m = mrr(retrieved, relevant)
    n = ndcg_at_k(retrieved, relevant, k=5)

    assert 0 <= p <= 1, f"Invalid precision: {p}"
    assert 0 <= r <= 1, f"Invalid recall: {r}"
    assert 0 <= m <= 1, f"Invalid MRR: {m}"
    assert 0 <= n <= 1, f"Invalid NDCG: {n}"


@check("Evaluation: RAGResponse creation")
def test_rag_response():
    from ragnarok_ai.core.types import Document, RAGResponse

    response = RAGResponse(
        answer="Test answer",
        retrieved_docs=[Document(id="doc1", content="Test context", metadata={})],
    )
    assert response.answer == "Test answer"
    assert len(response.retrieved_docs) == 1


# =============================================================================
# 3. Compare Test
# =============================================================================


@check("Benchmarks: compare()")
def test_compare():
    from ragnarok_ai.baselines import BaselineResult, compare

    # Create baseline with required fields
    baseline = BaselineResult(
        baseline_name="test-baseline",
        dataset="test-dataset",
        mrr=0.8,
        ndcg=0.75,
        faithfulness=0.85,
    )

    # Your results to compare (dict of metric name -> value)
    your_results = {
        "mrr": 0.85,
        "ndcg": 0.80,
        "faithfulness": 0.9,
    }

    comparison = compare(
        your_results=your_results,
        baseline=baseline,
    )

    assert comparison is not None


@check("Benchmarks: BenchmarkHistory")
def test_benchmark_history():
    import asyncio

    from ragnarok_ai.benchmarks import BenchmarkHistory
    from ragnarok_ai.core.evaluate import EvaluationResult
    from ragnarok_ai.core.types import Query, TestSet
    from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

    async def run_test():
        history = BenchmarkHistory()

        # Create a minimal TestSet
        testset = TestSet(
            queries=[Query(text="test question", ground_truth_docs=["doc1"])],
            name="test-set",
        )

        # Create a minimal EvaluationResult
        result = EvaluationResult(
            testset=testset,
            metrics=[RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.9, ndcg=0.85, k=5)],
            responses=["test answer"],
        )

        # Record the result (async)
        record = await history.record(result, config_name="test-config", testset=testset)

        # Check history has records
        all_history = await history.get_history(config_name="test-config")
        assert len(all_history) >= 1
        assert record.config_name == "test-config"

    asyncio.run(run_test())


# =============================================================================
# 4. Agents Module Test
# =============================================================================


@check("Agents: AgentResponse & ToolCall")
def test_agent_response():
    from ragnarok_ai.agents.types import AgentResponse, AgentStep, ToolCall

    tool_call = ToolCall(
        name="search",
        input={"query": "test"},
        output="Found: test result",
    )

    step = AgentStep(
        step_type="action",
        content="Searching for test",
        tool_call=tool_call,
    )

    response = AgentResponse(
        answer="The answer is 42",
        steps=[step],
        total_latency_ms=150.0,
    )

    assert response.answer == "The answer is 42"
    assert len(response.steps) == 1
    assert response.steps[0].tool_call is not None
    assert response.steps[0].tool_call.name == "search"


@check("Agents: evaluate_tool_use")
def test_evaluate_tool_use():
    from ragnarok_ai.agents.evaluators import (
        ExpectedToolCall,
        evaluate_tool_use,
    )
    from ragnarok_ai.agents.types import AgentResponse, AgentStep, ToolCall

    response = AgentResponse(
        answer="Done",
        steps=[
            AgentStep(
                step_type="action",
                content="search",
                tool_call=ToolCall(
                    name="search",
                    input={"query": "test"},
                    output="result",
                ),
            ),
            AgentStep(
                step_type="action",
                content="calculate",
                tool_call=ToolCall(
                    name="calculate",
                    input={"expr": "2+2"},
                    output="4",
                ),
            ),
        ],
    )

    # Without expected tools
    metrics = evaluate_tool_use(response)
    assert metrics.total_calls == 2
    assert metrics.success_rate == 1.0

    # With expected tools
    expected = [
        ExpectedToolCall(name="search", required_args={"query"}),
        ExpectedToolCall(name="calculate", required_args={"expr"}),
    ]
    metrics = evaluate_tool_use(response, expected_tools=expected)
    assert metrics.precision is not None
    assert metrics.recall is not None


@check("Agents: TrajectoryAnalyzer")
def test_trajectory_analyzer():
    from ragnarok_ai.agents.analysis import TrajectoryAnalyzer
    from ragnarok_ai.agents.types import AgentResponse, AgentStep, ToolCall

    response = AgentResponse(
        answer="42",
        steps=[
            AgentStep(step_type="thought", content="I need to calculate"),
            AgentStep(
                step_type="action",
                content="calculate",
                tool_call=ToolCall(name="calc", input={"x": 6, "y": 7}, output="42"),
            ),
            AgentStep(step_type="final_answer", content="42"),
        ],
        total_latency_ms=100.0,
    )

    analyzer = TrajectoryAnalyzer()
    summary = analyzer.analyze(response)

    assert summary.total_steps == 3
    assert summary.tool_calls == {"calc": 1}
    assert summary.failed_tool_calls == 0


@check("Agents: ReActParser")
def test_react_parser():
    from ragnarok_ai.adapters.agents import ReActParser

    parser = ReActParser()

    text = """
Thought: I need to search for information
Action: search
Action Input: {"query": "capital of France"}
Observation: Paris is the capital of France
Thought: Now I know the answer
Final Answer: Paris
"""

    steps = parser.parse(text)
    assert len(steps) >= 3

    response = parser.parse_to_response(text)
    assert response.answer == "Paris"


@check("Agents: format_trajectory_ascii")
def test_trajectory_ascii():
    from ragnarok_ai.agents.analysis import format_trajectory_ascii
    from ragnarok_ai.agents.types import AgentResponse, AgentStep

    response = AgentResponse(
        answer="test",
        steps=[
            AgentStep(step_type="thought", content="thinking..."),
            AgentStep(step_type="final_answer", content="test"),
        ],
    )

    output = format_trajectory_ascii(response)
    assert "Step 1" in output
    assert "thought" in output


@check("Agents: format_trajectory_mermaid")
def test_trajectory_mermaid():
    from ragnarok_ai.agents.analysis import format_trajectory_mermaid
    from ragnarok_ai.agents.types import AgentResponse, AgentStep

    response = AgentResponse(
        answer="test",
        steps=[
            AgentStep(step_type="thought", content="thinking..."),
            AgentStep(step_type="final_answer", content="test"),
        ],
    )

    output = format_trajectory_mermaid(response)
    assert "```mermaid" in output
    assert "graph TD" in output


# =============================================================================
# 5. Run All Tests
# =============================================================================


def print_summary():
    """Print test summary with emojis."""
    print("\n" + "=" * 60)
    print("RAGnarok-AI Installation Test Summary")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    for name, success, error in results:
        if success:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}")
            print(f"     Error: {error[:80]}...")
            failed += 1

    print("\n" + "-" * 60)
    print(f"  Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
    print("-" * 60)

    if failed == 0:
        print("\n  🎉 All tests passed! Ready for PyPI.\n")
        return 0
    else:
        print(f"\n  ⚠️  {failed} test(s) failed. Fix before PyPI.\n")
        return 1


def main():
    """Run all tests."""
    print("\nRunning RAGnarok-AI installation tests...\n")

    # 1. Import tests
    test_import_core_types()
    test_import_core_protocols()
    test_import_core_evaluate()
    test_import_evaluators()
    test_import_generators()
    test_import_reporters()
    test_import_cache()
    test_import_benchmarks()
    test_import_baselines_compare()
    test_import_regression()
    test_import_diff()
    test_import_agents_types()
    test_import_agents_evaluators()
    test_import_agents_analysis()
    test_import_adapters_agents()

    # 2. Evaluation tests
    test_retrieval_metrics()
    test_rag_response()

    # 3. Compare tests
    test_compare()
    test_benchmark_history()

    # 4. Agents tests
    test_agent_response()
    test_evaluate_tool_use()
    test_trajectory_analyzer()
    test_react_parser()
    test_trajectory_ascii()
    test_trajectory_mermaid()

    # 5. Summary
    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
