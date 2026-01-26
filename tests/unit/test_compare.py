"""Unit tests for the compare module."""

from __future__ import annotations

import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from ragnarok_ai.core.compare import (
    LOWER_IS_BETTER,
    ComparisonProgress,
    ComparisonResult,
    ConfigResult,
    _compute_testset_hash,
    compare,
)
from ragnarok_ai.core.evaluate import EvaluationResult
from ragnarok_ai.core.types import Query, TestSet
from ragnarok_ai.evaluators.retrieval import RetrievalMetrics

if TYPE_CHECKING:
    from types import ModuleType

# Get the actual module from sys.modules (avoids shadowing by ragnarok_ai.core.compare function)
compare_module: ModuleType = sys.modules["ragnarok_ai.core.compare"]

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_testset() -> TestSet:
    """Create a sample testset for testing."""
    return TestSet(
        name="test_set",
        queries=[
            Query(text="What is RAG?", ground_truth_docs=["doc1", "doc2"]),
            Query(text="How does retrieval work?", ground_truth_docs=["doc3"]),
        ],
    )


@pytest.fixture
def sample_metrics() -> RetrievalMetrics:
    """Create sample retrieval metrics."""
    return RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.9, ndcg=0.85, k=10)


@pytest.fixture
def sample_evaluation_result(sample_testset: TestSet, sample_metrics: RetrievalMetrics) -> EvaluationResult:
    """Create a sample evaluation result."""
    return EvaluationResult(
        testset=sample_testset,
        metrics=[sample_metrics, sample_metrics],
        responses=["Answer 1", "Answer 2"],
        total_latency_ms=100.0,
    )


@pytest.fixture
def sample_config_result(sample_evaluation_result: EvaluationResult) -> ConfigResult:
    """Create a sample config result."""
    return ConfigResult(
        config_name="baseline",
        evaluation=sample_evaluation_result,
        summary_metrics=sample_evaluation_result.summary(),
        k=10,
    )


@pytest.fixture
def sample_comparison_result(
    sample_testset: TestSet,
    sample_evaluation_result: EvaluationResult,
) -> ComparisonResult:
    """Create a sample comparison result with multiple configs."""
    # Create results with different metrics
    baseline_eval = sample_evaluation_result

    # Create experiment eval with better metrics
    better_metrics = RetrievalMetrics(precision=0.85, recall=0.75, mrr=0.92, ndcg=0.88, k=10)
    experiment_eval = EvaluationResult(
        testset=sample_testset,
        metrics=[better_metrics, better_metrics],
        responses=["Better answer 1", "Better answer 2"],
        total_latency_ms=120.0,
    )

    # Create fast eval with worse metrics but faster
    fast_metrics = RetrievalMetrics(precision=0.75, recall=0.65, mrr=0.85, ndcg=0.80, k=10)
    fast_eval = EvaluationResult(
        testset=sample_testset,
        metrics=[fast_metrics, fast_metrics],
        responses=["Fast answer 1", "Fast answer 2"],
        total_latency_ms=50.0,
    )

    return ComparisonResult(
        testset=sample_testset,
        results={
            "baseline": ConfigResult(
                config_name="baseline",
                evaluation=baseline_eval,
                summary_metrics=baseline_eval.summary(),
                k=10,
            ),
            "experiment": ConfigResult(
                config_name="experiment",
                evaluation=experiment_eval,
                summary_metrics=experiment_eval.summary(),
                k=10,
            ),
            "fast": ConfigResult(
                config_name="fast",
                evaluation=fast_eval,
                summary_metrics=fast_eval.summary(),
                k=10,
            ),
        },
    )


# ============================================================================
# TestSet Hash Tests
# ============================================================================


class TestTestsetHash:
    """Tests for testset hash computation."""

    def test_testset_hash_computed(self, sample_testset: TestSet) -> None:
        """testset_hash is computed from testset content."""
        hash_value = _compute_testset_hash(sample_testset)

        assert hash_value
        assert len(hash_value) == 16
        assert hash_value.isalnum()

    def test_testset_hash_deterministic(self, sample_testset: TestSet) -> None:
        """Same testset produces same hash."""
        hash1 = _compute_testset_hash(sample_testset)
        hash2 = _compute_testset_hash(sample_testset)

        assert hash1 == hash2

    def test_testset_hash_differs_for_different_testsets(self) -> None:
        """Different testsets produce different hashes."""
        testset1 = TestSet(
            queries=[Query(text="Question 1", ground_truth_docs=["doc1"])]
        )
        testset2 = TestSet(
            queries=[Query(text="Question 2", ground_truth_docs=["doc2"])]
        )

        hash1 = _compute_testset_hash(testset1)
        hash2 = _compute_testset_hash(testset2)

        assert hash1 != hash2


# ============================================================================
# ComparisonResult Tests
# ============================================================================


class TestComparisonResult:
    """Tests for ComparisonResult methods."""

    def test_timestamp_set_automatically(self, sample_testset: TestSet) -> None:
        """timestamp is set to current time on creation."""
        before = datetime.now()
        result = ComparisonResult(testset=sample_testset, results={})
        after = datetime.now()

        assert before <= result.timestamp <= after

    def test_testset_hash_computed_on_init(self, sample_testset: TestSet) -> None:
        """testset_hash is computed automatically if not provided."""
        result = ComparisonResult(testset=sample_testset, results={})

        assert result.testset_hash
        assert len(result.testset_hash) == 16

    def test_testset_hash_preserved_if_provided(self, sample_testset: TestSet) -> None:
        """testset_hash is preserved if provided."""
        custom_hash = "custom_hash_12345"
        result = ComparisonResult(
            testset=sample_testset,
            results={},
            testset_hash=custom_hash,
        )

        assert result.testset_hash == custom_hash

    def test_summary_format(self, sample_comparison_result: ComparisonResult) -> None:
        """summary() returns formatted table."""
        summary = sample_comparison_result.summary()

        assert isinstance(summary, str)
        assert "baseline" in summary
        assert "experiment" in summary
        assert "fast" in summary
        assert "Precision" in summary
        assert "NDCG" in summary
        assert "*" in summary  # Winner marker

    def test_summary_empty_results(self, sample_testset: TestSet) -> None:
        """summary() handles empty results."""
        result = ComparisonResult(testset=sample_testset, results={})
        summary = result.summary()

        assert "No results" in summary

    def test_winner_higher_is_better(self, sample_comparison_result: ComparisonResult) -> None:
        """winner() handles higher_is_better correctly (precision, ndcg)."""
        # Experiment has best ndcg (0.88)
        assert sample_comparison_result.winner("ndcg") == "experiment"
        assert sample_comparison_result.winner("precision") == "experiment"

    def test_winner_latency_lower_is_better(self, sample_comparison_result: ComparisonResult) -> None:
        """winner('latency_ms') returns fastest config."""
        # Fast has lowest latency (50ms)
        assert sample_comparison_result.winner("latency_ms") == "fast"

    def test_winner_total_latency_lower_is_better(self, sample_comparison_result: ComparisonResult) -> None:
        """winner('total_latency_ms') returns fastest config."""
        # Fast has lowest latency (50ms)
        assert sample_comparison_result.winner("total_latency_ms") == "fast"

    def test_winner_empty_results_raises(self, sample_testset: TestSet) -> None:
        """winner() raises ValueError for empty results."""
        result = ComparisonResult(testset=sample_testset, results={})

        with pytest.raises(ValueError, match="No results"):
            result.winner("ndcg")

    def test_rankings(self, sample_comparison_result: ComparisonResult) -> None:
        """rankings() returns sorted configs per metric."""
        rankings = sample_comparison_result.rankings()

        assert "ndcg" in rankings
        assert "precision" in rankings
        assert "total_latency_ms" in rankings

        # Best ndcg first
        assert rankings["ndcg"][0] == "experiment"

        # Best latency first (lowest)
        assert rankings["total_latency_ms"][0] == "fast"

    def test_pairwise(self, sample_comparison_result: ComparisonResult) -> None:
        """pairwise() returns comparison details."""
        comparison = sample_comparison_result.pairwise("baseline", "experiment")

        assert "ndcg" in comparison
        assert "precision" in comparison
        assert "total_latency_ms" in comparison

        ndcg = comparison["ndcg"]
        assert "value_a" in ndcg
        assert "value_b" in ndcg
        assert "difference" in ndcg
        assert "percent_change" in ndcg
        assert "winner" in ndcg

    def test_pairwise_invalid_config_raises(self, sample_comparison_result: ComparisonResult) -> None:
        """pairwise() raises ValueError for invalid config names."""
        with pytest.raises(ValueError, match="not found"):
            sample_comparison_result.pairwise("baseline", "nonexistent")

    def test_to_dict_serializable(self, sample_comparison_result: ComparisonResult) -> None:
        """to_dict() returns JSON-serializable dict."""
        import json

        d = sample_comparison_result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(d)
        assert json_str

        # Check structure
        assert "timestamp" in d
        assert "testset_hash" in d
        assert "results" in d
        assert "rankings" in d
        assert "winners" in d

    def test_export_json(self, sample_comparison_result: ComparisonResult, tmp_path: Any) -> None:
        """export() writes JSON file."""
        path = tmp_path / "comparison.json"
        sample_comparison_result.export(str(path))

        assert path.exists()

        import json
        with path.open() as f:
            data = json.load(f)

        assert "results" in data
        assert "baseline" in data["results"]

    def test_export_html(self, sample_comparison_result: ComparisonResult, tmp_path: Any) -> None:
        """export() writes HTML file."""
        path = tmp_path / "comparison.html"
        sample_comparison_result.export(str(path))

        assert path.exists()

        content = path.read_text()
        assert "<html>" in content
        assert "baseline" in content
        assert "experiment" in content

    def test_export_unsupported_format_raises(
        self, sample_comparison_result: ComparisonResult, tmp_path: Any
    ) -> None:
        """export() raises ValueError for unsupported format."""
        path = tmp_path / "comparison.xyz"

        with pytest.raises(ValueError, match="Unsupported"):
            sample_comparison_result.export(str(path))

    def test_warnings_included_in_summary(
        self, sample_testset: TestSet, sample_comparison_result: ComparisonResult
    ) -> None:
        """Warnings are included in summary output."""
        # Add a warning to the existing result
        result = ComparisonResult(
            testset=sample_testset,
            results=sample_comparison_result.results,
            warnings=["k differs: baseline=10, experiment=15"],
        )
        summary = result.summary()

        assert "Warnings" in summary
        assert "k differs" in summary


# ============================================================================
# compare() Function Tests
# ============================================================================


class TestCompare:
    """Tests for compare() function."""

    @pytest.mark.asyncio
    async def test_compare_basic(self, sample_testset: TestSet) -> None:
        """Compare two configs returns ComparisonResult."""

        async def mock_evaluate(*_args: Any, **_kwargs: Any) -> EvaluationResult:
            return EvaluationResult(
                testset=sample_testset,
                metrics=[RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.9, ndcg=0.85, k=10)],
                responses=["Test answer"],
            )

        with patch.object(compare_module, "evaluate", side_effect=mock_evaluate):
            mock_pipeline = MagicMock()

            def rag_factory(_config: dict[str, Any]) -> Any:
                return mock_pipeline

            configs = {
                "baseline": {"chunk_size": 512},
                "experiment": {"chunk_size": 256},
            }

            result = await compare(
                rag_factory=rag_factory,
                configs=configs,
                testset=sample_testset,
            )

            assert isinstance(result, ComparisonResult)
            assert "baseline" in result.results
            assert "experiment" in result.results

    @pytest.mark.asyncio
    async def test_compare_same_testset_enforced(self, sample_testset: TestSet) -> None:
        """All configs are evaluated on exact same testset."""
        testsets_used: list[TestSet] = []

        async def mock_evaluate(*_args: Any, **kwargs: Any) -> EvaluationResult:
            testsets_used.append(kwargs.get("testset"))
            return EvaluationResult(
                testset=sample_testset,
                metrics=[RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.9, ndcg=0.85, k=10)],
                responses=["Answer"],
            )

        with patch.object(compare_module, "evaluate", side_effect=mock_evaluate):
            mock_pipeline = MagicMock()

            await compare(
                rag_factory=lambda _c: mock_pipeline,
                configs={"a": {}, "b": {}, "c": {}},
                testset=sample_testset,
            )

            # All evaluations should use the same testset
            assert len(testsets_used) == 3
            assert all(ts is sample_testset for ts in testsets_used)

    @pytest.mark.asyncio
    async def test_compare_results_keyed_by_name(self, sample_testset: TestSet) -> None:
        """Results dict uses config names as keys."""

        async def mock_evaluate(*_args: Any, **_kwargs: Any) -> EvaluationResult:
            return EvaluationResult(
                testset=sample_testset,
                metrics=[RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.9, ndcg=0.85, k=10)],
                responses=["Test"],
            )

        with patch.object(compare_module, "evaluate", side_effect=mock_evaluate):
            mock_pipeline = MagicMock()

            result = await compare(
                rag_factory=lambda _c: mock_pipeline,
                configs={
                    "alpha": {"param": 1},
                    "beta": {"param": 2},
                },
                testset=sample_testset,
            )

            assert "alpha" in result.results
            assert "beta" in result.results
            assert result.results["alpha"].config_name == "alpha"
            assert result.results["beta"].config_name == "beta"

    @pytest.mark.asyncio
    async def test_compare_k_differs_warning(self, sample_testset: TestSet) -> None:
        """Warning when k differs between configs."""

        async def mock_evaluate(*_args: Any, **_kwargs: Any) -> EvaluationResult:
            return EvaluationResult(
                testset=sample_testset,
                metrics=[RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.9, ndcg=0.85, k=10)],
                responses=["Test"],
            )

        with patch.object(compare_module, "evaluate", side_effect=mock_evaluate):
            mock_pipeline = MagicMock()

            result = await compare(
                rag_factory=lambda _c: mock_pipeline,
                configs={
                    "baseline": {"k": 10},
                    "experiment": {"k": 15},
                },
                testset=sample_testset,
            )

            assert len(result.warnings) == 1
            assert "k differs" in result.warnings[0]
            assert "baseline=10" in result.warnings[0]
            assert "experiment=15" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_compare_empty_configs_error(self, sample_testset: TestSet) -> None:
        """Empty configs raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await compare(
                rag_factory=lambda _c: MagicMock(),
                configs={},
                testset=sample_testset,
            )

    @pytest.mark.asyncio
    async def test_compare_invalid_baseline_name_error(self, sample_testset: TestSet) -> None:
        """Invalid baseline_name raises ValueError."""
        with pytest.raises(ValueError, match="not in configs"):
            await compare(
                rag_factory=lambda _c: MagicMock(),
                configs={"a": {}, "b": {}},
                testset=sample_testset,
                baseline_name="nonexistent",
            )

    @pytest.mark.asyncio
    async def test_compare_baseline_name_stored(self, sample_testset: TestSet) -> None:
        """baseline_name is stored in ComparisonResult."""

        async def mock_evaluate(*_args: Any, **_kwargs: Any) -> EvaluationResult:
            return EvaluationResult(
                testset=sample_testset,
                metrics=[RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.9, ndcg=0.85, k=10)],
                responses=["Test"],
            )

        with patch.object(compare_module, "evaluate", side_effect=mock_evaluate):
            mock_pipeline = MagicMock()

            result = await compare(
                rag_factory=lambda _c: mock_pipeline,
                configs={"baseline": {}, "experiment": {}},
                testset=sample_testset,
                baseline_name="baseline",
            )

            assert result.baseline_name == "baseline"

    @pytest.mark.asyncio
    async def test_compare_on_progress_callback(self, sample_testset: TestSet) -> None:
        """on_progress receives ComparisonProgress."""

        async def mock_evaluate(*_args: Any, **_kwargs: Any) -> EvaluationResult:
            return EvaluationResult(
                testset=sample_testset,
                metrics=[RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.9, ndcg=0.85, k=10)],
                responses=["Test"],
            )

        progress_calls: list[ComparisonProgress] = []

        def progress_callback(progress: ComparisonProgress) -> None:
            progress_calls.append(progress)

        with patch.object(compare_module, "evaluate", side_effect=mock_evaluate):
            mock_pipeline = MagicMock()

            await compare(
                rag_factory=lambda _c: mock_pipeline,
                configs={"a": {}, "b": {}},
                testset=sample_testset,
                on_progress=progress_callback,
            )

            assert len(progress_calls) == 2
            assert progress_calls[0].current_config == "a"
            assert progress_calls[0].config_index == 1
            assert progress_calls[0].total_configs == 2
            assert progress_calls[1].current_config == "b"
            assert progress_calls[1].config_index == 2

    @pytest.mark.asyncio
    async def test_compare_winner(self, sample_testset: TestSet) -> None:
        """winner() returns best config for metric."""

        async def mock_evaluate(*_args: Any, **_kwargs: Any) -> EvaluationResult:
            return EvaluationResult(
                testset=sample_testset,
                metrics=[RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.9, ndcg=0.85, k=10)],
                responses=["Test"],
            )

        with patch.object(compare_module, "evaluate", side_effect=mock_evaluate):
            mock_pipeline = MagicMock()

            result = await compare(
                rag_factory=lambda _c: mock_pipeline,
                configs={"a": {}, "b": {}},
                testset=sample_testset,
            )

            # Should not raise
            winner = result.winner("ndcg")
            assert winner in ["a", "b"]

    @pytest.mark.asyncio
    async def test_compare_rankings(self, sample_testset: TestSet) -> None:
        """rankings() returns sorted configs per metric."""

        async def mock_evaluate(*_args: Any, **_kwargs: Any) -> EvaluationResult:
            return EvaluationResult(
                testset=sample_testset,
                metrics=[RetrievalMetrics(precision=0.8, recall=0.7, mrr=0.9, ndcg=0.85, k=10)],
                responses=["Test"],
            )

        with patch.object(compare_module, "evaluate", side_effect=mock_evaluate):
            mock_pipeline = MagicMock()

            result = await compare(
                rag_factory=lambda _c: mock_pipeline,
                configs={"x": {}, "y": {}, "z": {}},
                testset=sample_testset,
            )

            rankings = result.rankings()

            assert "ndcg" in rankings
            assert len(rankings["ndcg"]) == 3
            assert set(rankings["ndcg"]) == {"x", "y", "z"}


# ============================================================================
# LOWER_IS_BETTER Tests
# ============================================================================


class TestLowerIsBetter:
    """Tests for LOWER_IS_BETTER constant."""

    def test_latency_metrics_in_lower_is_better(self) -> None:
        """Latency metrics are in LOWER_IS_BETTER set."""
        assert "latency_ms" in LOWER_IS_BETTER
        assert "avg_latency_ms" in LOWER_IS_BETTER
        assert "total_latency_ms" in LOWER_IS_BETTER

    def test_quality_metrics_not_in_lower_is_better(self) -> None:
        """Quality metrics are NOT in LOWER_IS_BETTER set."""
        assert "precision" not in LOWER_IS_BETTER
        assert "recall" not in LOWER_IS_BETTER
        assert "mrr" not in LOWER_IS_BETTER
        assert "ndcg" not in LOWER_IS_BETTER
