"""Unit tests for the baselines module."""

from __future__ import annotations

import pytest

from ragnarok_ai.baselines import (
    BASELINE_CONFIGS,
    BaselineComparison,
    BaselineConfig,
    BaselineResult,
    MetricComparison,
    compare,
    get_baseline_config,
    get_baseline_result,
    list_baselines,
)

# ============================================================================
# BaselineConfig Tests
# ============================================================================


class TestBaselineConfig:
    """Tests for BaselineConfig model."""

    def test_create_with_required_fields(self) -> None:
        """Test creating a config with required fields."""
        config = BaselineConfig(
            name="test",
            description="Test configuration",
            chunk_size=512,
            chunk_overlap=50,
            embedder="test-embed",
        )
        assert config.name == "test"
        assert config.description == "Test configuration"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.embedder == "test-embed"

    def test_default_values(self) -> None:
        """Test default values."""
        config = BaselineConfig(
            name="test",
            description="Test",
            chunk_size=256,
            chunk_overlap=25,
            embedder="test-embed",
        )
        assert config.retrieval_k == 10
        assert config.metadata == {}

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = BaselineConfig(
            name="test",
            description="Test",
            chunk_size=512,
            chunk_overlap=50,
            embedder="test-embed",
            retrieval_k=15,
            metadata={"use_case": "testing"},
        )
        d = config.to_dict()
        assert d["chunk_size"] == 512
        assert d["chunk_overlap"] == 50
        assert d["embedder"] == "test-embed"
        assert d["retrieval_k"] == 15
        assert d["use_case"] == "testing"
        # name and description should not be in dict
        assert "name" not in d
        assert "description" not in d


class TestGetBaselineConfig:
    """Tests for get_baseline_config function."""

    def test_get_balanced(self) -> None:
        """Test getting balanced config."""
        config = get_baseline_config("balanced")
        assert config.name == "balanced"
        assert config.chunk_size == 512

    def test_get_precision(self) -> None:
        """Test getting precision config."""
        config = get_baseline_config("precision")
        assert config.name == "precision"
        assert config.chunk_size == 256

    def test_get_speed(self) -> None:
        """Test getting speed config."""
        config = get_baseline_config("speed")
        assert config.name == "speed"
        assert config.chunk_size == 1024

    def test_get_memory_efficient(self) -> None:
        """Test getting memory_efficient config."""
        config = get_baseline_config("memory_efficient")
        assert config.name == "memory_efficient"

    def test_get_semantic(self) -> None:
        """Test getting semantic config."""
        config = get_baseline_config("semantic")
        assert config.name == "semantic"

    def test_unknown_baseline_raises(self) -> None:
        """Test that unknown baseline raises KeyError."""
        with pytest.raises(KeyError, match="Unknown baseline 'nonexistent'"):
            get_baseline_config("nonexistent")


class TestListBaselines:
    """Tests for list_baselines function."""

    def test_returns_all_baselines(self) -> None:
        """Test that all baselines are returned."""
        baselines = list_baselines()
        assert "balanced" in baselines
        assert "precision" in baselines
        assert "speed" in baselines
        assert "memory_efficient" in baselines
        assert "semantic" in baselines

    def test_matches_configs(self) -> None:
        """Test that list matches BASELINE_CONFIGS keys."""
        assert list_baselines() == list(BASELINE_CONFIGS.keys())


# ============================================================================
# BaselineResult Tests
# ============================================================================


class TestBaselineResult:
    """Tests for BaselineResult model."""

    def test_create_with_required_fields(self) -> None:
        """Test creating a result with required fields."""
        result = BaselineResult(baseline_name="test")
        assert result.baseline_name == "test"
        assert result.dataset == "ragnarok-reference"

    def test_create_with_all_fields(self) -> None:
        """Test creating a result with all fields."""
        result = BaselineResult(
            baseline_name="test",
            dataset="custom-dataset",
            precision_at_k={5: 0.8, 10: 0.75},
            recall_at_k={5: 0.4, 10: 0.6},
            mrr=0.72,
            ndcg=0.68,
            faithfulness=0.85,
            relevance=0.80,
            latency_ms=150,
            metadata={"version": "1.0"},
        )
        assert result.precision_at_k[5] == 0.8
        assert result.recall_at_k[10] == 0.6
        assert result.mrr == 0.72
        assert result.faithfulness == 0.85
        assert result.latency_ms == 150


class TestBaselineResultGetMetric:
    """Tests for BaselineResult.get_metric method."""

    @pytest.fixture
    def sample_result(self) -> BaselineResult:
        """Create a sample result for testing."""
        return BaselineResult(
            baseline_name="test",
            precision_at_k={5: 0.8, 10: 0.75, 20: 0.7},
            recall_at_k={5: 0.4, 10: 0.6, 20: 0.75},
            mrr=0.72,
            ndcg=0.68,
            faithfulness=0.85,
            relevance=0.80,
            latency_ms=150,
        )

    def test_get_precision_with_k(self, sample_result: BaselineResult) -> None:
        """Test getting precision at specific k."""
        assert sample_result.get_metric("precision", k=5) == 0.8
        assert sample_result.get_metric("precision_at_k", k=10) == 0.75
        assert sample_result.get_metric("p@k", k=20) == 0.7

    def test_get_precision_default_k(self, sample_result: BaselineResult) -> None:
        """Test getting precision at default k=10."""
        assert sample_result.get_metric("precision") == 0.75

    def test_get_recall_with_k(self, sample_result: BaselineResult) -> None:
        """Test getting recall at specific k."""
        assert sample_result.get_metric("recall", k=5) == 0.4
        assert sample_result.get_metric("recall_at_k", k=10) == 0.6
        assert sample_result.get_metric("r@k", k=20) == 0.75

    def test_get_recall_default_k(self, sample_result: BaselineResult) -> None:
        """Test getting recall at default k=10."""
        assert sample_result.get_metric("recall") == 0.6

    def test_get_mrr(self, sample_result: BaselineResult) -> None:
        """Test getting MRR."""
        assert sample_result.get_metric("mrr") == 0.72

    def test_get_ndcg(self, sample_result: BaselineResult) -> None:
        """Test getting NDCG."""
        assert sample_result.get_metric("ndcg") == 0.68

    def test_get_faithfulness(self, sample_result: BaselineResult) -> None:
        """Test getting faithfulness."""
        assert sample_result.get_metric("faithfulness") == 0.85

    def test_get_relevance(self, sample_result: BaselineResult) -> None:
        """Test getting relevance."""
        assert sample_result.get_metric("relevance") == 0.80

    def test_get_latency(self, sample_result: BaselineResult) -> None:
        """Test getting latency."""
        assert sample_result.get_metric("latency") == 150
        assert sample_result.get_metric("latency_ms") == 150

    def test_get_unknown_metric(self, sample_result: BaselineResult) -> None:
        """Test getting unknown metric returns None."""
        assert sample_result.get_metric("unknown") is None

    def test_get_metric_case_insensitive(self, sample_result: BaselineResult) -> None:
        """Test that metric names are case-insensitive."""
        assert sample_result.get_metric("MRR") == 0.72
        assert sample_result.get_metric("NDCG") == 0.68
        assert sample_result.get_metric("Precision") == 0.75


class TestGetBaselineResult:
    """Tests for get_baseline_result function."""

    def test_get_balanced(self) -> None:
        """Test getting balanced result."""
        result = get_baseline_result("balanced")
        assert result.baseline_name == "balanced"
        assert result.mrr == 0.72

    def test_get_precision(self) -> None:
        """Test getting precision result."""
        result = get_baseline_result("precision")
        assert result.baseline_name == "precision"
        assert result.mrr == 0.80

    def test_unknown_baseline_raises(self) -> None:
        """Test that unknown baseline raises KeyError."""
        with pytest.raises(KeyError, match="Unknown baseline 'nonexistent'"):
            get_baseline_result("nonexistent")


# ============================================================================
# MetricComparison Tests
# ============================================================================


class TestMetricComparison:
    """Tests for MetricComparison model."""

    def test_create(self) -> None:
        """Test creating a metric comparison."""
        comp = MetricComparison(
            metric_name="precision",
            your_value=0.80,
            baseline_value=0.75,
            difference=0.05,
            percent_change=6.67,
            is_better=True,
        )
        assert comp.metric_name == "precision"
        assert comp.your_value == 0.80
        assert comp.baseline_value == 0.75
        assert comp.is_better is True

    def test_status_better(self) -> None:
        """Test status when better."""
        comp = MetricComparison(
            metric_name="precision",
            your_value=0.80,
            baseline_value=0.75,
            difference=0.05,
            percent_change=6.67,
            is_better=True,
        )
        assert comp.status == "better"

    def test_status_worse(self) -> None:
        """Test status when worse."""
        comp = MetricComparison(
            metric_name="precision",
            your_value=0.70,
            baseline_value=0.75,
            difference=-0.05,
            percent_change=-6.67,
            is_better=False,
        )
        assert comp.status == "worse"

    def test_status_equal(self) -> None:
        """Test status when equal (less than 1% change)."""
        comp = MetricComparison(
            metric_name="precision",
            your_value=0.75,
            baseline_value=0.75,
            difference=0.0,
            percent_change=0.5,
            is_better=False,
        )
        assert comp.status == "equal"

    def test_formatted_change_positive(self) -> None:
        """Test formatted change for positive change."""
        comp = MetricComparison(
            metric_name="precision",
            your_value=0.80,
            baseline_value=0.75,
            difference=0.05,
            percent_change=6.67,
            is_better=True,
        )
        assert comp.formatted_change == "+6.7%"

    def test_formatted_change_negative(self) -> None:
        """Test formatted change for negative change."""
        comp = MetricComparison(
            metric_name="precision",
            your_value=0.70,
            baseline_value=0.75,
            difference=-0.05,
            percent_change=-6.67,
            is_better=False,
        )
        assert comp.formatted_change == "-6.7%"

    def test_formatted_change_zero(self) -> None:
        """Test formatted change for zero change."""
        comp = MetricComparison(
            metric_name="precision",
            your_value=0.75,
            baseline_value=0.75,
            difference=0.0,
            percent_change=0.0,
            is_better=False,
        )
        assert comp.formatted_change == "+0.0%"


# ============================================================================
# BaselineComparison Tests
# ============================================================================


class TestBaselineComparison:
    """Tests for BaselineComparison model."""

    @pytest.fixture
    def sample_comparison(self) -> BaselineComparison:
        """Create a sample comparison for testing."""
        return BaselineComparison(
            baseline_name="balanced",
            metrics=[
                MetricComparison(
                    metric_name="precision",
                    your_value=0.80,
                    baseline_value=0.75,
                    difference=0.05,
                    percent_change=6.67,
                    is_better=True,
                ),
                MetricComparison(
                    metric_name="recall",
                    your_value=0.58,
                    baseline_value=0.62,
                    difference=-0.04,
                    percent_change=-6.45,
                    is_better=False,
                ),
                MetricComparison(
                    metric_name="mrr",
                    your_value=0.72,
                    baseline_value=0.72,
                    difference=0.0,
                    percent_change=0.0,
                    is_better=False,
                ),
            ],
            overall_score=0.15,
            summary_text="Your results are comparable to the baseline",
        )

    def test_better_count(self, sample_comparison: BaselineComparison) -> None:
        """Test counting better metrics."""
        assert sample_comparison.better_count == 1

    def test_worse_count(self, sample_comparison: BaselineComparison) -> None:
        """Test counting worse metrics."""
        assert sample_comparison.worse_count == 1

    def test_equal_count(self, sample_comparison: BaselineComparison) -> None:
        """Test counting equal metrics."""
        assert sample_comparison.equal_count == 1

    def test_summary(self, sample_comparison: BaselineComparison) -> None:
        """Test summary generation."""
        summary = sample_comparison.summary()
        assert "balanced" in summary
        assert "precision" in summary
        assert "recall" in summary
        assert "mrr" in summary
        assert "Better: 1" in summary
        assert "Worse: 1" in summary
        assert "Equal: 1" in summary

    def test_to_dict(self, sample_comparison: BaselineComparison) -> None:
        """Test conversion to dictionary."""
        d = sample_comparison.to_dict()
        assert d["baseline_name"] == "balanced"
        assert d["overall_score"] == 0.15
        assert d["better_count"] == 1
        assert d["worse_count"] == 1
        assert d["equal_count"] == 1
        assert len(d["metrics"]) == 3
        assert d["metrics"][0]["name"] == "precision"
        assert d["metrics"][0]["status"] == "better"


# ============================================================================
# compare() Function Tests
# ============================================================================


class TestCompareFunction:
    """Tests for compare function."""

    def test_compare_with_baseline_name(self) -> None:
        """Test comparing with baseline name string."""
        result = compare(
            your_results={"precision": 0.80, "recall": 0.65, "mrr": 0.74},
            baseline="balanced",
        )
        assert result.baseline_name == "balanced"
        assert len(result.metrics) == 3

    def test_compare_with_baseline_result(self) -> None:
        """Test comparing with BaselineResult object."""
        baseline_result = BaselineResult(
            baseline_name="custom",
            precision_at_k={10: 0.70},
            recall_at_k={10: 0.60},
            mrr=0.70,
        )
        result = compare(
            your_results={"precision": 0.75, "recall": 0.65, "mrr": 0.72},
            baseline=baseline_result,
        )
        assert result.baseline_name == "custom"

    def test_compare_better_results(self) -> None:
        """Test comparison with better results."""
        result = compare(
            your_results={"precision": 0.85, "recall": 0.70, "mrr": 0.80},
            baseline="balanced",
        )
        assert result.overall_score > 0.2
        assert "better than the baseline" in result.summary_text

    def test_compare_worse_results(self) -> None:
        """Test comparison with worse results."""
        result = compare(
            your_results={"precision": 0.60, "recall": 0.45, "mrr": 0.55},
            baseline="balanced",
        )
        assert result.overall_score < -0.2
        assert "below the baseline" in result.summary_text

    def test_compare_comparable_results(self) -> None:
        """Test comparison with comparable results."""
        # Use exact baseline values to get comparable result
        result = compare(
            your_results={"precision": 0.75, "recall": 0.62, "mrr": 0.72},
            baseline="balanced",
        )
        # With exact values, percent changes are 0 so overall score is 0
        assert result.overall_score == 0.0
        assert "comparable" in result.summary_text

    def test_compare_higher_is_better_latency(self) -> None:
        """Test that latency is correctly treated as lower-is-better."""
        result = compare(
            your_results={"latency_ms": 100},  # Lower than baseline (150)
            baseline="balanced",
        )
        # Lower latency should be better
        assert result.metrics[0].is_better is True

    def test_compare_custom_higher_is_better(self) -> None:
        """Test custom higher_is_better mapping."""
        # Test that custom higher_is_better doesn't break the function
        # Custom metrics not in get_metric will be skipped
        result = compare(
            your_results={"mrr": 0.60},
            baseline=BaselineResult(
                baseline_name="test",
                mrr=0.70,
            ),
            higher_is_better={"mrr": False},  # Treat lower as better
        )
        # With higher_is_better=False, 0.60 < 0.70 should be better
        assert result.metrics[0].is_better is True

    def test_compare_skips_missing_metrics(self) -> None:
        """Test that comparison skips metrics not in baseline."""
        result = compare(
            your_results={"precision": 0.80, "unknown_metric": 0.90},
            baseline="balanced",
        )
        # Should only have precision, unknown_metric should be skipped
        metric_names = [m.metric_name for m in result.metrics]
        assert "precision" in metric_names
        assert "unknown_metric" not in metric_names

    def test_compare_empty_results(self) -> None:
        """Test comparison with empty results."""
        result = compare(
            your_results={},
            baseline="balanced",
        )
        assert len(result.metrics) == 0
        assert result.overall_score == 0.0

    def test_compare_zero_baseline_value(self) -> None:
        """Test comparison when baseline value is zero."""
        result = compare(
            your_results={"mrr": 0.50},
            baseline=BaselineResult(
                baseline_name="zero-test",
                mrr=0.0,
            ),
        )
        # Should handle division by zero
        assert result.metrics[0].percent_change == 100.0

    def test_compare_percent_change_calculation(self) -> None:
        """Test percent change is calculated correctly."""
        result = compare(
            your_results={"mrr": 0.80},
            baseline=BaselineResult(
                baseline_name="test",
                mrr=0.72,
            ),
        )
        # (0.80 - 0.72) / 0.72 * 100 = 11.11%
        assert abs(result.metrics[0].percent_change - 11.11) < 0.1

    def test_compare_all_baselines(self) -> None:
        """Test that comparison works with all pre-defined baselines."""
        for baseline_name in list_baselines():
            result = compare(
                your_results={"precision": 0.75, "mrr": 0.70},
                baseline=baseline_name,
            )
            assert result.baseline_name == baseline_name
