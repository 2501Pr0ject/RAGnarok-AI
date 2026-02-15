"""Unit tests for artifacts module (fingerprints, runs, traces, experiments)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragnarok_ai.core.artifacts import (
    Environment,
    Experiment,
    ExperimentVariant,
    Fingerprint,
    GeneratorConfig,
    JudgeConfig,
    PipelineConfig,
    RetrieverConfig,
    Run,
    Trace,
    _canonical_json,
    _sha256_hex,
    _sha256_hex_short,
)

# =============================================================================
# Hashing Helpers
# =============================================================================


class TestHashingHelpers:
    """Tests for hashing helper functions."""

    def test_canonical_json_sorts_keys(self) -> None:
        """Canonical JSON should sort keys for determinism."""
        data1 = {"b": 1, "a": 2}
        data2 = {"a": 2, "b": 1}
        assert _canonical_json(data1) == _canonical_json(data2)

    def test_canonical_json_no_spaces(self) -> None:
        """Canonical JSON should have no extra spaces."""
        data = {"key": "value", "num": 123}
        result = _canonical_json(data)
        assert " " not in result
        assert result == '{"key":"value","num":123}'

    def test_sha256_hex_length(self) -> None:
        """SHA256 hex should be 64 characters."""
        result = _sha256_hex("test")
        assert len(result) == 64

    def test_sha256_hex_deterministic(self) -> None:
        """Same input should produce same hash."""
        assert _sha256_hex("hello") == _sha256_hex("hello")
        assert _sha256_hex("hello") != _sha256_hex("world")

    def test_sha256_hex_short(self) -> None:
        """Short hash should truncate correctly."""
        full = _sha256_hex("test")
        short = _sha256_hex_short("test", n=16)
        assert short == full[:16]
        assert len(short) == 16


# =============================================================================
# Environment
# =============================================================================


class TestEnvironment:
    """Tests for Environment model."""

    def test_capture_returns_environment(self) -> None:
        """Environment.capture() should return valid Environment."""
        env = Environment.capture()
        assert isinstance(env, Environment)
        assert env.python_version
        assert env.platform
        assert env.ragnarok_version

    def test_environment_fingerprint_is_deterministic(self) -> None:
        """Same environment should produce same fingerprint."""
        env = Environment.capture()
        assert env.fingerprint() == env.fingerprint()

    def test_environment_fingerprint_is_full_sha256(self) -> None:
        """Environment fingerprint should be full 64-char SHA256."""
        env = Environment.capture()
        fp = env.fingerprint()
        assert len(fp) == 64

    def test_environment_is_frozen(self) -> None:
        """Environment should be immutable."""
        env = Environment.capture()
        with pytest.raises(ValidationError):
            env.python_version = "changed"  # type: ignore[misc]


# =============================================================================
# Fingerprint
# =============================================================================


class TestFingerprint:
    """Tests for Fingerprint model."""

    def test_fingerprint_combined_is_full_sha256(self) -> None:
        """Combined fingerprint should be full 64-char SHA256."""
        fp = Fingerprint(
            dataset_hash="a" * 64,
            config_hash="b" * 64,
            env_hash="c" * 64,
        )
        assert len(fp.combined) == 64

    def test_fingerprint_short_is_16_chars(self) -> None:
        """Short fingerprint should be 16 characters."""
        fp = Fingerprint(
            dataset_hash="a" * 64,
            config_hash="b" * 64,
            env_hash="c" * 64,
        )
        assert len(fp.short) == 16
        assert fp.short == fp.combined[:16]

    def test_fingerprint_str_is_short(self) -> None:
        """str(fingerprint) should return short form."""
        fp = Fingerprint(
            dataset_hash="a" * 64,
            config_hash="b" * 64,
            env_hash="c" * 64,
        )
        assert str(fp) == fp.short

    def test_fingerprint_changes_with_different_inputs(self) -> None:
        """Different inputs should produce different fingerprints."""
        fp1 = Fingerprint(dataset_hash="a" * 64, config_hash="b" * 64, env_hash="c" * 64)
        fp2 = Fingerprint(dataset_hash="x" * 64, config_hash="b" * 64, env_hash="c" * 64)
        assert fp1.combined != fp2.combined


# =============================================================================
# Config Models
# =============================================================================


class TestRetrieverConfig:
    """Tests for RetrieverConfig model."""

    def test_fingerprint_is_stable(self) -> None:
        """Same config should produce same fingerprint."""
        cfg1 = RetrieverConfig(type="qdrant", embedding_model="bge-base", top_k=10)
        cfg2 = RetrieverConfig(type="qdrant", embedding_model="bge-base", top_k=10)
        assert cfg1.fingerprint() == cfg2.fingerprint()

    def test_fingerprint_excludes_metadata(self) -> None:
        """Metadata should not affect fingerprint."""
        cfg1 = RetrieverConfig(type="qdrant", metadata={"foo": "bar"})
        cfg2 = RetrieverConfig(type="qdrant", metadata={"different": "value"})
        assert cfg1.fingerprint() == cfg2.fingerprint()

    def test_fingerprint_changes_with_config(self) -> None:
        """Different config should produce different fingerprint."""
        cfg1 = RetrieverConfig(type="qdrant", top_k=10)
        cfg2 = RetrieverConfig(type="qdrant", top_k=20)
        assert cfg1.fingerprint() != cfg2.fingerprint()


class TestGeneratorConfig:
    """Tests for GeneratorConfig model."""

    def test_fingerprint_is_stable(self) -> None:
        """Same config should produce same fingerprint."""
        cfg1 = GeneratorConfig(model="mistral", provider="ollama", temperature=0.0)
        cfg2 = GeneratorConfig(model="mistral", provider="ollama", temperature=0.0)
        assert cfg1.fingerprint() == cfg2.fingerprint()

    def test_fingerprint_changes_with_temperature(self) -> None:
        """Different temperature should produce different fingerprint."""
        cfg1 = GeneratorConfig(model="mistral", provider="ollama", temperature=0.0)
        cfg2 = GeneratorConfig(model="mistral", provider="ollama", temperature=0.7)
        assert cfg1.fingerprint() != cfg2.fingerprint()


class TestJudgeConfig:
    """Tests for JudgeConfig model."""

    def test_fingerprint_is_stable(self) -> None:
        """Same config should produce same fingerprint."""
        cfg1 = JudgeConfig(model="prometheus", criteria=["faithfulness"])
        cfg2 = JudgeConfig(model="prometheus", criteria=["faithfulness"])
        assert cfg1.fingerprint() == cfg2.fingerprint()

    def test_fingerprint_changes_with_medical_mode(self) -> None:
        """Medical mode should affect fingerprint."""
        cfg1 = JudgeConfig(medical_mode=False)
        cfg2 = JudgeConfig(medical_mode=True)
        assert cfg1.fingerprint() != cfg2.fingerprint()


class TestPipelineConfig:
    """Tests for PipelineConfig model."""

    def test_fingerprint_is_stable(self) -> None:
        """Same pipeline config should produce same fingerprint."""
        cfg1 = PipelineConfig(
            name="test",
            generator=GeneratorConfig(model="mistral", provider="ollama"),
        )
        cfg2 = PipelineConfig(
            name="test",
            generator=GeneratorConfig(model="mistral", provider="ollama"),
        )
        assert cfg1.fingerprint() == cfg2.fingerprint()

    def test_fingerprint_includes_all_subconfigs(self) -> None:
        """Pipeline fingerprint should change when any subconfig changes."""
        base = PipelineConfig(
            name="test",
            retriever=RetrieverConfig(top_k=10),
            generator=GeneratorConfig(model="mistral", provider="ollama"),
            judge=JudgeConfig(criteria=["faithfulness"]),
        )

        # Change retriever
        with_different_retriever = PipelineConfig(
            name="test",
            retriever=RetrieverConfig(top_k=20),
            generator=GeneratorConfig(model="mistral", provider="ollama"),
            judge=JudgeConfig(criteria=["faithfulness"]),
        )
        assert base.fingerprint() != with_different_retriever.fingerprint()

        # Change judge
        with_different_judge = PipelineConfig(
            name="test",
            retriever=RetrieverConfig(top_k=10),
            generator=GeneratorConfig(model="mistral", provider="ollama"),
            judge=JudgeConfig(criteria=["relevance"]),
        )
        assert base.fingerprint() != with_different_judge.fingerprint()

    def test_fingerprint_excludes_metadata(self) -> None:
        """Metadata should not affect pipeline fingerprint."""
        cfg1 = PipelineConfig(name="test", metadata={"a": 1})
        cfg2 = PipelineConfig(name="test", metadata={"b": 2})
        assert cfg1.fingerprint() == cfg2.fingerprint()


# =============================================================================
# Run
# =============================================================================


class TestRun:
    """Tests for Run model."""

    def test_create_generates_id(self) -> None:
        """Run.create() should generate unique ID."""
        cfg = PipelineConfig(name="test")
        run = Run.create(
            dataset_name="test_ds",
            dataset_hash="a" * 64,
            dataset_size=100,
            pipeline_config=cfg,
        )
        assert run.run_id
        assert len(run.run_id) == 12

    def test_create_sets_pending_status(self) -> None:
        """New run should have pending status."""
        cfg = PipelineConfig(name="test")
        run = Run.create(
            dataset_name="test_ds",
            dataset_hash="a" * 64,
            dataset_size=100,
            pipeline_config=cfg,
        )
        assert run.status == "pending"

    def test_create_computes_fingerprint(self) -> None:
        """Run.create() should compute fingerprint."""
        cfg = PipelineConfig(name="test")
        run = Run.create(
            dataset_name="test_ds",
            dataset_hash="a" * 64,
            dataset_size=100,
            pipeline_config=cfg,
        )
        assert run.fingerprint.dataset_hash == "a" * 64
        assert run.fingerprint.config_hash == cfg.fingerprint()

    def test_fingerprint_changes_when_config_changes(self) -> None:
        """Different config should produce different run fingerprint."""
        cfg_a = PipelineConfig(
            name="test",
            generator=GeneratorConfig(model="mistral", provider="ollama", temperature=0.0),
        )
        cfg_b = PipelineConfig(
            name="test",
            generator=GeneratorConfig(model="mistral", provider="ollama", temperature=0.5),
        )
        run_a = Run.create(
            dataset_name="ds",
            dataset_hash="a" * 64,
            dataset_size=10,
            pipeline_config=cfg_a,
        )
        run_b = Run.create(
            dataset_name="ds",
            dataset_hash="a" * 64,
            dataset_size=10,
            pipeline_config=cfg_b,
        )
        assert run_a.fingerprint.combined != run_b.fingerprint.combined

    def test_mark_running(self) -> None:
        """mark_running() should update status."""
        cfg = PipelineConfig(name="test")
        run = Run.create(
            dataset_name="test_ds",
            dataset_hash="a" * 64,
            dataset_size=100,
            pipeline_config=cfg,
        )
        run.mark_running()
        assert run.status == "running"

    def test_complete(self) -> None:
        """complete() should update status and metrics."""
        cfg = PipelineConfig(name="test")
        run = Run.create(
            dataset_name="test_ds",
            dataset_hash="a" * 64,
            dataset_size=100,
            pipeline_config=cfg,
        )
        run.complete(
            metrics={"precision": 0.85, "recall": 0.90},
            duration_ms=1500.0,
            cost={"total": 0.05},
        )
        assert run.status == "completed"
        assert run.metrics["precision"] == 0.85
        assert run.duration_ms == 1500.0
        assert run.cost == {"total": 0.05}
        assert run.error is None

    def test_fail(self) -> None:
        """fail() should update status and error."""
        cfg = PipelineConfig(name="test")
        run = Run.create(
            dataset_name="test_ds",
            dataset_hash="a" * 64,
            dataset_size=100,
            pipeline_config=cfg,
        )
        run.fail(error="Connection timeout")
        assert run.status == "failed"
        assert run.error == "Connection timeout"

    def test_to_dict(self) -> None:
        """to_dict() should export all fields."""
        cfg = PipelineConfig(name="test")
        run = Run.create(
            dataset_name="test_ds",
            dataset_hash="a" * 64,
            dataset_size=100,
            pipeline_config=cfg,
        )
        data = run.to_dict()
        assert data["run_id"] == run.run_id
        assert data["status"] == "pending"
        assert data["dataset_name"] == "test_ds"


# =============================================================================
# Trace
# =============================================================================


class TestTrace:
    """Tests for Trace model."""

    def test_create_hashes_query(self) -> None:
        """Trace.create() should hash query by default."""
        trace = Trace.create(query_text="What is Python?")
        assert trace.query_hash
        assert len(trace.query_hash) == 16
        assert trace.query_text is None  # PII-safe by default

    def test_create_with_store_text(self) -> None:
        """Trace.create() with store_text=True should keep text."""
        trace = Trace.create(query_text="What is Python?", store_text=True)
        assert trace.query_text == "What is Python?"
        assert trace.query_hash

    def test_set_answer_hashes_by_default(self) -> None:
        """set_answer() should hash answer by default."""
        trace = Trace.create(query_text="test")
        trace.set_answer(answer_text="Python is a programming language.")
        assert trace.answer_hash
        assert len(trace.answer_hash) == 16
        assert trace.answer_text is None

    def test_set_answer_with_store_text(self) -> None:
        """set_answer() with store_text=True should keep text."""
        trace = Trace.create(query_text="test")
        trace.set_answer(answer_text="Python is great.", store_text=True)
        assert trace.answer_text == "Python is great."

    def test_to_prometheus_labels(self) -> None:
        """to_prometheus_labels() should return low-cardinality labels."""
        trace = Trace.create(query_text="test")
        trace.model = "mistral"
        trace.tenant = "acme"
        trace.route = "/api/chat"

        labels = trace.to_prometheus_labels()
        assert labels == {
            "model": "mistral",
            "tenant": "acme",
            "route": "/api/chat",
        }

    def test_to_prometheus_labels_defaults(self) -> None:
        """to_prometheus_labels() should use defaults for missing values."""
        trace = Trace.create(query_text="test")
        labels = trace.to_prometheus_labels()
        assert labels["model"] == "unknown"
        assert labels["tenant"] == "default"
        assert labels["route"] == "default"


# =============================================================================
# Experiment
# =============================================================================


class TestExperiment:
    """Tests for Experiment model."""

    def test_create(self) -> None:
        """Experiment.create() should create valid experiment."""
        control_cfg = PipelineConfig(
            name="control",
            generator=GeneratorConfig(model="mistral", provider="ollama"),
        )
        treatment_cfg = PipelineConfig(
            name="treatment",
            generator=GeneratorConfig(model="llama3", provider="ollama"),
        )
        exp = Experiment.create(
            name="Model comparison",
            hypothesis="Llama3 produces more faithful answers",
            dataset_name="test_ds",
            dataset_hash="a" * 64,
            control_config=control_cfg,
            treatment_config=treatment_cfg,
        )
        assert exp.experiment_id
        assert exp.name == "Model comparison"
        assert exp.status == "draft"
        assert exp.control.name == "control"
        assert exp.treatment.name == "treatment"

    def test_variants_have_correct_configs(self) -> None:
        """Experiment variants should have their pipeline configs."""
        control_cfg = PipelineConfig(name="v1")
        treatment_cfg = PipelineConfig(name="v2")
        exp = Experiment.create(
            name="test",
            hypothesis="test",
            dataset_name="ds",
            dataset_hash="a" * 64,
            control_config=control_cfg,
            treatment_config=treatment_cfg,
        )
        assert exp.control.pipeline_config.name == "v1"
        assert exp.treatment.pipeline_config.name == "v2"


class TestExperimentVariant:
    """Tests for ExperimentVariant model."""

    def test_variant_tracks_run_id(self) -> None:
        """Variant should be able to track run_id."""
        cfg = PipelineConfig(name="test")
        variant = ExperimentVariant(name="control", pipeline_config=cfg)
        assert variant.run_id is None

        variant.run_id = "abc123"
        assert variant.run_id == "abc123"

    def test_variant_tracks_metrics(self) -> None:
        """Variant should track metrics after evaluation."""
        cfg = PipelineConfig(name="test")
        variant = ExperimentVariant(name="control", pipeline_config=cfg)
        variant.metrics = {"precision": 0.85, "recall": 0.90}
        assert variant.metrics["precision"] == 0.85
