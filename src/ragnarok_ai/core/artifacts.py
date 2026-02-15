"""
Unified artifact schema for ragnarok-ai.

This module defines central data contracts for:
- Runs (evaluation executions)
- Experiments (A/B comparisons)
- Traces (production events)
- Fingerprints (reproducibility)

Design goals:
- Deterministic hashing via canonical JSON
- Full sha256 stored; short form only for display
- Pydantic v2 models (project depends on pydantic>=2.0)
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

# =============================================================================
# Hashing helpers (canonical JSON)
# =============================================================================


def _canonical_json(data: Any) -> str:
    """
    Stable JSON serialization used for fingerprints.

    Assumes `data` is already JSON-serializable (dict/list/str/int/float/bool/None).
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(text: str) -> str:
    """Compute full SHA256 hex digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_hex_short(text: str, n: int = 16) -> str:
    """Compute truncated SHA256 hex digest for display."""
    return _sha256_hex(text)[:n]


# =============================================================================
# Environment & Fingerprints
# =============================================================================


class Environment(BaseModel):
    """Runtime environment information (mostly for reproducibility + debugging)."""

    model_config = {"frozen": True, "extra": "forbid"}

    python_version: str
    platform: str
    ragnarok_version: str
    packages: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def capture(cls) -> Environment:
        """Capture current environment."""
        ragnarok_version = "unknown"
        try:
            from importlib.metadata import version as pkg_version

            ragnarok_version = pkg_version("ragnarok-ai")
        except Exception:
            try:
                import ragnarok_ai

                ragnarok_version = getattr(ragnarok_ai, "__version__", "unknown")
            except Exception:
                ragnarok_version = "unknown"

        return cls(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            ragnarok_version=ragnarok_version,
            packages=cls._get_relevant_packages(),
        )

    @staticmethod
    def _get_relevant_packages() -> dict[str, str]:
        """
        Keep this list short (avoid huge env payloads).

        Add packages that can affect eval logic.
        """
        pkgs = ["pydantic", "pydantic-settings", "httpx", "typer"]
        out: dict[str, str] = {}
        try:
            from importlib.metadata import version as pkg_version

            for p in pkgs:
                try:
                    out[p] = pkg_version(p)
                except Exception:
                    continue
        except Exception:
            pass
        return out

    def fingerprint(self) -> str:
        """Deterministic env fingerprint (full sha256 hex)."""
        payload = self.model_dump(mode="json")
        return _sha256_hex(_canonical_json(payload))


class Fingerprint(BaseModel):
    """
    Reproducibility fingerprint: dataset + config + env.

    Store full sha256. Use `.short` for human display.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    dataset_hash: str  # full sha256 hex recommended
    config_hash: str  # full sha256 hex recommended
    env_hash: str  # full sha256 hex recommended

    @property
    def combined(self) -> str:
        """Full sha256 for storage."""
        payload = f"{self.dataset_hash}:{self.config_hash}:{self.env_hash}"
        return _sha256_hex(payload)

    @property
    def short(self) -> str:
        """Short form for display (16 chars)."""
        return self.combined[:16]

    def __str__(self) -> str:
        return self.short


# =============================================================================
# Pipeline Configuration (hashable, deterministic)
# =============================================================================


class RetrieverConfig(BaseModel):
    """Retriever configuration."""

    model_config = {"frozen": True, "extra": "forbid"}

    type: str = "unknown"
    embedding_model: str | None = None
    top_k: int = 10
    similarity_threshold: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        """Compute config fingerprint (full sha256)."""
        payload = self.model_dump(mode="json", exclude={"metadata"})
        return _sha256_hex(_canonical_json(payload))


class GeneratorConfig(BaseModel):
    """LLM generator configuration."""

    model_config = {"frozen": True, "extra": "forbid"}

    model: str
    provider: str = "unknown"
    temperature: float = 0.0
    max_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        """Compute config fingerprint (full sha256)."""
        payload = self.model_dump(mode="json", exclude={"metadata"})
        return _sha256_hex(_canonical_json(payload))


class JudgeConfig(BaseModel):
    """LLM-as-Judge configuration."""

    model_config = {"frozen": True, "extra": "forbid"}

    model: str = "prometheus-7b-v2.0"
    provider: str = "ollama"
    criteria: list[str] = Field(default_factory=lambda: ["faithfulness", "relevance"])
    medical_mode: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        """Compute config fingerprint (full sha256)."""
        payload = self.model_dump(mode="json", exclude={"metadata"})
        return _sha256_hex(_canonical_json(payload))


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    model_config = {"frozen": True, "extra": "forbid"}

    name: str = "default"
    retriever: RetrieverConfig | None = None
    generator: GeneratorConfig | None = None
    judge: JudgeConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        """Compute pipeline fingerprint (full sha256, includes all sub-configs)."""
        payload = self.model_dump(mode="json", exclude={"metadata"})
        return _sha256_hex(_canonical_json(payload))


# =============================================================================
# Run (Evaluation Execution)
# =============================================================================


RunStatus = Literal["pending", "running", "completed", "failed"]


class Run(BaseModel):
    """
    A complete evaluation run with full traceability.

    Mutable on purpose: status/metrics filled as evaluation progresses.
    """

    model_config = {"extra": "forbid", "validate_assignment": True}

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: RunStatus = "pending"

    fingerprint: Fingerprint

    pipeline_config: PipelineConfig
    environment: Environment

    dataset_name: str
    dataset_size: int
    dataset_hash: str

    metrics: dict[str, float] = Field(default_factory=dict)
    cost: dict[str, Any] | None = None
    duration_ms: float | None = None
    error: str | None = None

    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        dataset_name: str,
        dataset_hash: str,
        dataset_size: int,
        pipeline_config: PipelineConfig,
        include_env_in_fingerprint: bool = True,
    ) -> Run:
        """Create a new run with auto-generated fingerprint."""
        env = Environment.capture()
        env_hash = env.fingerprint() if include_env_in_fingerprint else _sha256_hex("no-env")
        fp = Fingerprint(
            dataset_hash=dataset_hash,
            config_hash=pipeline_config.fingerprint(),
            env_hash=env_hash,
        )
        return cls(
            fingerprint=fp,
            pipeline_config=pipeline_config,
            environment=env,
            dataset_name=dataset_name,
            dataset_size=dataset_size,
            dataset_hash=dataset_hash,
        )

    def mark_running(self) -> None:
        """Mark run as running."""
        self.status = "running"
        self.updated_at = datetime.now(timezone.utc)

    def complete(
        self,
        *,
        metrics: dict[str, float],
        duration_ms: float,
        cost: dict[str, Any] | None = None,
    ) -> None:
        """Mark run as completed with results."""
        self.status = "completed"
        self.metrics = metrics
        self.duration_ms = duration_ms
        self.cost = cost
        self.error = None
        self.updated_at = datetime.now(timezone.utc)

    def fail(self, *, error: str) -> None:
        """Mark run as failed."""
        self.status = "failed"
        self.error = error
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return self.model_dump(mode="json")


# =============================================================================
# Trace (Production Event)
# =============================================================================


class Trace(BaseModel):
    """
    A lightweight production event for monitoring.

    Prefer storing hashes; text fields are optional (PII).
    """

    model_config = {"extra": "forbid", "validate_assignment": True}

    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    query_hash: str
    query_text: str | None = None

    answer_hash: str | None = None
    answer_text: str | None = None

    doc_ids: list[str] = Field(default_factory=list)
    context_length: int = 0

    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None

    # If sampled + evaluated
    faithfulness: float | None = None
    relevance: float | None = None

    model: str | None = None
    model_version: str | None = None
    pipeline_fingerprint: str | None = None

    tenant: str | None = None
    route: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(cls, *, query_text: str | None, store_text: bool = False) -> Trace:
        """Create a new trace (PII-safe by default)."""
        q_text = query_text or ""
        q_hash = _sha256_hex_short(q_text, n=16) if q_text else _sha256_hex_short("empty", n=16)
        return cls(
            query_hash=q_hash,
            query_text=q_text if store_text else None,
        )

    def set_answer(self, *, answer_text: str | None, store_text: bool = False) -> None:
        """Set answer (PII-safe by default)."""
        if answer_text is None:
            self.answer_hash = None
            self.answer_text = None
            return
        self.answer_hash = _sha256_hex_short(answer_text, n=16)
        self.answer_text = answer_text if store_text else None

    def to_prometheus_labels(self) -> dict[str, str]:
        """Export as Prometheus labels (low cardinality)."""
        return {
            "model": self.model or "unknown",
            "tenant": self.tenant or "default",
            "route": self.route or "default",
        }


# =============================================================================
# Experiment (A/B Testing)
# =============================================================================


ExperimentStatus = Literal["draft", "running", "completed", "cancelled"]


class ExperimentVariant(BaseModel):
    """A variant in an experiment."""

    model_config = {"extra": "forbid", "validate_assignment": True}

    name: str
    pipeline_config: PipelineConfig
    run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class Experiment(BaseModel):
    """An A/B experiment comparing pipeline variants."""

    model_config = {"extra": "forbid", "validate_assignment": True}

    experiment_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    hypothesis: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: ExperimentStatus = "draft"

    dataset_name: str
    dataset_hash: str

    control: ExperimentVariant
    treatment: ExperimentVariant

    winner: str | None = None
    confidence: float | None = None
    summary: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        name: str,
        hypothesis: str,
        dataset_name: str,
        dataset_hash: str,
        control_config: PipelineConfig,
        treatment_config: PipelineConfig,
    ) -> Experiment:
        """Create a new experiment."""
        return cls(
            name=name,
            hypothesis=hypothesis,
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
            control=ExperimentVariant(name="control", pipeline_config=control_config),
            treatment=ExperimentVariant(name="treatment", pipeline_config=treatment_config),
        )
