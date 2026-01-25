"""Baseline configurations for ragnarok-ai.

This module provides pre-defined baseline configurations for RAG pipelines.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BaselineConfig(BaseModel):
    """A baseline configuration for RAG pipelines.

    Attributes:
        name: Unique name for the baseline.
        description: Description of what this config optimizes for.
        chunk_size: Document chunk size in tokens/characters.
        chunk_overlap: Overlap between chunks.
        embedder: Recommended embedding model.
        retrieval_k: Number of documents to retrieve.
        metadata: Additional configuration metadata.
    """

    name: str = Field(..., description="Baseline name")
    description: str = Field(..., description="What this config optimizes for")
    chunk_size: int = Field(..., description="Chunk size")
    chunk_overlap: int = Field(..., description="Chunk overlap")
    embedder: str = Field(..., description="Embedding model")
    retrieval_k: int = Field(default=10, description="Number of docs to retrieve")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional config")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for easy use.

        Returns:
            Configuration as dictionary.
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedder": self.embedder,
            "retrieval_k": self.retrieval_k,
            **self.metadata,
        }


# Pre-defined baseline configurations
BASELINE_CONFIGS: dict[str, BaselineConfig] = {
    "balanced": BaselineConfig(
        name="balanced",
        description="Balanced configuration for general-purpose RAG. Good trade-off between precision and speed.",
        chunk_size=512,
        chunk_overlap=50,
        embedder="nomic-embed-text",
        retrieval_k=10,
        metadata={
            "use_case": "general",
            "priority": "balance",
        },
    ),
    "precision": BaselineConfig(
        name="precision",
        description="High-precision configuration. Smaller chunks for more accurate retrieval at the cost of speed.",
        chunk_size=256,
        chunk_overlap=100,
        embedder="mxbai-embed-large",
        retrieval_k=15,
        metadata={
            "use_case": "accuracy-critical",
            "priority": "precision",
        },
    ),
    "speed": BaselineConfig(
        name="speed",
        description="Speed-optimized configuration. Larger chunks for faster processing with acceptable precision.",
        chunk_size=1024,
        chunk_overlap=0,
        embedder="nomic-embed-text",
        retrieval_k=5,
        metadata={
            "use_case": "latency-critical",
            "priority": "speed",
        },
    ),
    "memory_efficient": BaselineConfig(
        name="memory_efficient",
        description="Memory-efficient configuration for resource-constrained environments.",
        chunk_size=768,
        chunk_overlap=25,
        embedder="all-minilm",
        retrieval_k=5,
        metadata={
            "use_case": "low-resource",
            "priority": "memory",
        },
    ),
    "semantic": BaselineConfig(
        name="semantic",
        description="Semantic-focused configuration. Optimized for understanding context and meaning.",
        chunk_size=384,
        chunk_overlap=128,
        embedder="bge-large",
        retrieval_k=10,
        metadata={
            "use_case": "semantic-search",
            "priority": "understanding",
        },
    ),
}


def get_baseline_config(name: str) -> BaselineConfig:
    """Get a baseline configuration by name.

    Args:
        name: Name of the baseline (balanced, precision, speed, etc.).

    Returns:
        The baseline configuration.

    Raises:
        KeyError: If baseline name is not found.
    """
    if name not in BASELINE_CONFIGS:
        available = ", ".join(BASELINE_CONFIGS.keys())
        msg = f"Unknown baseline '{name}'. Available: {available}"
        raise KeyError(msg)
    return BASELINE_CONFIGS[name]


def list_baselines() -> list[str]:
    """List all available baseline names.

    Returns:
        List of baseline names.
    """
    return list(BASELINE_CONFIGS.keys())
