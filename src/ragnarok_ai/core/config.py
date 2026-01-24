"""Configuration management for ragnarok-ai.

This module provides configuration classes using pydantic-settings
for environment variable management and validation.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be configured via environment variables with
    the RAGNAROK_ prefix.

    Attributes:
        ollama_base_url: Base URL for Ollama API.
        ollama_model: Default model to use with Ollama.
        qdrant_url: URL for Qdrant vector store.
        qdrant_api_key: Optional API key for Qdrant.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        batch_size: Default batch size for processing.
        timeout_seconds: Default timeout for operations.

    Example:
        >>> # Set via environment variables:
        >>> # export RAGNAROK_OLLAMA_MODEL=mistral
        >>> # export RAGNAROK_LOG_LEVEL=DEBUG
        >>>
        >>> settings = Settings()
        >>> print(settings.ollama_model)
        'mistral'

    Environment Variables:
        RAGNAROK_OLLAMA_BASE_URL: Ollama API URL (default: http://localhost:11434)
        RAGNAROK_OLLAMA_MODEL: Default Ollama model (default: mistral)
        RAGNAROK_QDRANT_URL: Qdrant URL (default: http://localhost:6333)
        RAGNAROK_QDRANT_API_KEY: Qdrant API key (optional)
        RAGNAROK_LOG_LEVEL: Logging level (default: INFO)
        RAGNAROK_BATCH_SIZE: Batch size (default: 10)
        RAGNAROK_TIMEOUT_SECONDS: Timeout in seconds (default: 30.0)
    """

    model_config = SettingsConfigDict(
        env_prefix="RAGNAROK_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API",
    )
    ollama_model: str = Field(
        default="mistral",
        description="Default model to use with Ollama",
    )

    # Qdrant settings
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="URL for Qdrant vector store",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Optional API key for Qdrant",
    )

    # General settings
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default batch size for processing",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Default timeout for operations in seconds",
    )
