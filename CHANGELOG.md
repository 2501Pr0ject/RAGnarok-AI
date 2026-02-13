# Changelog

All notable changes to RAGnarok-AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-02-13

### Added
- **More Integrations — 9 New Adapters**
  - **LLM Adapters (Cloud)**
    - `GroqLLM` — Fast inference via Groq API (OpenAI-compatible)
    - `MistralLLM` — Mistral AI models with embedding support
    - `TogetherLLM` — Open-source models via Together AI
  - **VectorStore Adapters**
    - `PineconeVectorStore` — Pinecone managed cloud vector database
    - `WeaviateVectorStore` — Weaviate cloud or self-hosted
    - `MilvusVectorStore` — Milvus self-hosted with HNSW indexing
    - `PgvectorVectorStore` — PostgreSQL with pgvector extension
  - **Framework Adapters**
    - `HaystackAdapter` — Haystack 2.x pipeline wrapper
    - `HaystackRetrieverAdapter` — Haystack retriever-only evaluation
    - `SemanticKernelAdapter` — Microsoft Semantic Kernel function wrapper
    - `SemanticKernelMemoryAdapter` — Semantic Kernel memory search evaluation
- **Medical Mode** (contributed by [@harish1120](https://github.com/harish1120))
  - `MedicalAbbreviationNormalizer` for reducing false positives in medical RAG evaluation
  - 350+ medical abbreviations across specialties (CHF, MI, COPD, DVT, etc.)
  - Context-aware disambiguation (same abbreviation, different meanings)
  - Integration with `LLMJudge` and `FaithfulnessEvaluator` via `medical_mode=True`
  - False positive filtering for explicit definitions
- **CLI Enhancements**
  - `ragnarok judge` command for standalone LLM-as-Judge evaluation
    - `--question`, `--answer`, `--context` parameters
    - `--criteria` selection (faithfulness, relevance, hallucination, completeness, all)
    - `--model` for custom Ollama model selection
    - `--json` output format support
  - `--config ragnarok.yaml` support for `evaluate` command
    - YAML-based configuration for reproducible evaluations
    - Supports all evaluate parameters (model, metrics, thresholds, etc.)
- **Performance Benchmarks**
  - `benchmarks/` directory with systematic performance tests
  - Retrieval metrics benchmarks (~24,000 queries/sec)
  - LLM-as-Judge benchmarks with timing data
  - Memory usage tracking
- **MkDocs Documentation Site**
  - Full documentation at `docs/`
  - API reference for adapters, evaluators, and types
  - Getting started guide
  - CI/CD integration guide
- **GitHub Action Reference**
  - Documentation for [`2501Pr0ject/ragnarok-evaluate-action`](https://github.com/2501Pr0ject/ragnarok-evaluate-action)

### Changed
- Updated `pyproject.toml` with 9 new optional dependencies
- Plugin registry now includes all new adapters
- Adapter classification updated (LOCAL/CLOUD for all new adapters)

## [1.3.1] - 2026-02-09

### Added
- **Jupyter Notebook Integration**
  - `ragnarok_ai.notebook` module for rich notebook display
  - `display_results()` — Rich HTML display for evaluation results
  - `display_comparison()` — Side-by-side pipeline comparison
  - `display_cost()` — Cost breakdown visualization
  - Metrics visualization with color-coded progress bars
  - Auto-detection of notebook environment (Jupyter, Colab, IPython)
  - Dark terminal theme for consistent appearance
- Google Colab quickstart link in README

### Changed
- Updated logo with dark background for better visibility

## [1.3.0] - 2026-02-09

### Added
- **Cost Tracking Module**
  - `ragnarok_ai.cost` module for token usage and cost tracking
  - `CostTracker` class with context manager support
  - `Pricing` class with rates for major providers:
    - OpenAI (GPT-4, GPT-4 Turbo, GPT-3.5 Turbo)
    - Anthropic (Claude 3 Opus, Sonnet, Haiku)
    - Groq (Llama, Mixtral)
    - Mistral (Small, Medium, Large)
    - Together AI (various models)
  - Token counting with tiktoken (fallback to character-based estimation)
  - `track_cost=True` parameter in `evaluate()` function
  - Formatted summary table output
  - JSON export for cost reports
  - Local providers (Ollama, vLLM) tracked as $0.00

### Changed
- LLM adapters now automatically track token usage when cost tracking is enabled

## [1.2.5] - 2026-02-08

### Added
- **Plugin Architecture**
  - Plugin system based on Python entry points (`ragnarok_ai.plugins` namespace)
  - `PluginRegistry` singleton for adapter discovery and management
  - Dynamic discovery of external plugins via `importlib.metadata`
  - `ragnarok plugins` CLI command:
    - `--list` — List all available plugins
    - `--info <name>` — Show detailed plugin information
    - `--local` / `--cloud` — Filter by adapter type
    - `--namespace <ns>` — Filter by namespace (llm, vectorstore, framework, evaluator)
  - Support for 4 plugin namespaces: `llm`, `vectorstore`, `framework`, `evaluator`
  - LOCAL/CLOUD classification for all built-in adapters
  - Plugin documentation (`docs/PLUGINS.md`)
  - E2E plugin test with mock package

### Changed
- All adapters now have `is_local: ClassVar[bool]` attribute for classification

## [1.2.0] - 2026-02-08

### Added
- **LLM-as-Judge with Prometheus 2**
  - `LLMJudge` class for multi-criteria evaluation
  - Faithfulness evaluation (is the answer grounded in context?)
  - Relevance evaluation (does the answer address the question?)
  - Hallucination detection (are there fabricated claims?)
  - Completeness evaluation (are all aspects covered?)
  - Rubric-based prompts with 1-5 scoring, normalized to 0-1
  - Detailed explanations for each judgment
  - `evaluate_all()` for comprehensive evaluation
  - `evaluate_batch()` for batch processing
- **Prometheus 2 Q5_K_M Integration**
  - Default model: `hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M`
  - ~5GB download, runs on 16GB RAM (M1/M2 compatible)
  - Automatic fallback to available models (mistral, llama3.2, etc.)
- **OllamaLLM `keep_alive` Support**
  - Prevents model unloading between requests
  - Default: 10 minutes (`keep_alive="10m"`)
  - Applied to both `generate()` and `embed()` methods

### Fixed
- **SyntheticQuestionGenerator**: Now handles both string and dict formats from LLM responses
- **JSON Parsing**: Robust parsing for incomplete JSON (missing closing brackets)
- **Integration Tests**: Excluded by default (`-m "not integration"` in pytest config)

### Changed
- Default pytest config now skips integration tests (run with `pytest -m integration`)

## [1.1.2] - 2025-01-27

### Fixed
- **`--json` flag now works** — v1.1.0/v1.1.1 required `--json-output`, now `--json` works
- `--json-output` kept as hidden alias for backward compatibility

## [1.1.1] - 2025-01-27 [YANKED]

### Note
- Incomplete fix for `--json` flag, use v1.1.2 instead

## [1.1.0] - 2025-01-27

### Added
- **CLI `generate` command**: Generate synthetic test sets from documents
  - `--docs`: Load documents from JSON file or directory
  - `--demo`: Use NovaTech demo dataset
  - `--model`: Select Ollama model (default: llama3.2)
  - `--seed`: Random seed for reproducibility
  - `--dry-run`: Preview what would be generated
  - Outputs `manifest.json` with version, config, timestamp for traceability
- **CLI `benchmark` command**: Track and compare evaluation runs over time
  - `--demo`: Run with demo data (3 simulated runs)
  - `--list`: List all benchmark configurations
  - `--history <config>`: Show history for a specific config
  - `--fail-under`: CI gating with minimum score threshold
  - `--dry-run`: Preview benchmark without running
  - `--output`: Save results to JSON file
  - Comparison table with baseline, deltas, and regression alerts
- **E2E tests**: Full CLI workflow tests (generate -> evaluate -> benchmark)
- **Trusted Publishing**: PyPI OIDC-based publishing (no API tokens)
- **Standardized JSON envelope**: All `--json` output now uses consistent format:
  - `command`: Command name (evaluate, generate, benchmark)
  - `status`: pass/fail/error/dry_run/success
  - `version`: CLI version for traceability
  - `data`: Command-specific payload
  - `errors`: Error messages (empty on success)

### Changed
- Publish workflow now runs tests before publishing
- Build verification with twine check before upload

## [1.0.2] - 2025-01-27

### Added
- **CLI `evaluate --demo`**: Working demo evaluation with realistic MockRAG
- **CI extras matrix**: Automated testing for all optional dependencies
- **STABILITY.md**: SemVer policy and compatibility matrix
- **Build verification**: Package build + twine check in CI

### Changed
- Classifier changed from "Production/Stable" to "Beta" (more honest)
- CLI commands `generate` and `benchmark` now show "Planned for v1.1"
- Updated README with working CLI examples

### Fixed
- CLI tests updated to match new behavior

## [1.0.1] - 2025-01-27

### Fixed
- Use absolute URLs for images (PyPI compatibility)
- Add noqa comments for e2e import tests (lint fix)

## [1.0.0] - 2025-01-27

### Added
- PyPI publication (`pip install ragnarok-ai`)
- End-to-end installation tests
- PyPI badge in README

### Changed
- Updated README with pip install instructions
- Updated project metadata (author, keywords, classifiers)
- Marked as Production/Stable

## [0.9.0] - 2025-01-26

### Added
- **Agent Evaluation Module**
  - `AgentResponse`, `AgentStep`, `ToolCall` types for agent pipelines
  - Tool-use correctness metrics (precision, recall, F1, success rate)
  - Multi-step reasoning evaluators (coherence, goal progress, efficiency)
  - `TrajectoryAnalyzer` for loop detection, dead-end detection, failure analysis
  - Trajectory visualization (ASCII for CLI, Mermaid for GitHub)
- **Agent Adapters**
  - `ReActParser` and `ReActAdapter` for ReAct pattern
  - `ChainOfThoughtAdapter` for CoT pattern
- Dual licensing (AGPL-3.0 + Commercial)

## [0.8.0] - 2025-01-25

### Added
- **Comparison & Benchmarking**
  - `compare()` function for side-by-side evaluation
  - `RegressionDetector` for quality drop alerts
  - `BenchmarkHistory` for time-series tracking
  - `DiffReport` for run-to-run comparison
- Baselines library with `BaselineResult` and `compare()`

## [0.7.0] - 2025-01-24

### Added
- **Framework Adapters**
  - LlamaIndex adapter (Retriever, QueryEngine, Index)
  - DSPy adapter (Retrieve, Module, RAG pattern)
- Custom RAG support via `RAGProtocol`
- Adapter contribution guide

## [0.6.0] - 2025-01-23

### Added
- **Cloud & Local LLM Adapters**
  - vLLM adapter for local high-performance inference
  - OpenAI adapter for optional cloud fallback
  - Anthropic adapter
- **Vector Store Adapters**
  - ChromaDB adapter
  - FAISS adapter (pure local, no server)

## [0.5.0] - 2025-01-22

### Added
- **Performance & Scale**
  - `MemoryCache` and `DiskCache` with `CacheProtocol`
  - `BatchEvaluator` for processing 1000+ queries
  - Async parallelization with `max_concurrency` parameter
  - Progress callbacks (sync and async)
  - Timeout and retry (`timeout`, `max_retries`, `retry_delay`)
  - Graceful cache error handling

## [0.4.0] - 2025-01-21

### Added
- **Framework Adapters**
  - LangChain integration
  - LangGraph integration
- **Observability**
  - OpenTelemetry export for tracing & debugging

## [0.3.0] - 2025-01-20

### Added
- **Test Generation**
  - Synthetic question generation from knowledge base
  - Multi-hop question support
  - Adversarial question generation
- Checkpointing system for crash recovery
- Golden set support (human-validated, versioned question sets)
- Baselines library (configs + expected results)
- NovaTech example dataset for quickstart

## [0.2.0] - 2025-01-19

### Added
- **Generation Metrics**
  - `FaithfulnessEvaluator` — Is the answer grounded in context?
  - `RelevanceEvaluator` — Does the answer address the question?
  - `HallucinationDetector` — Does the answer contain fabricated info?
- **Reporting**
  - HTML report with drill-down (failed questions, retrieved chunks)
- Qdrant vector store adapter
- Intelligent CI gating (stable metrics fail, LLM judgments warn)

## [0.1.0] - 2025-01-18

### Added
- **Core Foundation**
  - Core types (`Document`, `Query`, `TestSet`, `RAGResponse`)
  - Protocol definitions (`LLMProtocol`, `VectorStoreProtocol`, `EvaluatorProtocol`)
  - Custom exceptions
- **Retrieval Metrics**
  - `precision_at_k` — Relevant docs in top K results
  - `recall_at_k` — Coverage of relevant docs
  - `mrr` — Mean Reciprocal Rank
  - `ndcg_at_k` — Normalized Discounted Cumulative Gain
- **Adapters**
  - Ollama LLM adapter
- **Reporters**
  - Console reporter
  - JSON reporter
- Basic CLI
- CI/CD with GitHub Actions

[1.4.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.2.5...v1.3.0
[1.2.5]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.2.0...v1.2.5
[1.2.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/2501Pr0ject/RAGnarok-AI/releases/tag/v0.1.0
