# Changelog

All notable changes to RAGnarok-AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2025-01-27

### Fixed
- **`--json` flag now works** — v1.1.0 required `--json-output`, now both work
- `--json-output` kept as hidden alias for backward compatibility

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
- **E2E tests**: Full CLI workflow tests (generate → evaluate → benchmark)
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
