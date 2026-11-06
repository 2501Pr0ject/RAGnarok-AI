# Changelog

All notable changes to RAGnarok-AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI: the lint job fails when a tracked file contains Git conflict markers
  (`scripts/check_conflict_markers.sh`, usable locally)
- **Common judge contract, data types only** (`ragnarok_ai.judges`): native
  outputs (`OrdinalScore`, `BinaryProbability`, `ClassProbabilities`, each
  projecting to a score in [0, 1]), verdict policies (`OrdinalBands`,
  `ScoreThreshold`, `LabelMap`; the verdict is always computed from the
  output, never read from a model's text; unsupported information is a
  failure, never PARTIAL), `JudgeConfig` (everything that makes two runs
  comparable, with a canonical `config_hash` and `comparable_with()`),
  `CallTrace`, `UnitJudgment`, `AggregationPolicy`, `JudgeFailure` and
  `JudgeAttempt` (a result or a failure, never both). Nothing is wired to
  these types yet; `JudgeResult`, `JudgeResults` and `LLMJudge` are unchanged
- **`LLMResponse`** (`ragnarok_ai.core`): the generic shape of a detailed
  generation (`text`, `prompt_tokens`, `completion_tokens`, `model`,
  `options`, `latency_ms`). `LLMProtocol` does not require it; the judge
  reads the counters from any client whose `generate_detailed()` returns an
  object with `text`, `prompt_tokens` and `completion_tokens`, and falls
  back to `generate()` otherwise
- **`OllamaGeneration.latency_ms`** (last field, optional) and the
  `prompt_tokens` / `completion_tokens` properties mirroring
  `prompt_eval_count` / `eval_count`; existing fields and positional
  construction are unchanged
- **`LLMJudge.model_digest`**: the Ollama digest of the model in use, read
  once from the same `/api/tags` answer that lists the installed models;
  `None` for a client passed with `llm=`

## [1.12.3] - 2026-11-06

### Behavior changes
- **Verdicts can change for the `Score: N` prompt format** (#135). The
  verdict is now always derived from the score (4-5 PASS, 3 PARTIAL, 1-2
  FAIL); a model that wrote a `Verdict:` line contradicting its score used
  to produce that verdict. What to do: nothing for new runs; stored results
  can be realigned with `verdict_from_raw_score(raw_score)`
- **Batch-evaluation checkpoints written before 1.12.3 are refused on
  resume** (#136). `CheckpointData.format_version` is now 2; resuming a
  format 1 checkpoint (stored by batch number, without the data needed to
  rebuild results) raises `CheckpointFormatError`. What to do: finish the
  batch evaluations in progress before upgrading, or run them again from
  the start after upgrading, then delete the old checkpoint files
- **Calibration statistics are computed on the scored samples only**
  (#137). `CriterionCalibration.n` is the number of samples with a usable
  judge score; `n_labeled` adds the parse failures, which no longer abort
  the run. What to do: read `n_labeled` where the submitted count was
  expected; a report with `parse_failures > 0` covers fewer samples than
  were labeled
- **No more model fallback** (#139). When the requested model, explicit
  or default, is not installed, `LLMJudge` raises `JudgeModelNotFoundError`
  instead of silently using an installed `mistral`, `llama3.2`, `llama3`
  or `llama2` model. The `FALLBACK_MODELS` and `PROMETHEUS_ALIASES` class
  attributes and `_find_fallback_model()` are removed. What to do: install
  the model you ask for (`ollama pull <model>`; the default is in the
  error message); `mistral` still resolves to `mistral:latest`, and a
  Prometheus 2 alias created with `ollama cp` is still accepted
- **Judge scores are not comparable with runs made before 1.12.3**
  (#141). The Ollama client built by `LLMJudge` now sends `temperature=0`,
  `seed=42` and `num_ctx=8192`; earlier versions sent no option, so the
  judge sampled at the server default temperature 0.8 with a random seed
  in a 4096-token window on recent Ollama versions. What to do: run your
  baselines again with 1.12.3 before comparing; to keep other options,
  build the client yourself (`OllamaLLM(temperature=..., seed=...,
  num_ctx=...)`) and pass it with `llm=`, it is used as given

### Added
- **`LLMJudge(prompt_format=...)`**: `"prometheus"` forces the prometheus-eval
  template and the `[RESULT] N` parser, `"score-n"` forces the `Score: N`
  prompts, `"auto"` (default) keeps the detection by model name. For a
  Prometheus 2 model served under another name (`ollama cp` alias, gateway)
  or a client without a `model` attribute. The format in use is
  `judge.effective_prompt_format`, logged at INFO. CLI: `ragnarok judge
  --prompt-format`; the JSON output gains `model` and `prompt_format`.
  Closes #133
- **`OllamaLLM` sampling options**: `temperature`, `seed`, `num_ctx` and a
  free `options` dict, sent in the request `options`. By default nothing is
  sent and the adapter behaves exactly as before (server defaults:
  temperature 0.8, random seed, the server's context window). New
  `generate_detailed()` returns the text with Ollama's `prompt_eval_count`
  and `eval_count` (`OllamaGeneration`); `generate()` is unchanged
- **Truncation warning in `LLMJudge`**: Ollama truncates silently a prompt
  longer than `num_ctx` and reports a token count just above `num_ctx / 2`
  (measured on Ollama 0.40.13). The judge recognizes that signature in the
  counters, logs a warning naming the counts, and counts it in
  `judge.truncation_warnings`. Heuristic: a legitimate prompt whose length
  falls in that narrow band is a false positive

### Fixed
- **`parse_judge_response()` let a `Verdict:` line written by the model
  override the score**: "Score: 2, Verdict: PASS" gave PASS with a score of
  0.25. The verdict now follows the score only, for both prompt formats;
  the rule is exposed as `verdict_from_raw_score()`
- **`BatchEvaluator` resume lost the results of the batches already done**
  - `evaluate(testset, resume_from=...)` skipped the completed batches but
    never read their results back from the checkpoint: the returned
    `EvaluationResult` only covered the batches evaluated after the resume,
    and the checkpoint was deleted at the end, so those results were lost
  - Checkpoints are now keyed by query (`metadata["id"]`, or a hash of the
    query text with an occurrence suffix for duplicates), not by batch
    number: a resumed run restores the result of every query already
    evaluated, whatever the `batch_size`, and evaluates only the remaining
    ones. Two queries with the same explicit `metadata["id"]` are refused
    with a `ValueError` naming the id and both positions; the occurrence
    suffix is reserved to keys derived from the text. Queries that failed before the interruption are evaluated again.
    The first yielded `BatchResult` carries the restored results
    (`restored=True`)
  - `get_checkpoint_id()` returns the checkpoint used by the last evaluation
    (it always returned `None`)
  - With `checkpoint_interval > 1`, results were only kept for the batch
    that triggered the save; every batch is now kept until the next save,
    and the last batch is always saved
- **`JudgeCalibrator` aborted at the first unparseable judge response**
  - Since 1.12.2 an unparseable response raises `EvaluationError`; the
    calibrator did not catch it, so one bad response lost the whole run
  - The calibration now continues: the sample is counted in
    `CriterionCalibration.parse_failures` (indices in `failed_indices`),
    excluded from the statistics, and reported by `summary()` and `to_dict()`
- **`LLMJudge()` said "no suitable judge model" whatever went wrong**
  - A stopped Ollama server, a timeout or an HTTP error all ended in the
    same `RuntimeError` asking to install Prometheus 2; on a base install
    without the `[ollama]` extra, `LLMJudge()` failed with a bare
    `ModuleNotFoundError: No module named 'httpx'`
  - `LLMJudge()` now raises one of three errors, each naming the remedy:
    `JudgeDependencyError` with `pip install 'ragnarok-ai[ollama]'` when
    httpx is missing; `JudgeConnectionError` with the server URL when
    Ollama cannot be reached or answers an error status;
    `JudgeModelNotFoundError` with `ollama pull <model>` and the installed
    models when the server is up but does not serve the requested model.
    All three are also `RuntimeError`, so `except RuntimeError` written
    for earlier versions keeps working (`JudgeDependencyError` is also an
    `ImportError`, `JudgeConnectionError` also an `LLMConnectionError`)
- **`LLMJudge` silently replaced the requested model by another one**
  - When the requested model (explicit or default) was not installed, the
    constructor picked any installed `mistral`, `llama3.2`, `llama3` or
    `llama2` model with only a log warning. Scores then came from a
    different judge than the one named in the code, with the `Score: N`
    prompt format, and runs were not comparable
  - The generic fallback is removed (see Behavior changes). Two resolutions
    remain: `mistral` resolves to `mistral:latest`, and a Prometheus 2 model
    requested under a name that is not installed resolves to an installed
    model whose name contains "prometheus" (an `ollama cp` alias), with an
    INFO log

### Changed
- `CalibrationReport.summary()` states that the recommended threshold and
  its kappa are measured on the same samples that selected them
  (in-sample, optimistic), and asks for confirmation on held-out labels;
  `CalibrationReport.insufficient_data` also considers the scored count

## [1.12.2] - 2026-11-04

### Fixed
- **`LLMJudge` with Prometheus 2 (the default judge) returned no real score**
  - The judge prompts ask for a `Score: N` line, which Prometheus 2 does not
    follow: it answers in its own format, `(feedback) [RESULT] N`, and
    sometimes gives no number at all. The parser then silently used a
    default score of 3 (0.5 normalized, verdict `PARTIAL`)
  - **Prometheus scores produced by earlier versions are not usable and must
    be discarded**: they were the default 3 whenever the output did not
    contain `Score: N`, which is indistinguishable from a real 3
  - Prometheus 2 models (detected by model name) now get the official
    prometheus-eval absolute-grading template without reference answer
    (`ABS_SYSTEM_PROMPT`, `ABSOLUTE_PROMPT_WO_REF`, `SCORE_RUBRIC_TEMPLATE`,
    copied verbatim from prometheus-eval 0.1.20, Apache License 2.0) and a
    `[RESULT] N` parser
  - The Prometheus rubrics reuse, word for word, the question and the five
    score descriptions of the `Score: N` prompts, for all four criteria
    (faithfulness, hallucination, relevance, completeness); a test enforces it
  - Other judge models keep the `Score: N` format, unchanged
- **A judge response that cannot be parsed is now an explicit error**
  (`EvaluationError`), for both formats, instead of a default score

## [1.12.1.post1] - 2026-11-03

Documentation-only post-release (PEP 440): the code is identical to 1.12.1.
It publishes the updated README on PyPI.

### Documentation
- **README no longer names or compares itself to other evaluation
  frameworks**: the Design Principles introduction now only introduces the
  project's own guarantees

## [1.12.1] - 2026-10-30

### Fixed
- **`RecursionError` on `from ragnarok_ai.adapters import ...` (regression
  in 1.12.0)**
  - In 1.12.0, importing an LLM adapter or an LLM classification tuple from
    the `ragnarok_ai.adapters` package in a fresh interpreter
    (`from ragnarok_ai.adapters import OllamaLLM`, `OpenAILLM`, ...,
    `LOCAL_LLM_ADAPTERS`, `CLOUD_LLM_ADAPTERS`) raised `RecursionError`,
    even with httpx installed; any unknown attribute of the package did the
    same instead of raising `AttributeError`
  - Cause: the lazy `__getattr__` added in 1.12.0 started with
    `from ragnarok_ai.adapters import llm`, which looks `llm` up on the
    package and re-entered `__getattr__` endlessly while the submodule was
    not imported yet
  - Fix: the requested name is checked first and the submodule is loaded
    with `importlib.import_module`. `from ragnarok_ai.adapters.llm import ...`
    was not affected
  - `ragnarok_ai.adapters.llm` and its adapter modules
    (`ragnarok_ai.adapters.llm.ollama`, ...) are reachable as attributes
    again, as in 1.11
  - Why tests missed it: other test modules imported
    `ragnarok_ai.adapters.llm` first, so the full suite passed

### Tests
- **Public import paths checked in a fresh interpreter**, with and without
  httpx (a plain `ImportError` is expected without it), so state cached by
  other tests can no longer hide import bugs
- **README examples test run in isolation in CI**
- **Random test order in CI** with pytest-randomly (new dev dependency); the
  seed is the workflow run id, so failures reproduce with
  `--randomly-seed=<id>`. Sixteen seeds surfaced no other order dependency

## [1.12.0] - 2026-10-30

**Bug-fix release.** No new feature: the minor version bump is required by
Semantic Versioning only, because the fixes add backward-compatible public
API (`PluginUnavailableError`, `PluginInfo.available` /
`PluginInfo.requires_extra`, `list_adapters(available_only=...)`) and two
core dependencies.

### Fixed
- **The `ragnarok` CLI works on a base install** (`pip install ragnarok-ai`)
  - `typer` and `rich` are now core dependencies; previously `typer` was
    only in the `[cli]` extra, so the installed `ragnarok` command crashed
    with `ModuleNotFoundError: No module named 'typer'`
  - The `[cli]` extra is kept for backward compatibility
  - Core lower bounds verified and corrected: `typer>=0.16` (older versions
    break `--help` with click 8.2+), `rich>=12.3` (progress columns of the
    live display), `pydantic>=2.0.1` (2.0 rejects a `model_version` field)
- **`ragnarok plugins` no longer crashes without optional dependencies**
  - Builtin adapters whose dependencies are missing are listed as
    unavailable with the extra to install
    (`unavailable: pip install 'ragnarok-ai[ollama]'`); `--json` and
    `--info` expose `available` and `requires_extra`
  - `ragnarok_ai.adapters` and `ragnarok_ai.adapters.llm` resolve the LLM
    adapters on first access, so importing them no longer requires `httpx`
  - New `PluginInfo` fields `available` and `requires_extra`, and
    `list_adapters(available_only=True)`
- **`ragnarok generate` and `ragnarok judge` without the `[ollama]` extra**
  exit with an install hint instead of a traceback (`generate --dry-run`
  still works)
- **No more pydantic warning on import**: with pydantic < 2.10, `Trace` and
  `TraceEvent` emitted a `UserWarning` because their `model_version` field
  falls in pydantic's protected `model_` namespace; both models now set
  `protected_namespaces=()`

### Changed
- **`PluginInfo.adapter_class` can be `None`** (type `type | None`) for a
  builtin adapter whose optional dependencies are not installed. Requesting
  such an adapter with `PluginRegistry.get_adapter()` raises the new
  `PluginUnavailableError`, a subclass of `ImportError` (and of
  `PluginError`) carrying the install command. With all dependencies
  installed, behavior is unchanged; code type-checked with mypy that uses
  `adapter_class` directly may need a `None` check

### CI
- **Install smoke test**: the wheel is installed with pip in a clean
  environment without extras, then `ragnarok --version`,
  `ragnarok evaluate --demo` and `ragnarok plugins --list` must succeed
- **Minimum versions test**: core dependencies (pydantic, pydantic-settings,
  pyyaml, typer, rich) installed at their declared lower bounds
  (`uv pip install --resolution lowest-direct`); the package must import
  with `-W error::UserWarning`, then `ragnarok --version` and
  `ragnarok evaluate --demo` must succeed; optional extras are not covered
- **uv pinned to 0.13.6** in every workflow (previously the latest release);
  `uv.lock` regenerated with it (lock format revision 1 to 5, no package
  version change)

### Notes
- The base install still works in Pyodide 0.27.3 via `micropip`, now
  including the CLI dependencies

## [1.11.1] - 2026-10-30

### Fixed
- **`evaluate()` sequential mode now fills `query_results`**
  - With `max_concurrency=1` (the default), `EvaluationResult.query_results`
    was always empty, `errors` stayed empty with `fail_fast=False`, and
    `on_progress` received `result=None`; only the parallel mode populated
    them. Both modes now return the same per-query details
  - Impact with the default sequential mode:
    - `RAGDiagnostician.diagnose()` returned an empty diagnosis
      (`total_queries=0`, no failure causes, no recommendations)
    - `BenchmarkHistory` and regression detection recorded latency from
      the total wall-clock time instead of per-query latencies; quality
      metrics (precision, recall, MRR, NDCG) were not affected
  - `evaluate_stream()` public behavior unchanged, now covered by contract
    tests (yielded tuples, skipped failures with `fail_fast=False`,
    `EvaluationError` with `fail_fast=True`, cached results)

### Documentation
- **README examples aligned with the real API**
  - Removed APIs that never existed: `generate_testset()`,
    `evaluate(metrics=..., llm=...)`, `results.export()`, `benchmark()`,
    `results.log_to()`, `examples/basic_evaluation.py`
  - HTML report via `HTMLReporter.report_to_file()`, configuration
    comparison via `compare()`, tracking over time via `BenchmarkHistory`
  - Same `metrics=` fix in the diagnosis and production test set guides
- **README feature descriptions aligned with the code**: checkpointing is
  described as what it is (`BatchEvaluator`, per completed batch);
  benchmark figures labeled with the version measured; navigation anchors
  fixed; version banner removed
- **Data sovereignty wording**: the README roadmap and the air-gapped guide
  no longer claim GDPR/HIPAA compliance; they describe the actual guarantee
  (data never leaves your infrastructure)
- **README examples tested in CI**: every `python` block of the README is
  executed against the library (offline, scripted judge), so documentation
  drift fails the build

### Notes
- Since 1.11.0, `ragnarok evaluate --testset` requires `--pipeline
  module:attribute` (the pipeline to evaluate); without it the command
  exits with code 2. `--demo` is unchanged

## [1.11.0] - 2026-09-04

### Added
- **Real test set evaluation in the CLI**
  - `ragnarok evaluate --testset t.json --pipeline module:attribute` runs
    a real evaluation against your own RAG pipeline (an object
    implementing `RAGProtocol`, or a zero-argument factory) — previously
    only `--demo` worked
  - Streaming evaluation surfaced end to end: results appear as each
    query completes, with running metric averages (built on the existing
    `evaluate_stream` engine)
  - Live terminal panel (rich): progress, running Precision/Recall/MRR/
    NDCG, latest queries, pass/warn/fail counts; plain line-per-query
    fallback in CI logs; `--live/--no-live` toggle
  - `--k` option; per-query rows included in `--output` JSON
  - Readable errors for bad pipeline specs, missing files, and failures
- **Interactive results viewer (TUI)**
  - `ragnarok view results.json`: browse per-query results in a Textual
    app — sortable table with PASS/WARN/FAIL status, detail panel for
    the selected query, `f`/`a` to filter failing queries
  - New optional extra: `pip install ragnarok-ai[tui]` (textual);
    graceful message when not installed

### Changed
- README repositioned: leads with what ragnarok-ai guarantees (local-first,
  private, resumable, measured trust) instead of comparisons against other
  tools; competitor-specific claims removed

## [1.10.0] - 2026-09-04

### Added
- **Judge Calibration**
  - `JudgeCalibrator`: measure judge-vs-human agreement on a labeled
    sample set, per criterion (faithfulness, relevance, hallucination,
    completeness)
  - Statistics without new dependencies: Cohen's kappa with Landis &
    Koch interpretation, 95% Wilson confidence interval on agreement,
    false accept/reject rates split by direction
  - Threshold sweep recommending the pass cutoff that maximizes kappa
    on your labels (ties keep the current threshold)
  - `CalibrationSample` (with per-criterion human label overrides) and
    `CalibrationSet` with JSON save/load for versionable labeled sets
  - `insufficient_data` flag below 20 labels; disagreement indices for
    review; public exports from package root
  - Documentation at `docs/user-guide/calibration.md`
- **Production-Mined Testsets**
  - `TestsetMiner`: build evaluation test sets from real production
    traffic — strategies `frequent` (most-asked), `failures` (highest
    failure rate), `slow` (highest latency); dedup by query hash,
    production stats (frequency, failure rate, latency, first/last seen)
    in query metadata
  - Opt-in query capture: `MonitorClient(capture_queries=True)` records
    query text scrubbed of inline PII client-side (emails, IPs, SSN/card
    numbers, home paths) via the new `privacy.scrub_text()`
  - `TraceEvent.query_text` (nullable) with automatic SQLite schema
    migration for existing monitor databases
  - Public export from package root; documentation at
    `docs/user-guide/production-testsets.md`

## [1.9.0] - 2026-09-03

### Added
- **Root-Cause Diagnosis**
  - `RAGDiagnostician`: classify every failing query of an evaluation into
    a failure cause — `pipeline_error`, `retrieval_miss`,
    `retrieval_ranking`, `context_insufficient`,
    `generation_hallucination`, `generation_incomplete`
  - Two tiers: heuristic (free, from metrics + ground truth vs. retrieved
    doc IDs) and LLM-assisted (opt-in closed YES/NO checks on the
    generation side, requires the document corpus)
  - `DiagnosisReport`: cause breakdown, failure-rate patterns per query
    metadata (e.g. by question type), recommendations ordered by dominant
    cause, `summary()` and `to_dict()`
  - `QueryResult.retrieved_doc_ids`: evaluation now records retrieved
    document IDs (backward-compatible; round-tripped by the disk cache)
  - Public exports from package root; documentation at
    `docs/user-guide/diagnosis.md`
- **Live A/B Testing**
  - `Experiment`: deterministic, stateless traffic splitting between named
    variants (hash-based bucketing with configurable weights and salt);
    `tag()` records assignments into monitor trace metadata
  - `ABAnalyzer`: compare two variants from monitor traces
    - Success rate via two-proportion z-test, mean latency via Welch's
      t-test (normal approximation, no scipy dependency)
    - Overall winner only when a variant is significantly better on at
      least one metric and worse on none
    - `min_samples` guard flags underpowered windows as `insufficient_data`
  - `ABTestConfig` (alpha, min_samples), `ABTestReport.to_dict()` export
  - Public exports from package root; documentation at
    `docs/user-guide/ab-testing.md`
- **Production Drift Detection**
  - `build_baseline()`: record a self-contained, PII-free snapshot of
    production behavior (bucketized distributions + health metrics) from
    monitor traces; JSON save/load via `DriftBaseline`
  - `DriftDetector`: compare a window of traces against the baseline
    - Distribution drift via Population Stability Index (PSI) on
      `total_latency_ms`, `query_length`, `answer_length`, `retrieval_count`
    - Metric drift on success rate (absolute drop) and p95 latency
      (relative increase), with warning/critical escalation
    - `min_samples` guard: small windows are flagged `insufficient_data`
      instead of producing noisy findings
  - Alerting integration: `DriftReport.to_alerts()` (source="drift") and
    `DriftDetector.check_and_alert()` dispatch through `AlertManager`
  - `MonitorStore.get_traces()`: fetch raw traces for a time window
  - Configurable `DriftThresholds`; public exports from package root
  - Documentation at `docs/user-guide/drift.md`

## [1.8.0] - 2026-09-03

### Added
- **SLM-based medical abbreviation disambiguation** (#95)
  - `SLMDisambiguator`: resolves ambiguous medical abbreviations (MS, PE, CP...)
    with a small local model via closed-set classification over the dictionary's
    own candidate meanings
  - Escalation only when the keyword scorer has zero context hits; decisions
    cached per (abbreviation, context window); abstention and errors fall back
    to the existing priority default
  - `MedicalAbbreviationNormalizer.normalize_text_async()` and a pluggable
    `DisambiguationStrategy` protocol
  - New `disambiguation_llm` parameter on `LLMJudge` and `FaithfulnessEvaluator`
  - Expansions resolved by the SLM are tagged `[slm]` for traceability

## [1.7.0] - 2026-02-16

### Added
- **Alerting System**
  - `AlertManager` for multi-channel alert dispatch
    - Concurrent sending to multiple adapters via `asyncio.gather`
    - Rule-based alerting with `check_and_alert()` method
    - Adapter and rule management (add/remove/list)
  - `AlertRule` for threshold-based alerting
    - Conditions: `lt`, `gt`, `lte`, `gte`, `eq`, `neq`
    - Configurable severity (INFO, WARNING, CRITICAL)
    - Cooldown support to prevent alert fatigue
  - `WebhookAlertAdapter` for generic HTTP notifications
    - POST requests with JSON payload
    - Custom headers support
    - Timeout configuration
  - `SlackAlertAdapter` for Slack notifications
    - Block Kit formatting with severity colors
    - Emoji indicators per severity level
    - Channel/webhook URL configuration
  - Core protocols and types
    - `Alert` dataclass (title, message, severity, source, metadata, timestamp)
    - `AlertResult` for delivery status tracking
    - `AlertSeverity` enum (INFO, WARNING, CRITICAL)
    - `AlertAdapter` protocol for custom adapter implementations
- **New Exports**: `AlertManager`, `AlertRule`, `AlertSeverity` available from `ragnarok_ai` package root

### Changed
- Roadmap updated: Alerting marked as complete, v1.8+ planned items updated

## [1.6.0] - 2026-02-16

### Added
- **Production Monitoring MVP**
  - `MonitorClient` for instrumenting RAG pipelines with sampling
  - Monitor daemon with HTTP endpoints:
    - `POST /ingest` - Receive traces from clients
    - `GET /metrics` - Prometheus scrape endpoint
    - `GET /health` - Health check
    - `GET /stats` - JSON statistics for CLI
  - SQLite storage with automatic retention (7 days traces, 90 days aggregates)
  - Prometheus metrics export (request counts, success rate, latency percentiles)
  - CLI commands:
    - `ragnarok monitor start` - Start daemon (background or foreground)
    - `ragnarok monitor stop` - Stop running daemon
    - `ragnarok monitor status` - Show daemon status and metrics
    - `ragnarok monitor stats` - Detailed statistics (1h, 24h, 7d periods)
  - Sampling support (configurable rate, force option for important queries)
  - PII-safe query hashing (SHA256)
  - Custom metadata support for tenant/route slicing
  - Documentation at `docs/user-guide/monitoring.md`
- **New Export**: `MonitorClient` available from `ragnarok_ai` package root

### Changed
- Roadmap updated: Production Monitoring MVP marked as complete

## [1.5.0] - 2026-02-16

### Added
- **Kubernetes Helm Chart**
  - Full Helm chart in `helm/ragnarok-ai/`
  - Job mode for one-shot evaluations (default)
  - CronJob mode for scheduled evaluations
  - ConfigMap support for `ragnarok.yaml` configuration
  - PVC support for testset/results persistence
  - Ollama URL configuration via environment variables
  - Security context (non-root, read-only filesystem, dropped capabilities)
  - Resource limits and requests
  - ServiceAccount with configurable annotations
- **Air-Gapped Deployment Guide**
  - Complete documentation at `docs/deployment/air-gapped.md`
  - Prerequisites and component sizing
  - Preparation steps (Docker save, pip download, Ollama model export)
  - Transfer options (USB, private registry, data diode)
  - Installation on air-gapped machines
  - Verification and troubleshooting
  - Security considerations
- **Enterprise Deployment Stack**
  - Docker + Helm + Air-gapped = complete enterprise offering
  - Data sovereignty and regulatory compliance support
  - Local-first architecture for GDPR, HIPAA, defense environments

### Fixed
- **Dockerfile**: Source files now correctly copied to runtime image (fixes `ModuleNotFoundError`)

### Changed
- Roadmap updated: Docker/Kubernetes helm charts and Air-gapped deployment guide marked as complete

## [1.4.1] - 2026-02-15

### Added
- **Dataset Versioning**
  - Shared hashing utilities in `ragnarok_ai.core.hashing`
    - `canonical_json()` — Deterministic JSON serialization
    - `sha256_hex()` / `sha256_short()` — Content-based hashing
    - `compute_hash()` — Generic hash computation for any serializable object
  - TestSet versioning fields
    - `schema_version` — Schema version for future migrations
    - `dataset_version` — Semantic version of the dataset content
    - `created_at` — Creation timestamp (UTC)
    - `author` — Dataset author
    - `source` — Data source identifier
  - `TestSet.compute_hash()` / `TestSet.hash_short` — Content-based identity
  - Dataset I/O module (`ragnarok_ai.dataset`)
    - `load_testset()` — Load from JSON, JSONL, or structured object
    - `save_testset()` — Export TestSet to JSON
  - Dataset diff functionality
    - `build_index()` — Create stable key index (metadata.id or content hash)
    - `diff_testsets()` — Compare two TestSet versions
    - `DatasetDiffReport` — Added/removed/modified/unchanged detection
- **CLI `dataset` Command**
  - `ragnarok dataset diff <v1> <v2>` — Compare two dataset versions
    - `--key` — Custom key field for matching items
    - `--ignore-metadata` — Exclude metadata from comparison
    - `--show` — Limit number of items shown
    - `--output` — Export diff report to JSON
    - `--fail-on-change` — Exit with error if changes detected (CI gating)
  - `ragnarok dataset info <path>` — Show dataset metadata and statistics
- **Unified Artifact Schema**
  - `EvaluationArtifact` model for reproducible evaluation results
  - Automatic capture of environment, config, and RAGnarok version

### Fixed
- Version consistency between `pyproject.toml` and `__init__.py`

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

## [1.1.2] - 2026-01-27

### Fixed
- **`--json` flag now works** — v1.1.0/v1.1.1 required `--json-output`, now `--json` works
- `--json-output` kept as hidden alias for backward compatibility

## [1.1.1] - 2026-01-27 [YANKED]

### Note
- Incomplete fix for `--json` flag, use v1.1.2 instead

## [1.1.0] - 2026-01-27

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

## [1.0.2] - 2026-01-27

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

## [1.0.1] - 2026-01-27

### Fixed
- Use absolute URLs for images (PyPI compatibility)
- Add noqa comments for e2e import tests (lint fix)

## [1.0.0] - 2026-01-27

### Added
- PyPI publication (`pip install ragnarok-ai`)
- End-to-end installation tests
- PyPI badge in README

### Changed
- Updated README with pip install instructions
- Updated project metadata (author, keywords, classifiers)
- Marked as Production/Stable

## [0.9.0] - 2026-01-26

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

## [0.8.0] - 2026-01-26

### Added
- **Comparison & Benchmarking**
  - `compare()` function for side-by-side evaluation
  - `RegressionDetector` for quality drop alerts
  - `BenchmarkHistory` for time-series tracking
  - `DiffReport` for run-to-run comparison
- Baselines library with `BaselineResult` and `compare()`

## [0.7.0] - 2026-01-26

### Added
- **Framework Adapters**
  - LlamaIndex adapter (Retriever, QueryEngine, Index)
  - DSPy adapter (Retrieve, Module, RAG pattern)
- Custom RAG support via `RAGProtocol`
- Adapter contribution guide

## [0.6.0] - 2026-01-26

### Added
- **Cloud & Local LLM Adapters**
  - vLLM adapter for local high-performance inference
  - OpenAI adapter for optional cloud fallback
  - Anthropic adapter
- **Vector Store Adapters**
  - ChromaDB adapter
  - FAISS adapter (pure local, no server)

## [0.5.0] - 2026-01-26

### Added
- **Performance & Scale**
  - `MemoryCache` and `DiskCache` with `CacheProtocol`
  - `BatchEvaluator` for processing 1000+ queries
  - Async parallelization with `max_concurrency` parameter
  - Progress callbacks (sync and async)
  - Timeout and retry (`timeout`, `max_retries`, `retry_delay`)
  - Graceful cache error handling

## [0.4.0] - 2026-01-26

### Added
- **Framework Adapters**
  - LangChain integration
  - LangGraph integration
- **Observability**
  - OpenTelemetry export for tracing & debugging

## [0.3.0] - 2026-01-25

### Added
- **Test Generation**
  - Synthetic question generation from knowledge base
  - Multi-hop question support
  - Adversarial question generation
- Checkpointing system for crash recovery
- Golden set support (human-validated, versioned question sets)
- Baselines library (configs + expected results)
- NovaTech example dataset for quickstart

## [0.2.0] - 2026-01-25

### Added
- **Generation Metrics**
  - `FaithfulnessEvaluator` — Is the answer grounded in context?
  - `RelevanceEvaluator` — Does the answer address the question?
  - `HallucinationDetector` — Does the answer contain fabricated info?
- **Reporting**
  - HTML report with drill-down (failed questions, retrieved chunks)
- Qdrant vector store adapter
- Intelligent CI gating (stable metrics fail, LLM judgments warn)

## [0.1.0] - 2026-01-24

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

[1.7.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.4.1...v1.5.0
[1.4.1]: https://github.com/2501Pr0ject/RAGnarok-AI/compare/v1.4.0...v1.4.1
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
