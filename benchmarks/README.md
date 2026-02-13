# RAGnarok-AI Performance Benchmarks

Real-world performance measurements for RAGnarok-AI.

## Latest Results

**System:** Apple M2, 16GB RAM, Python 3.10, RAGnarok-AI v1.4.0

*Last updated: 2026-02-12*

### Evaluation Performance (Retrieval Metrics)

| Queries | Time    | Queries/sec | Peak RAM |
|---------|---------|-------------|----------|
| 50      | 0.002s  | 24,316      | 0.02 MB  |
| 500     | 0.021s  | 24,058      | 0.03 MB  |
| 5000    | 0.217s  | 23,057      | 0.17 MB  |

*Note: Retrieval metrics (precision, recall, MRR, NDCG) are pure computation — very fast.*

### LLM-as-Judge (Prometheus 2 Q5_K_M)

| Criterion     | Avg Time | Note |
|---------------|----------|------|
| Faithfulness  | ~25s     | Context + question + answer |
| Relevance     | ~22s     | Question + answer |
| Hallucination | ~28s     | Context + answer |
| Completeness  | ~24s     | Question + answer + context |

*Note: LLM-as-Judge requires Ollama with Prometheus 2. First call is slower (model loading).*
*These are the real bottleneck — retrieval metrics are near-instant by comparison.*

### Checkpointing Overhead

| Metric              | Value |
|---------------------|-------|
| Without checkpoint  | 0.001s |
| With checkpoint     | 0.005s |
| Overhead            | ~3-5% (on real workloads) |

*Note: Overhead is high (300%+) on pure metric computation because I/O dominates.*
*On real workloads with LLM calls (~25s each), checkpointing overhead becomes negligible (<1%).*

## Running Benchmarks

```bash
# Full benchmark suite
python benchmarks/run_benchmarks.py

# Skip LLM-as-Judge (faster, no Ollama needed)
python benchmarks/run_benchmarks.py --skip-llm

# Custom output path
python benchmarks/run_benchmarks.py --output results/my_run.json
```

## Requirements

- `psutil` for system info (included in dev dependencies)
- Ollama running with Prometheus 2 for LLM-as-Judge benchmarks

```bash
# Install Prometheus 2 for LLM-as-Judge benchmarks
ollama pull hf.co/RichardErkhov/prometheus-eval_-_prometheus-7b-v2.0-gguf:Q5_K_M
```

## Methodology

### Evaluation Performance
- Uses mock testsets with synthetic queries
- Measures wall-clock time with `time.perf_counter()`
- Tracks memory with `tracemalloc` (stdlib, no heavy dependencies)
- Runs retrieval metrics (precision, recall, MRR, NDCG)

### LLM-as-Judge Latency
- 3 samples per criterion (configurable)
- Includes model loading time in first sample
- Uses `keep_alive` to prevent model unloading between calls

### Checkpointing Overhead
- Compares same workload with and without checkpointing
- Uses temporary directory for checkpoint files
- Measures percentage overhead

## Interpreting Results

**Evaluation Performance:**
- Queries/sec depends on metric complexity
- RAM scales sub-linearly with query count
- 5000 queries is a realistic production batch

**LLM-as-Judge:**
- First call is slower (model loading)
- ~20-30s per evaluation is expected on M2 16GB
- Conservative scoring is normal for Prometheus 2

**Checkpointing:**
- <5% overhead is acceptable
- Higher overhead on small batches (fixed I/O cost)
- Enables crash recovery — worth the tradeoff

## Historical Results

Results are saved in `results/` with date-based filenames.

| Date       | System      | 500q Time | LLM Avg |
|------------|-------------|-----------|---------|
| 2026-02-12 | Apple M2    | -         | -       |

## Contributing

Run benchmarks on your hardware and submit a PR to add your results!

```bash
python benchmarks/run_benchmarks.py --output results/$(date +%Y-%m-%d)-$(hostname).json
```
