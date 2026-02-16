# Production Monitoring

Monitor your RAG pipelines in production with sampling, Prometheus metrics, and SQLite storage.

---

## Overview

RAGnarok-AI Production Monitoring provides:

- **Trace Collection**: Capture production requests with configurable sampling
- **SQLite Storage**: Lightweight persistence with automatic retention
- **Prometheus Export**: `/metrics` endpoint for Grafana dashboards
- **CLI Management**: Start, stop, and inspect the monitoring daemon

```
                    Application
                        |
                        | MonitorClient (10% sampling)
                        v
+-------------------------------------------+
|  ragnarok monitor start --port 9090       |
|  +-------------------------------------+  |
|  |  Monitor Daemon                     |  |
|  |  - POST /ingest  (receive traces)   |  |
|  |  - GET /metrics  (Prometheus)       |  |
|  |  - GET /health   (health check)     |  |
|  |  - GET /stats    (JSON stats)       |  |
|  +-------------------------------------+  |
|                    |                      |
|                    v                      |
|  +-------------------------------------+  |
|  |  SQLite (~/.ragnarok/monitor.db)    |  |
|  |  - traces (7 days retention)        |  |
|  |  - aggregates (90 days retention)   |  |
|  +-------------------------------------+  |
+-------------------------------------------+
                    |
                    | Prometheus scrape
                    v
              Grafana Dashboard
```

---

## Quick Start

### 1. Start the Daemon

```bash
# Start in background (default)
ragnarok monitor start

# Start in foreground (for debugging)
ragnarok monitor start --foreground

# Custom port and retention
ragnarok monitor start --port 8080 --retention 14
```

### 2. Instrument Your Code

```python
from ragnarok_ai import MonitorClient

# Initialize client (connects to daemon)
client = MonitorClient(
    endpoint="http://localhost:9090",
    sample_rate=0.1,  # 10% sampling
)

# In your RAG pipeline
async def handle_query(query: str) -> str:
    with client.trace(query) as trace:
        # Retrieval
        docs = await retriever.search(query)
        trace.record_retrieval(docs, latency_ms=120.5)

        # Generation
        answer = await llm.generate(query, docs)
        trace.record_generation(answer, latency_ms=450.2, model="mistral:7b")

        return answer
```

### 3. View Metrics

```bash
# CLI stats
ragnarok monitor stats

# Prometheus endpoint
curl http://localhost:9090/metrics

# Health check
curl http://localhost:9090/health
```

---

## CLI Commands

### ragnarok monitor start

Start the monitoring daemon.

```bash
ragnarok monitor start [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--port`, `-p` | Port to listen on | 9090 |
| `--host` | Host to bind to | 0.0.0.0 |
| `--db` | Path to SQLite database | ~/.ragnarok/monitor.db |
| `--retention` | Days to keep raw traces | 7 |
| `--foreground`, `-f` | Run in foreground | false |

### ragnarok monitor stop

Stop the running daemon.

```bash
ragnarok monitor stop
```

### ragnarok monitor status

Show daemon status and basic metrics.

```bash
ragnarok monitor status

# Output:
# Monitor Status: RUNNING
# ------------------------------------
#   PID:              12345
#   Uptime:           2h 34m
#   Traces collected: 12,566
#   Success rate:     99.8%
#   Latency P50:      234ms
#   Latency P99:      1234ms
```

### ragnarok monitor stats

Show detailed statistics.

```bash
ragnarok monitor stats [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--period`, `-p` | Time period: 1h, 24h, 7d | 24h |

```bash
ragnarok monitor stats --period 1h

# Output:
# RAGnarok Monitor Stats (last 1h)
# ========================================
#
# Requests:     423 total
# Success Rate: 99.8%
# Errors:       0.2%
#
# Latency:
#   P50:  234ms
#   P95:  567ms
#   P99:  1234ms
```

---

## Python API

### MonitorClient

The main client for instrumenting your code.

```python
from ragnarok_ai import MonitorClient

client = MonitorClient(
    endpoint="http://localhost:9090",  # Daemon URL
    sample_rate=0.1,                   # 10% of requests
    enabled=True,                      # Enable/disable
)
```

### TraceContext

Context manager returned by `client.trace()`.

```python
with client.trace(query) as trace:
    # Record retrieval metrics
    trace.record_retrieval(
        docs=retrieved_docs,       # List of documents (for count)
        latency_ms=120.5,          # Retrieval latency
        count=5,                   # Or explicit count
    )

    # Record generation metrics
    trace.record_generation(
        answer="The answer is...", # For length calculation
        latency_ms=450.2,          # Generation latency
        model="mistral:7b",        # Model version
    )

    # Record errors
    trace.record_error(ValueError("Something went wrong"))

    # Add custom metadata
    trace.add_metadata("tenant", "acme")
    trace.add_metadata("route", "/api/query")
```

### Sampling Control

```python
# Normal trace (respects sample_rate)
with client.trace(query) as trace:
    ...

# Force trace regardless of sampling
with client.trace(query, force=True) as trace:
    ...

# Check if sampled
if trace.is_sampled:
    logger.debug("This request is being traced")
```

### Context Manager

The client can be used as a context manager to ensure traces are flushed:

```python
with MonitorClient(sample_rate=0.1) as client:
    for query in queries:
        with client.trace(query) as trace:
            ...
# Traces automatically flushed on exit
```

---

## API Endpoints

### POST /ingest

Receive traces from MonitorClient.

**Request:**
```json
{
  "traces": [
    {
      "query_hash": "a1b2c3d4",
      "query_length": 42,
      "retrieval_latency_ms": 120.5,
      "retrieval_count": 5,
      "generation_latency_ms": 450.2,
      "answer_length": 156,
      "total_latency_ms": 580.7,
      "model_version": "mistral:7b",
      "success": true
    }
  ]
}
```

**Response:**
```json
{"accepted": 1, "dropped": 0}
```

### GET /metrics

Prometheus-format metrics.

```
# HELP ragnarok_requests_total Total number of RAG requests
# TYPE ragnarok_requests_total counter
ragnarok_requests_total{status="success"} 12543
ragnarok_requests_total{status="error"} 23

# HELP ragnarok_success_rate Success rate (0.0-1.0)
# TYPE ragnarok_success_rate gauge
ragnarok_success_rate 0.9982

# HELP ragnarok_latency_seconds RAG request latency in seconds
# TYPE ragnarok_latency_seconds summary
ragnarok_latency_seconds{quantile="0.5"} 0.234
ragnarok_latency_seconds{quantile="0.95"} 0.567
ragnarok_latency_seconds{quantile="0.99"} 1.234
ragnarok_latency_seconds_count 12566

# HELP ragnarok_last_trace_seconds Seconds since last trace
# TYPE ragnarok_last_trace_seconds gauge
ragnarok_last_trace_seconds 2.3
```

### GET /health

Health check endpoint.

```json
{
  "status": "healthy",
  "uptime_seconds": 3600.5,
  "traces_collected": 12566
}
```

### GET /stats

JSON statistics for CLI.

```json
{
  "uptime_seconds": 3600.5,
  "traces_total": 12566,
  "traces_last_hour": 423,
  "success_rate": 0.998,
  "latency": {
    "p50": 0.234,
    "p95": 0.567,
    "p99": 1.234
  }
}
```

---

## Prometheus + Grafana Setup

### prometheus.yml

```yaml
scrape_configs:
  - job_name: 'ragnarok'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

### Grafana Dashboard

Create a dashboard with these panels:

1. **Request Rate**: `rate(ragnarok_requests_total[5m])`
2. **Success Rate**: `ragnarok_success_rate`
3. **Latency P50**: `ragnarok_latency_seconds{quantile="0.5"}`
4. **Latency P99**: `ragnarok_latency_seconds{quantile="0.99"}`
5. **Error Rate**: `rate(ragnarok_requests_total{status="error"}[5m])`

---

## Data Model

### TraceEvent

Each trace captures:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique trace ID |
| `timestamp` | datetime | When the trace occurred |
| `query_hash` | string | SHA256 hash of query (PII-safe) |
| `query_length` | int | Length of query string |
| `retrieval_latency_ms` | float | Retrieval step latency |
| `retrieval_count` | int | Number of documents retrieved |
| `generation_latency_ms` | float | Generation step latency |
| `answer_length` | int | Length of generated answer |
| `total_latency_ms` | float | Total request latency |
| `model_version` | string | LLM model used |
| `success` | bool | Whether request succeeded |
| `error_type` | string | Error type if failed |
| `metadata` | dict | Custom metadata |

### Storage

- **Raw traces**: Kept for 7 days (configurable via `--retention`)
- **Hourly aggregates**: Kept for 90 days
- **Database**: SQLite at `~/.ragnarok/monitor.db`

---

## Best Practices

### Sampling Rate

| Environment | Recommended Rate |
|-------------|------------------|
| Development | 1.0 (100%) |
| Staging | 0.5 (50%) |
| Production (low traffic) | 0.2 (20%) |
| Production (high traffic) | 0.05-0.1 (5-10%) |

### Error Handling

Always record errors to track failure rates:

```python
try:
    with client.trace(query) as trace:
        result = await process(query)
except Exception as e:
    trace.record_error(e)
    raise
```

### Metadata for Segmentation

Add metadata for filtering in dashboards:

```python
with client.trace(query) as trace:
    trace.add_metadata("tenant", tenant_id)
    trace.add_metadata("route", request.path)
    trace.add_metadata("user_tier", "premium")
```

---

## Troubleshooting

### Daemon not starting

```bash
# Check if already running
ragnarok monitor status

# Stop existing daemon
ragnarok monitor stop

# Start with verbose logging
ragnarok monitor start --foreground
```

### No traces being collected

1. Check daemon is running: `ragnarok monitor status`
2. Check endpoint is correct in MonitorClient
3. Check sample_rate is > 0
4. Check network connectivity to daemon

### High memory usage

Reduce retention period:

```bash
ragnarok monitor start --retention 3  # 3 days instead of 7
```

---

## Next Steps

- [CLI Reference](../ci-cd/cli-reference.md) - All CLI commands
- [Air-Gapped Deployment](../deployment/air-gapped.md) - Deploy without internet
