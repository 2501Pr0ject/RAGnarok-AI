# RAGnarok-AI Helm Chart

Deploy RAGnarok-AI evaluations on Kubernetes.

## Installation

```bash
helm install ragnarok ./helm/ragnarok-ai
```

## Modes

The chart supports three deployment modes:

### Job (default)

Run a one-shot evaluation:

```bash
helm install ragnarok ./helm/ragnarok-ai \
  --set mode=job \
  --set job.args="{evaluate,--demo}"
```

### CronJob

Run scheduled evaluations:

```bash
helm install ragnarok ./helm/ragnarok-ai \
  --set mode=cronjob \
  --set cronjob.schedule="0 6 * * *" \
  --set config.enabled=true
```

## Configuration

### Using a config file

```bash
helm install ragnarok ./helm/ragnarok-ai \
  --set config.enabled=true \
  --set-file config.ragnarokYaml=./ragnarok.yaml
```

### With Ollama

```bash
helm install ragnarok ./helm/ragnarok-ai \
  --set ollama.enabled=true \
  --set ollama.url="http://ollama.default.svc:11434"
```

### With persistence

```bash
helm install ragnarok ./helm/ragnarok-ai \
  --set persistence.enabled=true \
  --set persistence.size=5Gi
```

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | string | `job` | Deployment mode: `job`, `cronjob` |
| `image.repository` | string | `ghcr.io/2501pr0ject/ragnarok-ai` | Image repository |
| `image.tag` | string | `""` | Image tag (defaults to appVersion) |
| `job.args` | list | `["evaluate", "--demo"]` | Command arguments for job |
| `cronjob.schedule` | string | `"0 6 * * *"` | Cron schedule |
| `config.enabled` | bool | `false` | Mount ragnarok.yaml ConfigMap |
| `ollama.enabled` | bool | `false` | Enable Ollama URL env var |
| `ollama.url` | string | `"http://ollama:11434"` | Ollama service URL |
| `persistence.enabled` | bool | `false` | Enable PVC for /data |
| `resources.limits.memory` | string | `"512Mi"` | Memory limit |

See [values.yaml](values.yaml) for all options.

## Examples

### CI/CD evaluation job

```yaml
# values-ci.yaml
mode: job
job:
  args:
    - evaluate
    - --config
    - /config/ragnarok.yaml
    - --fail-under
    - "0.7"
config:
  enabled: true
  ragnarokYaml: |
    testset: /data/testset.json
    metrics:
      - precision
      - recall
persistence:
  enabled: true
  existingClaim: testset-pvc
```

```bash
helm install ragnarok ./helm/ragnarok-ai -f values-ci.yaml
```

### Daily regression check

```yaml
# values-daily.yaml
mode: cronjob
cronjob:
  schedule: "0 6 * * *"
  args:
    - evaluate
    - --config
    - /config/ragnarok.yaml
    - --output
    - /data/results.json
config:
  enabled: true
ollama:
  enabled: true
  url: "http://ollama.ml.svc:11434"
persistence:
  enabled: true
  size: 10Gi
```
