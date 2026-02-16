# Air-Gapped Deployment Guide

Deploy RAGnarok-AI in environments without internet access.

---

## Why Air-Gapped?

RAGnarok-AI is built **local-first** by design. This means:

- **Data Sovereignty**: Your evaluation data never leaves your infrastructure
- **Regulatory Compliance**: Meet strict data residency requirements (GDPR, HIPAA, defense, finance)
- **Security**: Eliminate external attack vectors and data exfiltration risks
- **Reproducibility**: Fully controlled environment with pinned versions

RAGnarok-AI gives you the choice: use cloud services when convenient, or go fully offline when required. The framework is **open to existing solutions** while enabling complete independence.

---

## 1. Prerequisites

### On the Internet-Connected Machine

- Docker (for saving images)
- Python 3.10+ with pip
- Ollama CLI (for model export)
- Sufficient disk space (~15GB for full stack)

### On the Air-Gapped Machine

- Docker runtime
- Python 3.10+
- Ollama runtime

### Components to Transfer

| Component | Size (approx) | Purpose |
|-----------|---------------|---------|
| RAGnarok-AI Docker image | ~170MB | CLI and evaluation framework |
| Ollama Docker image | ~1GB | LLM inference runtime |
| Ollama models | 4-8GB each | LLM models (Mistral, Llama, etc.) |
| Python wheels | ~50MB | Offline pip install |

---

## 2. Preparation (Internet-Connected Machine)

### 2.1 Save Docker Images

```bash
# Pull and save RAGnarok-AI image
docker pull ghcr.io/2501pr0ject/ragnarok-ai:1.4.1
docker save ghcr.io/2501pr0ject/ragnarok-ai:1.4.1 -o ragnarok-ai-1.4.1.tar

# Pull and save Ollama image
docker pull ollama/ollama:latest
docker save ollama/ollama:latest -o ollama-latest.tar

# Compress for transfer (optional)
gzip ragnarok-ai-1.4.1.tar
gzip ollama-latest.tar
```

**Total: ~1.2GB compressed**

### 2.2 Download Python Wheels

For pip-based installation without Docker:

```bash
# Create a directory for wheels
mkdir ragnarok-wheels && cd ragnarok-wheels

# Download RAGnarok-AI and all dependencies
pip download ragnarok-ai[cli,ollama] --dest .

# For additional adapters (optional)
pip download ragnarok-ai[all] --dest .
```

**Verify the download:**

```bash
ls -la *.whl
# Should include: ragnarok_ai-1.4.1-py3-none-any.whl, pydantic, typer, httpx, etc.
```

### 2.3 Export Ollama Models

```bash
# Pull models you need
ollama pull mistral
ollama pull llama3.2
ollama pull nomic-embed-text

# Find model locations
ls ~/.ollama/models/

# Create archive of models
tar -cvzf ollama-models.tar.gz -C ~/.ollama models/
```

**Model sizes:**
- `mistral` (7B): ~4GB
- `llama3.2` (3B): ~2GB
- `nomic-embed-text`: ~275MB

### 2.4 Prepare Configuration Files

Create a bundle with your configuration:

```bash
mkdir ragnarok-bundle
cd ragnarok-bundle

# Copy your testsets
cp /path/to/testset.json .

# Create ragnarok.yaml
cat > ragnarok.yaml << 'EOF'
testset: /data/testset.json
output: /data/results.json
fail_under: 0.7
metrics:
  - precision
  - recall
  - mrr
criteria:
  - faithfulness
  - relevance
ollama_url: http://localhost:11434
EOF
```

### 2.5 Create Transfer Package

```bash
# Create final transfer package
mkdir air-gapped-bundle
mv ragnarok-ai-1.4.1.tar.gz air-gapped-bundle/
mv ollama-latest.tar.gz air-gapped-bundle/
mv ollama-models.tar.gz air-gapped-bundle/
mv ragnarok-wheels/ air-gapped-bundle/
mv ragnarok-bundle/ air-gapped-bundle/

# Add installation script
cat > air-gapped-bundle/install.sh << 'EOF'
#!/bin/bash
set -e

echo "=== RAGnarok-AI Air-Gapped Installation ==="

# Load Docker images
echo "[1/4] Loading Docker images..."
gunzip -c ragnarok-ai-1.4.1.tar.gz | docker load
gunzip -c ollama-latest.tar.gz | docker load

# Install Python package
echo "[2/4] Installing Python package..."
pip install --no-index --find-links=ragnarok-wheels/ ragnarok-ai[cli,ollama]

# Extract Ollama models
echo "[3/4] Installing Ollama models..."
mkdir -p ~/.ollama
tar -xvzf ollama-models.tar.gz -C ~/.ollama

# Verify installation
echo "[4/4] Verifying installation..."
ragnarok version
echo ""
echo "=== Installation complete ==="
echo "Run 'ragnarok evaluate --demo' to test"
EOF

chmod +x air-gapped-bundle/install.sh

# Create checksum file
cd air-gapped-bundle
sha256sum * > SHA256SUMS
cd ..

# Final archive
tar -cvf ragnarok-air-gapped-bundle.tar air-gapped-bundle/
```

---

## 3. Transfer

### Option A: Physical Media (USB)

```bash
# Copy to USB drive
cp ragnarok-air-gapped-bundle.tar /media/usb/

# Verify integrity after copy
sha256sum /media/usb/ragnarok-air-gapped-bundle.tar
```

**Security considerations:**
- Use encrypted USB drives for sensitive environments
- Maintain chain of custody documentation
- Scan media on a dedicated terminal before transfer

### Option B: Private Registry

If you have a private Docker registry accessible from the air-gapped network:

```bash
# On internet machine: push to private registry
docker tag ghcr.io/2501pr0ject/ragnarok-ai:1.4.1 registry.internal/ragnarok-ai:1.4.1
docker push registry.internal/ragnarok-ai:1.4.1

docker tag ollama/ollama:latest registry.internal/ollama:latest
docker push registry.internal/ollama:latest
```

### Option C: Diode Transfer

For high-security environments with data diodes:

```bash
# Split large files if needed
split -b 500M ragnarok-air-gapped-bundle.tar bundle-part-

# Transfer parts through diode
# Reassemble on target
cat bundle-part-* > ragnarok-air-gapped-bundle.tar
```

---

## 4. Installation (Air-Gapped Machine)

### 4.1 Extract Bundle

```bash
# Copy from USB or transfer location
cp /media/usb/ragnarok-air-gapped-bundle.tar ~/

# Extract
tar -xvf ragnarok-air-gapped-bundle.tar
cd air-gapped-bundle

# Verify checksums
sha256sum -c SHA256SUMS
```

### 4.2 Run Installation Script

```bash
./install.sh
```

Or manually:

### 4.3 Manual Installation

**Load Docker images:**

```bash
gunzip -c ragnarok-ai-1.4.1.tar.gz | docker load
gunzip -c ollama-latest.tar.gz | docker load

# Verify
docker images | grep -E "ragnarok|ollama"
```

**Install Python package:**

```bash
pip install --no-index --find-links=ragnarok-wheels/ ragnarok-ai[cli,ollama]
```

**Install Ollama models:**

```bash
# Extract models
mkdir -p ~/.ollama
tar -xvzf ollama-models.tar.gz -C ~/.ollama

# Start Ollama server
ollama serve &

# Verify models are available
ollama list
```

### 4.4 Kubernetes / Helm Installation

For Kubernetes deployments:

```bash
# Load images into cluster nodes or push to internal registry
docker load -i ragnarok-ai-1.4.1.tar.gz
docker tag ghcr.io/2501pr0ject/ragnarok-ai:1.4.1 internal-registry/ragnarok-ai:1.4.1
docker push internal-registry/ragnarok-ai:1.4.1

# Install Helm chart with internal registry
helm install ragnarok ./helm/ragnarok-ai \
  --set image.repository=internal-registry/ragnarok-ai \
  --set image.tag=1.4.1
```

---

## 5. Verification

### 5.1 Check Installation

```bash
# Version check
ragnarok version
# Expected: ragnarok-ai v1.4.1

# List plugins
ragnarok plugins --list
# Should show: ollama [local], and other adapters
```

### 5.2 Run Demo Evaluation

```bash
# Start Ollama if not running
ollama serve &

# Pull a model (from local cache)
ollama list  # Should show pre-loaded models

# Run demo evaluation
ragnarok evaluate --demo
```

### 5.3 Run Full Evaluation

```bash
# With your testset
ragnarok evaluate \
  --config ragnarok-bundle/ragnarok.yaml \
  --testset ragnarok-bundle/testset.json
```

### 5.4 Docker Compose Verification

```bash
# Create docker-compose for air-gapped
cat > docker-compose-airgap.yml << 'EOF'
services:
  ragnarok:
    image: ghcr.io/2501pr0ject/ragnarok-ai:1.4.1
    depends_on:
      - ollama
    environment:
      - RAGNAROK_OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./ragnarok-bundle:/data:ro
    command: ["evaluate", "--config", "/data/ragnarok.yaml"]

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ~/.ollama:/root/.ollama
EOF

docker compose -f docker-compose-airgap.yml up
```

---

## Troubleshooting

### "Model not found" error

```bash
# Check Ollama models are loaded
ollama list

# If empty, re-extract models
tar -xvzf ollama-models.tar.gz -C ~/.ollama
ollama serve &
ollama list
```

### "No module named ragnarok_ai"

```bash
# Reinstall with correct path
pip install --no-index --find-links=/path/to/ragnarok-wheels/ ragnarok-ai[cli,ollama]
```

### Docker image not loading

```bash
# Check file integrity
gunzip -t ragnarok-ai-1.4.1.tar.gz

# Try without compression
gunzip ragnarok-ai-1.4.1.tar.gz
docker load -i ragnarok-ai-1.4.1.tar
```

---

## Security Considerations

### Image Verification

Always verify image digests match expected values:

```bash
# Check digest after loading
docker inspect ghcr.io/2501pr0ject/ragnarok-ai:1.4.1 --format='{{.Id}}'
```

### Network Isolation

RAGnarok-AI in air-gapped mode requires **no outbound network access**:

- Ollama runs locally on `localhost:11434`
- No telemetry or phone-home functionality
- All evaluation data stays on-premise

### Updates

For updating in air-gapped environments:

1. Prepare new bundle on internet-connected machine
2. Follow same transfer procedure
3. Load new images alongside existing ones
4. Update Helm values or docker-compose to use new tags

---

## Summary

RAGnarok-AI fully supports air-gapped deployments, enabling:

- Complete data sovereignty
- Regulatory compliance (GDPR, HIPAA, defense)
- Reproducible, auditable evaluations
- Zero external dependencies at runtime

The **local-first** architecture means you choose your deployment model: cloud-connected for convenience, or fully isolated for maximum security. RAGnarok-AI adapts to your requirements, not the other way around.
