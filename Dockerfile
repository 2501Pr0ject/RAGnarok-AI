# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files (README.md needed by hatchling)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
RUN uv sync --frozen --no-dev --extra cli --extra ollama

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src/ ./src/

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Run as non-root user
USER nobody

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["ragnarok", "version"]

ENTRYPOINT ["ragnarok"]
CMD ["--help"]
