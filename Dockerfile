# ── Stage 1: Build dependencies ─────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Production image ──────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Railway provides PORT env var, default to 8080
ENV PORT=8080
EXPOSE ${PORT}

# IMPORTANT: Must run from project root so that:
#   1. "artifacts/" relative path resolves correctly
#   2. sys.path insert finds graph/ and models/ packages
# Gunicorn points to deployment.app:app (the module path)
CMD gunicorn \
    --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --threads 2 \
    --timeout 120 \
    --preload \
    deployment.app:app
