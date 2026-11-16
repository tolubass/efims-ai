# EFIMS AI — hosting image (repo-root Dockerfile for container platforms
# that require the Dockerfile at the repo root, e.g. Hugging Face Spaces).
# docker/Dockerfile is left as-is for local/compose use; this one is a
# separate, parallel build target, not a replacement.
#
# Not build/run-tested here as part of this change unless Part 1 step 6 of
# the task reports a real `docker build` + /api/v1/health result below —
# check that before relying on this image.
FROM python:3.14-slim

WORKDIR /app

# libgomp1: lightgbm's Linux wheel links against OpenMP and needs this at
# import time (triggered as soon as a severity/anomaly/recovery model is
# unpickled) — not part of python:3.14-slim by default. Same requirement
# already documented in docker/Dockerfile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying source, so this layer only rebuilds
# when requirements.txt actually changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY model_registry/ model_registry/
COPY data/synthetic/ data/synthetic/

# Non-root user (uid 1000) -- /app/logs is where
# src/governance/prediction_audit_logger.py writes audit logs; created and
# owned by that user up front so it doesn't need root at runtime. /app
# itself is made writable by the same user for the same reason.
RUN useradd --uid 1000 --create-home --shell /bin/bash appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app

USER appuser

ENV PORT=7860

EXPOSE 7860

# Shell form (not exec/JSON form) so the shell actually expands ${PORT},
# with a default of 7860 if the platform doesn't set it.
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}
