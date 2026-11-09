# EFIMS AI — production image.
#
# Base image matches the development venv's Python exactly (3.14.0): the
# official "python:3.14-slim" tag exists and is actively maintained on
# Docker Hub (verified there on 2026-11-09). Not built/run locally as part
# of this change — the Docker CLI is present on this machine but its daemon
# isn't reachable here, so this image has NOT been build-tested. Build and
# smoke-test it (see README.md) before relying on it.
FROM python:3.14-slim

WORKDIR /app

# libgomp1 is a common, well-known runtime requirement for lightgbm's Linux
# wheel (it links against OpenMP) and is not part of python:3.14-slim.
# Without it, `import lightgbm` — triggered as soon as a severity or
# recovery model is unpickled — fails with an ImportError at startup.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying the rest of the source, so this layer
# is only rebuilt when requirements.txt actually changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY model_registry/ model_registry/

# Audit logs (src/governance/prediction_audit_logger.py) are written here;
# mounted as a volume in docker-compose.yml so they survive a container
# restart instead of being lost with it.
RUN mkdir -p logs

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
