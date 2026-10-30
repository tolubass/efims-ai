"""EFIMS AI Service — FastAPI application entry point.

Creates the app, loads models on startup via ``lifespan``, and wires up the
health and prediction routes. Every prediction returned by this service is
advisory: human review and approval is required before any action is taken
on a case (see each response's ``disclaimer`` field).
"""

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.lifespan import lifespan
from src.api.routes import anomaly, health, recovery, severity
from src.utils.logger import get_logger

logger = get_logger(__name__)

SERVICE_NAME: str = "EFIMS AI Service"
SERVICE_VERSION: str = "1.0.0"

app = FastAPI(
    title=SERVICE_NAME,
    description=(
        "AI-powered fraud and incident management for Nigerian banking "
        "institutions. All predictions are advisory — human approval "
        "required before any action is taken."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # development only — restrict before production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status code, and duration for every request."""
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


app.include_router(health.router)
app.include_router(severity.router)
app.include_router(anomaly.router)
app.include_router(recovery.router)


@app.get("/")
async def root() -> dict:
    """Basic service info and pointers to the docs and health endpoint."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
