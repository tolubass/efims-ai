"""Health check endpoint.

The EFIMS platform pings this before relying on the AI service, so it must
respond even when one or more models failed to load — it reads service
state straight off ``app.state`` rather than going through the dependency
functions in ``src.api.dependencies``, which would 503 on a down model.
"""

from fastapi import APIRouter, Request

from src.utils.date_utils import format_wat_timestamp

router = APIRouter()

SERVICE_NAME: str = "EFIMS AI Service"
SERVICE_VERSION: str = "1.0.0"


@router.get("/api/v1/health")
async def health(request: Request) -> dict:
    """Return overall and per-model health.

    Overall status is "healthy" only if every model loaded successfully;
    otherwise "degraded".
    """
    services = {
        "severity": request.app.state.severity_service,
        "anomaly": request.app.state.anomaly_service,
        "recovery": request.app.state.recovery_service,
    }

    models = {name: service.health_check() for name, service in services.items()}
    all_loaded = all(service.is_loaded for service in services.values())

    return {
        "status": "healthy" if all_loaded else "degraded",
        "timestamp_wat": format_wat_timestamp(),
        "models": {
            name: {"status": info["status"], "version": info["model_version"]}
            for name, info in models.items()
        },
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
    }
