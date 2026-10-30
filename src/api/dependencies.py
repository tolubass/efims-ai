"""FastAPI dependency-injection functions.

Routes pull their service instance through these rather than constructing a
new one per request, so they always get the single instance loaded once at
startup by ``src.api.lifespan``. Each dependency returns 503 if its model
isn't loaded, so a route never has to check that itself.
"""

from fastapi import HTTPException, Request

from src.inference.anomaly_service import AnomalyService
from src.inference.recovery_service import RecoveryService
from src.inference.severity_service import SeverityService


def get_severity_service(request: Request) -> SeverityService:
    """Return the shared SeverityService, or 503 if its model isn't loaded."""
    service = request.app.state.severity_service
    if not service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Severity model not available. Check /api/v1/health for service status.",
        )
    return service


def get_anomaly_service(request: Request) -> AnomalyService:
    """Return the shared AnomalyService, or 503 if its model isn't loaded."""
    service = request.app.state.anomaly_service
    if not service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Anomaly model not available. Check /api/v1/health for service status.",
        )
    return service


def get_recovery_service(request: Request) -> RecoveryService:
    """Return the shared RecoveryService, or 503 if its model isn't loaded."""
    service = request.app.state.recovery_service
    if not service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Recovery model not available. Check /api/v1/health for service status.",
        )
    return service
