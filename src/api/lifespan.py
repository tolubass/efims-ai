"""FastAPI lifespan: load every model service once at startup.

Loading up front (instead of per-request) means a prediction never pays the
cost of a disk load, and storing the services on ``app.state`` lets every
route share the same loaded instances. A model that fails to load never
crashes startup — it's just reported as unavailable via /api/v1/health, and
any route that needs it returns 503 until it's fixed.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.inference.anomaly_service import AnomalyService
from src.inference.recovery_service import RecoveryService
from src.inference.severity_service import SeverityService
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model services on startup; nothing to tear down on shutdown."""
    logger.info("EFIMS AI Service starting up...")

    app.state.severity_service = SeverityService()
    app.state.anomaly_service = AnomalyService()
    app.state.recovery_service = RecoveryService()

    for name, service in [
        ("severity", app.state.severity_service),
        ("anomaly", app.state.anomaly_service),
        ("recovery", app.state.recovery_service),
    ]:
        loaded = service.load_model()
        if loaded:
            logger.info(f"{name} model loaded successfully")
        else:
            logger.warning(f"{name} model failed to load — predictions will be unavailable")

    logger.info("EFIMS AI Service startup complete")

    yield

    logger.info("EFIMS AI Service shutting down...")
