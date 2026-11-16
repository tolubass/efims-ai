"""Severity prediction endpoint."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.dependencies import get_severity_service
from src.api.schemas.common import ErrorResponse
from src.api.schemas.severity_schema import SeverityRequest, SeverityResponse
from src.api.security import verify_api_key
from src.inference.severity_service import SeverityService
from src.utils.date_utils import format_wat_timestamp
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/api/v1/severity/predict",
    response_model=SeverityResponse,
    dependencies=[Depends(verify_api_key)],
)
async def predict_severity(
    payload: SeverityRequest,
    severity_service: SeverityService = Depends(get_severity_service),
):
    """Classify an incident's severity (S1-S4) from its intake-time details.

    The prediction is advisory only: ``requires_human_review`` and the
    response disclaimer apply regardless of confidence. Returns 503 if the
    severity model isn't loaded, or 500 with an ``ErrorResponse`` body if
    prediction itself fails.
    """
    try:
        return severity_service.predict(payload)
    except Exception as error:
        logger.error("predict_severity: prediction failed: %s", error)
        error_response = ErrorResponse(
            error_code="PREDICTION_FAILED",
            message=str(error),
            prediction_id=None,
            timestamp_wat=format_wat_timestamp(),
        )
        return JSONResponse(status_code=500, content=error_response.model_dump())
