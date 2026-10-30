"""Recovery propensity prediction endpoint."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.dependencies import get_recovery_service
from src.api.schemas.common import ErrorResponse
from src.api.schemas.recovery_schema import RecoveryRequest, RecoveryResponse
from src.inference.recovery_service import RecoveryService
from src.utils.date_utils import format_wat_timestamp
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/api/v1/recovery/predict", response_model=RecoveryResponse)
async def predict_recovery(
    payload: RecoveryRequest,
    recovery_service: RecoveryService = Depends(get_recovery_service),
):
    """Predict recovery propensity for a confirmed-loss case, with recommended actions.

    The prediction is advisory only: ``requires_human_review`` and the
    response disclaimer apply regardless of confidence. Returns 503 if the
    recovery model isn't loaded, or 500 with an ``ErrorResponse`` body if
    prediction itself fails.
    """
    try:
        return recovery_service.predict(payload)
    except Exception as error:
        logger.error("predict_recovery: prediction failed: %s", error)
        error_response = ErrorResponse(
            error_code="PREDICTION_FAILED",
            message=str(error),
            prediction_id=None,
            timestamp_wat=format_wat_timestamp(),
        )
        return JSONResponse(status_code=500, content=error_response.model_dump())
