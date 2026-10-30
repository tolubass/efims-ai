"""Transaction anomaly scoring endpoint."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.dependencies import get_anomaly_service
from src.api.schemas.anomaly_schema import AnomalyRequest, AnomalyResponse
from src.api.schemas.common import ErrorResponse
from src.inference.anomaly_service import AnomalyService
from src.utils.date_utils import format_wat_timestamp
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/api/v1/anomaly/score", response_model=AnomalyResponse)
async def score_anomaly(
    payload: AnomalyRequest,
    anomaly_service: AnomalyService = Depends(get_anomaly_service),
):
    """Score a transaction for anomalous behaviour (0-1, higher = more suspicious).

    The score is advisory only: ``requires_human_review`` and the response
    disclaimer apply regardless of how suspicious the score looks. Returns
    503 if the anomaly model isn't loaded, or 500 with an ``ErrorResponse``
    body if scoring itself fails.
    """
    try:
        return anomaly_service.predict(payload)
    except Exception as error:
        logger.error("score_anomaly: prediction failed: %s", error)
        error_response = ErrorResponse(
            error_code="PREDICTION_FAILED",
            message=str(error),
            prediction_id=None,
            timestamp_wat=format_wat_timestamp(),
        )
        return JSONResponse(status_code=500, content=error_response.model_dump())
