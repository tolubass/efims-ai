"""Shared Pydantic (v2) schemas for the efims-ai API.

Every model endpoint builds its request and response schemas on top of the
classes here, so all responses share one consistent, regulator-friendly shape:

* a unique ``prediction_id`` for audit tracing,
* a ``confidence_score`` between 0.0 and 1.0,
* an ``explanation`` made of ``PredictionFactor`` items,
* a ``requires_human_review`` flag, and
* a ``disclaimer`` stating that a human must approve any action.

Nigerian banking AI requirements mean no prediction may trigger an action on
its own. These schemas make that explicit in every API response.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_DISCLAIMER: str = (
    "This is an AI-generated suggestion. Human review and "
    "approval is required before any action is taken on a case."
)


class PredictionFactor(BaseModel):
    """One factor that explains a prediction.

    Example: ``feature="ongoing_threat", impact="high",
    direction="increases_risk", value=True``.
    """

    feature: str = Field(..., description="Name of the input feature.")
    impact: Literal["high", "medium", "low"] = Field(
        ..., description="How strongly this feature influenced the prediction."
    )
    direction: Literal["increases_risk", "decreases_risk"] = Field(
        ..., description="Whether this feature pushed the prediction up or down."
    )
    value: Optional[Any] = Field(None, description="The feature's input value, if useful to show.")


class BaseRequest(BaseModel):
    """Parent class for every API request schema."""

    requester_id: Optional[str] = Field(
        None, description="ID of the user or system making the request (for audit)."
    )
    request_timestamp: Optional[str] = Field(
        None, description="When the request was made, e.g. '2026-09-23 14:52:15 WAT'."
    )


class BaseResponse(BaseModel):
    """Parent class for every API response schema.

    Guarantees that each response carries a prediction ID, a confidence score,
    an explanation, a human review flag, and the human-approval disclaimer.
    """

    # Allow field names starting with "model_" (model_name, model_version);
    # pydantic v2 otherwise warns that they clash with its protected namespace.
    model_config = ConfigDict(protected_namespaces=())

    prediction_id: str = Field(..., description="Unique ID, e.g. 'PRED-SEV-20260923-A3F2B1'.")
    model_name: str = Field(..., description="Model that produced the prediction, e.g. 'severity'.")
    model_version: str = Field("v1.0", description="Version of the model used.")
    confidence_score: float = Field(..., description="Model confidence, from 0.0 to 1.0.")
    requires_human_review: bool = Field(
        ..., description="True if confidence is below the review threshold in thresholds.yaml."
    )
    explanation: List[PredictionFactor] = Field(
        default_factory=list, description="Factors that explain the prediction."
    )
    timestamp_wat: str = Field(..., description="When the prediction was made, in WAT.")
    disclaimer: str = Field(DEFAULT_DISCLAIMER, description="Human-approval notice.")

    @field_validator("confidence_score")
    @classmethod
    def _confidence_in_range(cls, value: float) -> float:
        """Reject confidence scores outside 0.0 to 1.0."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence_score must be between 0.0 and 1.0, got {value}")
        return value


class ErrorResponse(BaseModel):
    """Returned by the API when a request cannot be processed."""

    error_code: str = Field(..., description="Machine-readable code, e.g. 'MODEL_NOT_LOADED'.")
    message: str = Field(..., description="Human-readable explanation of the error.")
    prediction_id: Optional[str] = Field(
        None, description="Prediction ID, if one was assigned before the error."
    )
    timestamp_wat: str = Field(..., description="When the error occurred, in WAT.")
