"""Request and response schemas for ``POST /api/v1/recovery/predict``.

The recovery model predicts the probability that a confirmed fraud loss will
be successfully recovered, to help prioritise recovery actions; see
configs/base/thresholds.yaml -> recovery for the human review threshold.
"""

from __future__ import annotations

from typing import List

from pydantic import ConfigDict, Field, field_validator

from src.api.schemas.common import BaseRequest, BaseResponse, PredictionFactor

SEVERITY_CLASSES: tuple[str, ...] = ("S1", "S2", "S3", "S4")


class RecoveryRequest(BaseRequest):
    """Input for a recovery prediction: the case facts known once loss is confirmed."""

    case_id: str = Field(..., description="Case ID, e.g. 'CASE-2026-000031'.")
    gross_actual_loss_ngn: float = Field(..., ge=0, description="Confirmed gross loss in NGN.")
    severity: str = Field(..., description="Incident severity: S1, S2, S3, or S4.")
    fraud_type: str = Field(..., description="Fraud typology, e.g. 'SIM Swap'.")
    channel: str = Field(..., description="Channel involved, e.g. 'Mobile Banking'.")
    days_to_detection: float = Field(..., ge=0, description="Days from event to discovery.")
    containment_confirmed: bool = Field(..., description="True if the threat has been contained.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "case_id": "CASE-2026-000031",
                "gross_actual_loss_ngn": 1200000.0,
                "severity": "S2",
                "fraud_type": "SIM Swap",
                "channel": "Mobile Banking",
                "days_to_detection": 1.5,
                "containment_confirmed": True,
            }
        }
    )

    @field_validator("severity")
    @classmethod
    def _valid_severity(cls, value: str) -> str:
        """Reject anything other than S1, S2, S3, or S4."""
        if value not in SEVERITY_CLASSES:
            raise ValueError(f"severity must be one of {SEVERITY_CLASSES}, got {value!r}")
        return value


class RecoveryResponse(BaseResponse):
    """Recovery prediction, with the factors behind it and suggested actions."""

    recovery_probability: float = Field(..., ge=0, le=1, description="Probability recovery will succeed.")
    recovery_success_prediction: bool = Field(..., description="Predicted outcome at the decision threshold.")
    key_recovery_factors: List[PredictionFactor] = Field(
        default_factory=list, description="Factors that most influenced the prediction."
    )
    recommended_recovery_actions: List[str] = Field(
        default_factory=list, description="Ordered plain-text actions, most urgent first."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_id": "PRED-REC-20260923-D2A8E1",
                "model_name": "recovery",
                "model_version": "v1.0",
                "confidence_score": 0.79,
                "requires_human_review": False,
                "explanation": [],
                "timestamp_wat": "2026-09-23 13:05:00 WAT",
                "recovery_probability": 0.79,
                "recovery_success_prediction": True,
                "key_recovery_factors": [
                    {
                        "feature": "days_to_detection",
                        "impact": "high",
                        "direction": "increases_risk",
                        "value": 1.5,
                    },
                    {
                        "feature": "containment_confirmed",
                        "impact": "medium",
                        "direction": "decreases_risk",
                        "value": True,
                    },
                ],
                "recommended_recovery_actions": [
                    "Contact beneficiary bank immediately",
                    "File court order to freeze receiving account",
                    "Notify NFIU of suspected proceeds movement",
                ],
            }
        }
    )
