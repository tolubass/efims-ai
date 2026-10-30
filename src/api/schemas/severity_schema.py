"""Request and response schemas for ``POST /api/v1/severity/predict``.

The severity model classifies an incident as S1 (most critical) to S4 (least
critical). The request carries only information available when the incident
is reported. Post-investigation values are deliberately absent, to prevent
target leakage.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator

from src.api.schemas.common import BaseRequest, BaseResponse, PredictionFactor

SEVERITY_CLASSES: tuple[str, ...] = ("S1", "S2", "S3", "S4")


class SeverityRequest(BaseRequest):
    """Input for a severity prediction: the facts known at incident intake."""

    incident_id: str = Field(..., description="Incident ID, e.g. 'INC-0000042'.")
    financial_exposure_ngn: float = Field(..., ge=0, description="Estimated exposure in NGN.")
    customer_impact_count: int = Field(..., ge=0, description="Number of customers affected.")
    ongoing_threat: bool
    insider_involvement: bool
    data_compromise: bool
    regulatory_relevance: bool
    source: str = Field(..., description="How the incident was received, e.g. 'Customer Complaint'.")
    channel: str = Field(..., description="Banking channel involved, e.g. 'Mobile Banking'.")
    incident_type: str = Field(..., description="Type of incident, e.g. 'Account Takeover'.")
    reported_by_type: str = Field(..., description="Who reported it, e.g. 'Customer'.")
    narrative: Optional[str] = Field(None, description="Free-text description of the incident.")
    narrative_length: Optional[int] = Field(
        None, ge=0, description="Character count of the narrative; filled in automatically if omitted."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "incident_id": "INC-0000042",
                "financial_exposure_ngn": 4800000.0,
                "customer_impact_count": 3,
                "ongoing_threat": True,
                "insider_involvement": False,
                "data_compromise": False,
                "regulatory_relevance": True,
                "source": "Customer Complaint",
                "channel": "Mobile Banking",
                "incident_type": "Account Takeover",
                "reported_by_type": "Customer",
                "narrative": "Customer called to report unauthorised transfers",
            }
        }
    )

    @model_validator(mode="after")
    def _fill_narrative_length(self) -> "SeverityRequest":
        """Set narrative_length from the narrative when it is not supplied."""
        if self.narrative is not None and self.narrative_length is None:
            self.narrative_length = len(self.narrative)
        return self


class SeverityResponse(BaseResponse):
    """Severity prediction returned to the triage officer, with explanation."""

    severity_prediction: str = Field(..., description="Predicted class: S1, S2, S3 or S4.")
    severity_score: float = Field(..., description="Model's severity score for the incident.")
    all_class_probabilities: Dict[str, float] = Field(
        ..., description="Probability for each class; keys are exactly S1, S2, S3, S4."
    )
    available_at_prediction_time: bool = Field(
        True, description="Confirms only intake-time features were used (no leakage)."
    )
    top_factors: List[PredictionFactor] = Field(
        default_factory=list, description="Most influential features for this prediction."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_id": "PRED-SEV-20260923-A3F2B1",
                "model_name": "severity",
                "model_version": "v1.0",
                "confidence_score": 0.84,
                "requires_human_review": False,
                "explanation": [],
                "timestamp_wat": "2026-09-23 14:52:15 WAT",
                "severity_prediction": "S2",
                "severity_score": 0.84,
                "all_class_probabilities": {"S1": 0.05, "S2": 0.84, "S3": 0.09, "S4": 0.02},
                "available_at_prediction_time": True,
                "top_factors": [
                    {
                        "feature": "ongoing_threat",
                        "impact": "high",
                        "direction": "increases_risk",
                        "value": True,
                    },
                    {
                        "feature": "financial_exposure_ngn",
                        "impact": "medium",
                        "direction": "increases_risk",
                        "value": 4800000.0,
                    },
                ],
            }
        },
    )

    @field_validator("severity_prediction")
    @classmethod
    def _valid_severity(cls, value: str) -> str:
        """Reject anything other than S1, S2, S3 or S4."""
        if value not in SEVERITY_CLASSES:
            raise ValueError(f"severity_prediction must be one of {SEVERITY_CLASSES}, got {value!r}")
        return value

    @field_validator("all_class_probabilities")
    @classmethod
    def _valid_probabilities(cls, value: Dict[str, float]) -> Dict[str, float]:
        """Require exactly the keys S1-S4, each with a probability between 0 and 1."""
        if set(value) != set(SEVERITY_CLASSES):
            raise ValueError(
                f"all_class_probabilities keys must be exactly {SEVERITY_CLASSES}, got {sorted(value)}"
            )
        out_of_range = {k: v for k, v in value.items() if not 0.0 <= v <= 1.0}
        if out_of_range:
            raise ValueError(f"Probabilities must be between 0.0 and 1.0, got {out_of_range}")
        return value
