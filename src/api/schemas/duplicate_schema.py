"""Request and response schemas for ``POST /api/v1/duplicate/detect``.

The duplicate model checks whether a newly reported incident is the same
real-world event as one already on file, comparing narrative text and
structured fields against existing incidents.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import ConfigDict, Field

from src.api.schemas.common import BaseRequest, BaseResponse

DEFAULT_ACTION: str = "No action — not a duplicate"


class DuplicateRequest(BaseRequest):
    """Input for a duplicate check: the incoming incident to compare."""

    incident_id: str = Field(..., description="Incident ID, e.g. 'INC-0000042'.")
    narrative: str = Field(..., description="Free-text description of the incident.")
    incident_type: str = Field(..., description="Type of incident, e.g. 'Account Takeover'.")
    channel: str = Field(..., description="Banking channel involved, e.g. 'Mobile Banking'.")
    financial_exposure_ngn: float = Field(..., ge=0, description="Estimated exposure in NGN.")
    customer_impact_count: int = Field(..., ge=0, description="Number of customers affected.")
    state: str = Field(..., description="Nigerian state where the incident occurred.")
    event_timestamp: Optional[str] = Field(
        None, description="When the incident occurred, e.g. '2026-09-23 14:52:15 WAT'."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "incident_id": "INC-0000051",
                "narrative": "Customer reported unauthorised transfer of NGN 500000 from savings account",
                "incident_type": "Account Takeover",
                "channel": "Mobile Banking",
                "financial_exposure_ngn": 500000.0,
                "customer_impact_count": 1,
                "state": "Lagos",
                "event_timestamp": "2026-09-23 09:15:00 WAT",
            }
        }
    )


class DuplicateResponse(BaseResponse):
    """Duplicate check result, with any matching incidents found."""

    is_duplicate: bool = Field(..., description="True if this incident matches an existing one.")
    duplicate_group_id: Optional[str] = Field(
        None, description="Shared group ID if a match was found, e.g. 'DUP-0000003'."
    )
    similar_incidents: List[dict] = Field(
        default_factory=list,
        description="Candidate matches, each with incident_id, similarity_score, and snippet.",
    )
    match_explanation: str = Field("", description="Plain-text reason for the match decision.")
    action_required: str = Field(
        DEFAULT_ACTION, description="Suggested next step, e.g. 'Confirm merge with INC-0000042'."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_id": "PRED-DUP-20260923-B7C1D2",
                "model_name": "duplicate",
                "model_version": "v1.0",
                "confidence_score": 0.91,
                "requires_human_review": False,
                "explanation": [],
                "timestamp_wat": "2026-09-23 09:15:05 WAT",
                "is_duplicate": True,
                "duplicate_group_id": "DUP-0000003",
                "similar_incidents": [
                    {
                        "incident_id": "INC-0000042",
                        "similarity_score": 0.91,
                        "snippet": "Customer called to report unauthorised transfers from account",
                    }
                ],
                "match_explanation": "Narrative similarity 0.91 and matching channel, state, and incident type.",
                "action_required": "Confirm merge with INC-0000042 before creating a new case",
            }
        }
    )
