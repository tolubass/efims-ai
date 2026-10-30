"""Request and response schemas for ``POST /api/v1/next-action/recommend``.

The next-action model recommends what an investigator should do next on an
open case, given its severity, status, and how long it has been open.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.api.schemas.common import BaseRequest, BaseResponse

SEVERITY_CLASSES: tuple[str, ...] = ("S1", "S2", "S3", "S4")


class RecommendedAction(BaseModel):
    """One recommended next step, with its priority and rationale."""

    action: str = Field(..., description="What to do, e.g. 'Freeze receiving account'.")
    priority: Literal["immediate", "high", "medium", "low"] = Field(
        ..., description="Urgency of this action."
    )
    rationale: str = Field(..., description="Why this action is recommended.")
    sla_hours: Optional[float] = Field(
        None, ge=0, description="Hours allowed to complete this action, if an SLA applies."
    )


class NextActionRequest(BaseRequest):
    """Input for a next-action recommendation: the case's current state."""

    case_id: str = Field(..., description="Case ID, e.g. 'CASE-2026-000011'.")
    severity: str = Field(..., description="Incident severity: S1, S2, S3, or S4.")
    classification: str = Field(..., description="Fraud typology, e.g. 'Account Takeover'.")
    case_status: str = Field(..., description="Current case status, e.g. 'Containment'.")
    containment_confirmed: bool = Field(..., description="True if the threat has been contained.")
    days_open: float = Field(..., ge=0, description="Days since the case was opened.")
    channel: str = Field(..., description="Channel involved, e.g. 'Mobile Banking'.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "case_id": "CASE-2026-000011",
                "severity": "S1",
                "classification": "Account Takeover",
                "case_status": "Containment",
                "containment_confirmed": False,
                "days_open": 0.5,
                "channel": "Mobile Banking",
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


class NextActionResponse(BaseResponse):
    """Recommended next actions for a case, most urgent first."""

    recommended_actions: List[RecommendedAction] = Field(
        default_factory=list, description="Ordered actions, most urgent first."
    )
    sla_status: str = Field(
        ..., description="e.g. 'Within SLA', 'SLA breach imminent', 'SLA breached'."
    )
    escalation_required: bool = Field(..., description="True if the case must be escalated now.")
    escalation_reason: Optional[str] = Field(
        None, description="Why escalation is required, if applicable."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_id": "PRED-ACT-20260923-F4B2C7",
                "model_name": "next_action",
                "model_version": "v1.0",
                "confidence_score": 0.93,
                "requires_human_review": False,
                "explanation": [],
                "timestamp_wat": "2026-09-23 08:30:00 WAT",
                "recommended_actions": [
                    {
                        "action": "Contain the compromised account immediately",
                        "priority": "immediate",
                        "rationale": "S1 case with containment not yet confirmed.",
                        "sla_hours": 1.0,
                    },
                    {
                        "action": "Notify CBN and NDIC per S1 incident reporting rules",
                        "priority": "immediate",
                        "rationale": "Regulatory notification is mandatory for S1 incidents.",
                        "sla_hours": 4.0,
                    },
                    {
                        "action": "Open a case file and assign an investigator",
                        "priority": "high",
                        "rationale": "Formal investigation must start promptly for S1 cases.",
                        "sla_hours": 8.0,
                    },
                ],
                "sla_status": "SLA breach imminent",
                "escalation_required": True,
                "escalation_reason": "S1 case open for 0.5 days without confirmed containment.",
            }
        }
    )
