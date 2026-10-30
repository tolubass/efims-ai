"""Request and response schemas for ``POST /api/v1/anomaly/score``.

The anomaly model scores a single transaction for how unusual it is. A higher
``anomaly_score`` means more suspicious; see configs/base/thresholds.yaml ->
anomaly for the score at which a transaction is surfaced to fraud analysts.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import ConfigDict, Field, field_validator

from src.api.schemas.common import BaseRequest, BaseResponse

VELOCITY_INDICATORS: tuple[str, ...] = ("Normal", "Elevated", "High")


class AnomalyRequest(BaseRequest):
    """Input for an anomaly score: the transaction and its risk signals."""

    transaction_id: str = Field(..., description="Transaction ID, e.g. 'TXN-0004821'.")
    amount_ngn: float = Field(..., ge=0, description="Transaction amount in NGN.")
    channel: str = Field(..., description="Channel used, e.g. 'ATM', 'Mobile Banking'.")
    transaction_type: str = Field(..., description="Type of transaction, e.g. 'ATM Withdrawal'.")
    velocity_indicator: str = Field(
        ..., description="Recent transaction velocity: Normal, Elevated, or High."
    )
    ip_risk_score: float = Field(..., ge=0, le=1, description="Risk score of the originating IP, 0-1.")
    device_risk_score: float = Field(..., ge=0, le=1, description="Risk score of the device used, 0-1.")
    is_reversal: bool = Field(..., description="True if this transaction reverses a prior one.")
    is_chargeback: bool = Field(..., description="True if this transaction is a chargeback.")
    account_risk_rating: str = Field(..., description="Account's standing risk rating, e.g. 'High'.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN-0004821",
                "amount_ngn": 850000.0,
                "channel": "ATM",
                "transaction_type": "ATM Withdrawal",
                "velocity_indicator": "High",
                "ip_risk_score": 0.72,
                "device_risk_score": 0.65,
                "is_reversal": False,
                "is_chargeback": False,
                "account_risk_rating": "Medium",
            }
        }
    )

    @field_validator("velocity_indicator")
    @classmethod
    def _valid_velocity(cls, value: str) -> str:
        """Reject anything other than Normal, Elevated, or High."""
        if value not in VELOCITY_INDICATORS:
            raise ValueError(f"velocity_indicator must be one of {VELOCITY_INDICATORS}, got {value!r}")
        return value


class AnomalyResponse(BaseResponse):
    """Anomaly score for a transaction, with the signals behind it."""

    anomaly_score: float = Field(
        ..., ge=0, le=1, description="0-1; higher means more anomalous."
    )
    is_suspicious: bool = Field(..., description="True if the score is above the anomaly threshold.")
    anomaly_type: Optional[str] = Field(
        None, description="Likely anomaly category, e.g. 'Velocity spike', if identifiable."
    )
    behavioural_signals: List[str] = Field(
        default_factory=list, description="Plain-text signals detected for this transaction."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_id": "PRED-ANO-20260923-C9E4F0",
                "model_name": "anomaly",
                "model_version": "v1.0",
                "confidence_score": 0.88,
                "requires_human_review": False,
                "explanation": [],
                "timestamp_wat": "2026-09-23 11:20:00 WAT",
                "anomaly_score": 0.88,
                "is_suspicious": True,
                "anomaly_type": "Velocity spike",
                "behavioural_signals": [
                    "Transaction at unusual hour",
                    "New device not previously seen",
                    "Velocity above normal threshold",
                ],
            }
        }
    )
