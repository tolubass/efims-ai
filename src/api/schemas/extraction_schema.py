"""Request and response schemas for ``POST /api/v1/extraction/entities``.

The extraction model pulls investigation-relevant entities out of a free-text
incident narrative. Entities below the confidence threshold in
configs/base/thresholds.yaml -> nlp are flagged for the user to confirm.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.api.schemas.common import BaseRequest, BaseResponse

ENTITY_TYPES: tuple[str, ...] = ("AMOUNT", "ACCOUNT", "DATE", "LOCATION", "CUSTOMER", "FRAUD_TYPE")

# Must match configs/base/thresholds.yaml -> nlp.confidence_threshold.
NLP_CONFIDENCE_THRESHOLD: float = 0.60


class ExtractedEntity(BaseModel):
    """One entity found in a narrative, with its position and confidence."""

    entity_type: str = Field(
        ..., description="One of AMOUNT, ACCOUNT, DATE, LOCATION, CUSTOMER, FRAUD_TYPE."
    )
    entity_value: str = Field(..., description="The extracted text span.")
    start_char: int = Field(..., ge=0, description="Start character position in the narrative.")
    end_char: int = Field(..., ge=0, description="End character position in the narrative.")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence for this entity, 0-1.")
    requires_confirmation: bool = Field(
        ..., description=f"True if confidence is below {NLP_CONFIDENCE_THRESHOLD} (nlp threshold)."
    )

    @field_validator("entity_type")
    @classmethod
    def _valid_entity_type(cls, value: str) -> str:
        """Reject anything outside the known entity types."""
        if value not in ENTITY_TYPES:
            raise ValueError(f"entity_type must be one of {ENTITY_TYPES}, got {value!r}")
        return value


class ExtractionRequest(BaseRequest):
    """Input for entity extraction: the raw narrative to parse."""

    incident_id: str = Field(..., description="Incident ID, e.g. 'INC-0000042'.")
    narrative: str = Field(..., description="Raw incident text to extract entities from.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "incident_id": "INC-0000042",
                "narrative": (
                    "Customer reported an unauthorised transfer of NGN 500000 "
                    "from a Lagos branch account on 23 September 2026."
                ),
            }
        }
    )


class ExtractionResponse(BaseResponse):
    """Entities found in a narrative, with counts derived from the list."""

    incident_id: str = Field(..., description="Incident ID the entities were extracted from.")
    entities: List[ExtractedEntity] = Field(default_factory=list, description="Entities found.")
    entity_count: int = Field(0, ge=0, description="Total entities found; set automatically.")
    high_confidence_count: int = Field(
        0, ge=0, description=f"Entities with confidence >= {NLP_CONFIDENCE_THRESHOLD}; set automatically."
    )
    extraction_note: str = Field("", description="Free-text note about the extraction, if any.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_id": "PRED-NLP-20260923-E1F3A9",
                "model_name": "nlp",
                "model_version": "v1.0",
                "confidence_score": 0.75,
                "requires_human_review": False,
                "explanation": [],
                "timestamp_wat": "2026-09-23 10:00:00 WAT",
                "incident_id": "INC-0000042",
                "entities": [
                    {
                        "entity_type": "AMOUNT",
                        "entity_value": "NGN 500000",
                        "start_char": 40,
                        "end_char": 50,
                        "confidence": 0.92,
                        "requires_confirmation": False,
                    },
                    {
                        "entity_type": "LOCATION",
                        "entity_value": "Lagos",
                        "start_char": 60,
                        "end_char": 65,
                        "confidence": 0.55,
                        "requires_confirmation": True,
                    },
                ],
                "entity_count": 2,
                "high_confidence_count": 1,
                "extraction_note": "One entity below the confidence threshold; confirm before saving.",
            }
        }
    )

    @model_validator(mode="after")
    def _derive_counts(self) -> "ExtractionResponse":
        """Set entity_count and high_confidence_count from the entities list."""
        self.entity_count = len(self.entities)
        self.high_confidence_count = sum(
            1 for entity in self.entities if entity.confidence >= NLP_CONFIDENCE_THRESHOLD
        )
        return self
