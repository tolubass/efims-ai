"""Live inference service for the severity model.

Takes a ``SeverityRequest`` (the fields known at incident intake) and returns
a ``SeverityResponse``: the predicted class, confidence, per-class
probabilities, and a short SHAP-based explanation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.api.schemas.common import PredictionFactor
from src.api.schemas.severity_schema import SeverityRequest, SeverityResponse
from src.governance.human_review_flag import requires_human_review
from src.governance.prediction_audit_logger import log_prediction
from src.inference.base_service import BaseInferenceService
from src.preprocessing.cleaning import clean_incidents
from src.utils.date_utils import format_wat_timestamp
from src.utils.id_generator import generate_prediction_id
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEVERITY_LABELS: tuple[str, ...] = ("S1", "S2", "S3", "S4")
INDEX_TO_LABEL: dict[int, str] = dict(enumerate(SEVERITY_LABELS))

# Must match src.features.incident_features.CATEGORICAL_COLUMNS, since that
# is what the deployed model was actually trained on.
CATEGORICAL_COLUMNS: tuple[str, ...] = ("source", "channel", "incident_type", "reported_by_type")

# Used once, at service start, to rebuild the same sorted-unique label
# encoding the training pipeline used (see src.features.incident_features
# .encode_categoricals). A single inference request has no population of
# values to sort against on its own, so there is no way to reproduce the
# training-time codes without this.
TRAINING_DATA_PATH: str = "data/synthetic/10_incidents.csv"

_IMPACT_BY_RANK: tuple[str, ...] = ("high", "medium", "low")


class SeverityService(BaseInferenceService):
    """Wraps the trained severity model for live predictions."""

    def __init__(self) -> None:
        super().__init__("severity")
        self._category_mappings: dict[str, dict[str, int]] = self._build_category_mappings()

    @staticmethod
    def _build_category_mappings() -> dict[str, dict[str, int]]:
        """Rebuild the training-time category -> code mappings from the raw data.

        Returns an empty dict (so every category maps to -1) if the training
        data isn't available in this environment.
        """
        try:
            raw = pd.read_csv(TRAINING_DATA_PATH)
            cleaned = clean_incidents(raw)
            mappings: dict[str, dict[str, int]] = {}
            for col in CATEGORICAL_COLUMNS:
                values = cleaned[col].astype(str)
                uniques = sorted(values.unique())
                mappings[col] = {value: code for code, value in enumerate(uniques)}
            return mappings
        except Exception as error:
            logger.warning(
                "SeverityService: could not rebuild category mappings from %s (%s); "
                "all categorical features will map to -1 (unknown)",
                TRAINING_DATA_PATH,
                error,
            )
            return {}

    def _encode_category(self, column: str, value: str) -> int:
        """Return the training-time code for ``value``, or -1 if unseen/unmapped."""
        return self._category_mappings.get(column, {}).get(str(value), -1)

    def _build_feature_row(self, request: SeverityRequest) -> pd.DataFrame:
        """Build the single-row feature DataFrame the model expects."""
        narrative_length = (
            request.narrative_length
            if request.narrative_length is not None
            else (len(request.narrative) if request.narrative else 0)
        )
        exposure = float(request.financial_exposure_ngn)

        row = {
            "financial_exposure_ngn": exposure,
            "log_financial_exposure": float(np.log1p(max(exposure, 0.0))),
            "customer_impact_count": int(request.customer_impact_count),
            "ongoing_threat": int(bool(request.ongoing_threat)),
            "insider_involvement": int(bool(request.insider_involvement)),
            "data_compromise": int(bool(request.data_compromise)),
            "regulatory_relevance": int(bool(request.regulatory_relevance)),
            "source_encoded": self._encode_category("source", request.source),
            "channel_encoded": self._encode_category("channel", request.channel),
            "incident_type_encoded": self._encode_category("incident_type", request.incident_type),
            "reported_by_type_encoded": self._encode_category("reported_by_type", request.reported_by_type),
            "narrative_length": int(narrative_length),
            # received_timestamp isn't part of SeverityRequest, so these two
            # are fixed defaults rather than computed, as specified.
            "hour_of_day": 0,
            "is_business_hours_flag": 1,
        }

        columns = self.feature_columns or list(row.keys())
        return pd.DataFrame([row]).reindex(columns=columns, fill_value=0)

    def _explain(self, X: pd.DataFrame, predicted_index: int) -> list[PredictionFactor]:
        """Return up to 3 PredictionFactors from SHAP, or [] if SHAP fails."""
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                values = np.asarray(shap_values[predicted_index])[0]
            else:
                values = np.asarray(shap_values)
                values = values[0, :, predicted_index] if values.ndim == 3 else values[0]

            order = np.argsort(-np.abs(values))[:3]
            factors = []
            for rank, idx in enumerate(order):
                feature = X.columns[idx]
                value = X.iloc[0][feature]
                value = value.item() if hasattr(value, "item") else value
                factors.append(
                    PredictionFactor(
                        feature=feature,
                        impact=_IMPACT_BY_RANK[rank] if rank < len(_IMPACT_BY_RANK) else "low",
                        # Direction is relative to the predicted class: a
                        # positive SHAP value pushed the model toward the
                        # class it actually predicted.
                        direction="increases_risk" if values[idx] > 0 else "decreases_risk",
                        value=value,
                    )
                )
            return factors
        except Exception as error:
            logger.warning("SeverityService: SHAP explanation failed (%s); returning []", error)
            return []

    def predict(self, request: SeverityRequest) -> SeverityResponse:
        """Predict incident severity and return a full SeverityResponse."""
        if not self.is_loaded and not self.load_model():
            raise RuntimeError("SeverityService: model is not loaded and could not be loaded")

        X = self._build_feature_row(request)
        probabilities = self.model.predict_proba(X)[0]
        predicted_index = int(np.argmax(probabilities))
        confidence_score = float(probabilities[predicted_index])
        severity_prediction = INDEX_TO_LABEL[predicted_index]
        all_class_probabilities = {
            SEVERITY_LABELS[i]: float(p) for i, p in enumerate(probabilities)
        }

        top_factors = self._explain(X, predicted_index)
        requires_review = requires_human_review(confidence_score, "severity")
        prediction_id = generate_prediction_id("severity")

        log_prediction(
            prediction_id=prediction_id,
            model_name="severity",
            model_version=self.model_version,
            input_summary={
                "incident_id": request.incident_id,
                "financial_exposure_ngn": request.financial_exposure_ngn,
                "channel": request.channel,
                "incident_type": request.incident_type,
            },
            output_summary={
                "severity_prediction": severity_prediction,
                "confidence_score": confidence_score,
            },
            requester_id=request.requester_id or "system",
            requires_review=requires_review,
        )

        return SeverityResponse(
            prediction_id=prediction_id,
            model_name="severity",
            model_version=self.model_version,
            confidence_score=confidence_score,
            requires_human_review=requires_review,
            explanation=top_factors,
            timestamp_wat=format_wat_timestamp(),
            severity_prediction=severity_prediction,
            severity_score=confidence_score,
            all_class_probabilities=all_class_probabilities,
            available_at_prediction_time=True,
            top_factors=top_factors,
        )
