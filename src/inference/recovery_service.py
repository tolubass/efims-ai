"""Live inference service for the recovery propensity model.

Takes a ``RecoveryRequest`` and returns a ``RecoveryResponse``: a recovery
probability, a bool prediction, and a prioritised list of recommended
recovery actions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.api.schemas.common import PredictionFactor
from src.api.schemas.recovery_schema import RecoveryRequest, RecoveryResponse
from src.governance.human_review_flag import requires_human_review
from src.governance.prediction_audit_logger import log_prediction
from src.inference.base_service import BaseInferenceService
from src.preprocessing.cleaning import clean_cases, clean_incidents
from src.utils.config_loader import load_thresholds
from src.utils.date_utils import format_wat_timestamp
from src.utils.id_generator import generate_prediction_id
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEVERITY_ENCODING: dict[str, int] = {"S1": 3, "S2": 2, "S3": 1, "S4": 0}

HIGH_EXPOSURE_THRESHOLD_NGN: float = 5_000_000
HIGH_RECOVERY_LIKELIHOOD: float = 0.7
LOW_RECOVERY_LIKELIHOOD: float = 0.3

_IMPACT_BY_RANK: tuple[str, ...] = ("high", "medium", "low")

# Used once, at service start, to rebuild a stable sorted-unique label
# encoding for the two categorical recovery features, the same approach used
# by SeverityService/AnomalyService. classification values live in
# 11_cases.csv; channel values live in 10_incidents.csv.
CASES_DATA_PATH: str = "data/synthetic/11_cases.csv"
INCIDENTS_DATA_PATH: str = "data/synthetic/10_incidents.csv"


class RecoveryService(BaseInferenceService):
    """Wraps the trained recovery model for live predictions."""

    def __init__(self) -> None:
        super().__init__("recovery")
        self.thresholds: dict = load_thresholds()
        self._category_mappings: dict[str, dict[str, int]] = self._build_category_mappings()

    @staticmethod
    def _build_category_mappings() -> dict[str, dict[str, int]]:
        """Rebuild stable classification/channel -> code mappings from the raw data.

        Built from the full cases/incidents tables (not just the 31 cases
        that have a recovery record), so real category values that simply
        never appeared in the small labelled recovery set still get a code
        instead of falling back to -1 (unknown). Each column that fails to
        load is skipped independently and logged, rather than one failure
        blocking both.
        """
        mappings: dict[str, dict[str, int]] = {}

        try:
            cases = clean_cases(pd.read_csv(CASES_DATA_PATH))
            values = cases["classification"].astype(str)
            uniques = sorted(values.unique())
            mappings["classification"] = {value: code for code, value in enumerate(uniques)}
        except Exception as error:
            logger.warning(
                "RecoveryService: could not rebuild classification mapping from %s (%s); "
                "classification_encoded will map to -1 (unknown)",
                CASES_DATA_PATH,
                error,
            )

        try:
            incidents = clean_incidents(pd.read_csv(INCIDENTS_DATA_PATH))
            values = incidents["channel"].astype(str)
            uniques = sorted(values.unique())
            mappings["channel"] = {value: code for code, value in enumerate(uniques)}
        except Exception as error:
            logger.warning(
                "RecoveryService: could not rebuild channel mapping from %s (%s); "
                "channel_encoded will map to -1 (unknown)",
                INCIDENTS_DATA_PATH,
                error,
            )

        return mappings

    def _encode_category(self, column: str, value: str) -> int:
        """Return the stable code for ``value``, or -1 if unseen/unmapped."""
        return self._category_mappings.get(column, {}).get(str(value), -1)

    def _build_feature_row(self, request: RecoveryRequest) -> pd.DataFrame:
        """Build the single-row feature DataFrame the model expects."""
        gross_loss = float(request.gross_actual_loss_ngn)

        row = {
            "gross_actual_loss_ngn": gross_loss,
            "log_gross_loss": float(np.log1p(max(gross_loss, 0.0))),
            "severity_encoded": SEVERITY_ENCODING.get(request.severity, -1),
            "classification_encoded": self._encode_category("classification", request.fraud_type),
            "channel_encoded": self._encode_category("channel", request.channel),
            "days_to_detection": float(request.days_to_detection),
        }

        columns = self.feature_columns or list(row.keys())
        return pd.DataFrame([row]).reindex(columns=columns, fill_value=0)

    def _recommended_actions(self, request: RecoveryRequest, recovery_probability: float) -> list[str]:
        """Return an ordered list of plain-text recovery actions."""
        actions = ["Contact beneficiary bank immediately to flag the receiving account"]

        if request.gross_actual_loss_ngn > HIGH_EXPOSURE_THRESHOLD_NGN:
            actions.append("Escalate to Senior Recovery Officer — exposure exceeds NGN 5,000,000")
        if recovery_probability > HIGH_RECOVERY_LIKELIHOOD:
            actions.append("High recovery likelihood — initiate formal demand letter within 24 hours")
        if recovery_probability < LOW_RECOVERY_LIKELIHOOD:
            actions.append("Low recovery likelihood — consider insurance claim and legal referral")
        if request.containment_confirmed is True:
            actions.append("Containment confirmed — proceed to beneficiary bank recall request via NIBSS")
        else:
            actions.append("Containment not yet confirmed — prioritise account block before recovery action")

        return actions

    def _explain(self, X: pd.DataFrame) -> list[PredictionFactor]:
        """Return up to 3 PredictionFactors from SHAP, or [] if SHAP fails.

        Direction is relative to the positive class (recovery_success=True):
        a positive SHAP value pushed the prediction toward "will recover".
        """
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                values = np.asarray(shap_values[1])[0]
            else:
                values = np.asarray(shap_values)
                values = values[0, :, 1] if values.ndim == 3 else values[0]

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
                        direction="increases_risk" if values[idx] > 0 else "decreases_risk",
                        value=value,
                    )
                )
            return factors
        except Exception as error:
            logger.warning("RecoveryService: SHAP explanation failed (%s); returning []", error)
            return []

    def predict(self, request: RecoveryRequest) -> RecoveryResponse:
        """Predict recovery propensity and return a full RecoveryResponse."""
        if not self.is_loaded and not self.load_model():
            raise RuntimeError("RecoveryService: model is not loaded and could not be loaded")

        X = self._build_feature_row(request)
        probabilities = self.model.predict_proba(X)[0]
        recovery_probability = float(probabilities[1])
        recovery_success_prediction = recovery_probability >= 0.5

        key_factors = self._explain(X)
        requires_review = requires_human_review(recovery_probability, "recovery")
        prediction_id = generate_prediction_id("recovery")

        log_prediction(
            prediction_id=prediction_id,
            model_name="recovery",
            model_version=self.model_version,
            input_summary={
                "case_id": request.case_id,
                "severity": request.severity,
                "gross_actual_loss_ngn": request.gross_actual_loss_ngn,
            },
            output_summary={
                "recovery_probability": recovery_probability,
                "recovery_success_prediction": recovery_success_prediction,
            },
            requester_id=request.requester_id or "system",
            requires_review=requires_review,
        )

        return RecoveryResponse(
            prediction_id=prediction_id,
            model_name="recovery",
            model_version=self.model_version,
            confidence_score=recovery_probability,
            requires_human_review=requires_review,
            explanation=key_factors,
            timestamp_wat=format_wat_timestamp(),
            recovery_probability=recovery_probability,
            recovery_success_prediction=bool(recovery_success_prediction),
            key_recovery_factors=key_factors,
            recommended_recovery_actions=self._recommended_actions(request, recovery_probability),
        )
