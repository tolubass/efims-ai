"""Live inference service for the anomaly model.

Takes an ``AnomalyRequest`` and returns an ``AnomalyResponse``. Higher
``anomaly_score`` (0-1) means more suspicious.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.api.schemas.anomaly_schema import AnomalyRequest, AnomalyResponse
from src.governance.human_review_flag import requires_human_review
from src.governance.prediction_audit_logger import log_prediction
from src.inference.base_service import BaseInferenceService
from src.utils.config_loader import load_thresholds
from src.utils.date_utils import format_wat_timestamp, now_wat
from src.utils.id_generator import generate_prediction_id
from src.utils.logger import get_logger

logger = get_logger(__name__)

VELOCITY_ENCODING: dict[str, int] = {"Normal": 0, "Elevated": 1, "High": 2}

# Must match src.features.transaction_features.CATEGORICAL_COLUMNS, since
# that is what the deployed model was actually trained on.
CATEGORICAL_COLUMNS: tuple[str, ...] = ("channel", "transaction_type")

# Persisted once (from the exact training-time data, over the full
# 08_transactions.csv — verified to match what src.models.anomaly.trainer
# .run_anomaly_training actually encodes against) alongside the model
# artefact, instead of being rebuilt from that CSV at every service startup.
CATEGORY_MAPPINGS_PATH: str = "model_registry/anomaly/v1.0/category_mappings.json"

HIGH_RISK_SCORE_THRESHOLD: float = 0.7
HIGH_VALUE_THRESHOLD_NGN: float = 1_000_000


class AnomalyService(BaseInferenceService):
    """Wraps the trained Isolation Forest for live anomaly scoring."""

    def __init__(self) -> None:
        super().__init__("anomaly")
        self.scaler = None
        self.thresholds: dict = load_thresholds()
        self._category_mappings: dict[str, dict[str, int]] = {}

    @staticmethod
    def _load_category_mappings() -> dict[str, dict[str, int]]:
        """Load the persisted training-time category -> code mappings."""
        path = Path(CATEGORY_MAPPINGS_PATH)
        if not path.exists():
            raise RuntimeError(
                f"AnomalyService: required category mappings file not found at "
                f"{CATEGORY_MAPPINGS_PATH!r}. Predictions cannot be computed "
                f"correctly without it — this is a deployment configuration "
                f"error, not a condition to degrade past silently."
            )
        with path.open("r", encoding="utf-8") as f:
            mappings: dict[str, dict[str, int]] = json.load(f)
        missing = [col for col in CATEGORICAL_COLUMNS if col not in mappings]
        if missing:
            raise RuntimeError(
                f"AnomalyService: category mappings file {CATEGORY_MAPPINGS_PATH!r} "
                f"is missing required column(s): {missing}"
            )
        return mappings

    def _encode_category(self, column: str, value: str) -> int:
        """Return the training-time code for ``value``, or -1 if unseen/unmapped."""
        return self._category_mappings.get(column, {}).get(str(value), -1)

    def load_model(self) -> bool:
        """Load the Isolation Forest and its score scaler.

        Handles three artifact shapes: a plain ``{"model": ..., "scaler": ...}``
        dict, one of our ``AnomalyTrainer`` instances (the actual Phase 5
        artifact, unwrapped via the base class helper), or a bare estimator
        with no scaler at all. Never raises.
        """
        try:
            champion_file = self.registry_path / "champion.txt"
            version = champion_file.read_text(encoding="utf-8").strip() if champion_file.exists() else self.model_version
            model_path = self.registry_path / version / "model.pkl"

            loaded = joblib.load(model_path)
            if isinstance(loaded, dict) and "model" in loaded and "scaler" in loaded:
                self.model = loaded["model"]
                self.scaler = loaded["scaler"]
                self.feature_columns = list(loaded.get("feature_columns", []))
            else:
                self.model, self.feature_columns, scaler = self._unwrap_model_artifact(loaded)
                self.scaler = scaler

            self.model_version = version
            self._category_mappings = self._load_category_mappings()
            self.is_loaded = True
            logger.info("anomaly: loaded model version %s from %s", version, model_path)
            return True
        except Exception as error:
            logger.error("anomaly: failed to load model: %s", error)
            self.is_loaded = False
            return False

    def _build_feature_row(self, request: AnomalyRequest) -> pd.DataFrame:
        """Build the single-row feature DataFrame the model expects."""
        amount = float(request.amount_ngn)
        # transaction_timestamp isn't part of AnomalyRequest, so hour_of_day
        # and is_weekend use the real current WAT time instead of a fixed
        # default (unlike severity's hour_of_day/is_business_hours_flag,
        # which this request schema has no equivalent live signal for).
        current = now_wat()

        row = {
            "amount_ngn": amount,
            "log_amount": float(np.log1p(max(amount, 0.0))),
            "ip_risk_score": float(request.ip_risk_score),
            "device_risk_score": float(request.device_risk_score),
            "velocity_encoded": VELOCITY_ENCODING.get(request.velocity_indicator, 0),
            "is_reversal": int(bool(request.is_reversal)),
            "is_chargeback": int(bool(request.is_chargeback)),
            "channel_encoded": self._encode_category("channel", request.channel),
            "transaction_type_encoded": self._encode_category("transaction_type", request.transaction_type),
            "hour_of_day": current.hour,
            "is_weekend": int(current.weekday() >= 5),
        }

        columns = self.feature_columns or list(row.keys())
        return pd.DataFrame([row]).reindex(columns=columns, fill_value=0)

    def _behavioural_signals(self, request: AnomalyRequest) -> list[str]:
        """Return plain-text signals describing why a transaction looks risky."""
        signals: list[str] = []
        if request.velocity_indicator == "High":
            signals.append("Transaction velocity above normal threshold")
        if request.ip_risk_score > HIGH_RISK_SCORE_THRESHOLD:
            signals.append("High-risk IP address detected")
        if request.device_risk_score > HIGH_RISK_SCORE_THRESHOLD:
            signals.append("High-risk device detected")
        if request.is_reversal is True:
            signals.append("Transaction is a reversal")
        if request.amount_ngn > HIGH_VALUE_THRESHOLD_NGN:
            signals.append("High-value transaction above NGN 1,000,000")
        return signals

    def predict(self, request: AnomalyRequest) -> AnomalyResponse:
        """Score a transaction and return a full AnomalyResponse."""
        if not self.is_loaded and not self.load_model():
            raise RuntimeError("AnomalyService: model is not loaded and could not be loaded")

        X = self._build_feature_row(request)
        raw_score = self.model.decision_function(X)[0]
        negated = -raw_score
        score = self.scaler.transform([[negated]])[0][0] if self.scaler is not None else negated
        anomaly_score = float(np.clip(score, 0.0, 1.0))

        threshold = self.thresholds["anomaly"]["score_threshold"]
        is_suspicious = anomaly_score >= threshold
        requires_review = bool(is_suspicious)

        prediction_id = generate_prediction_id("anomaly")

        log_prediction(
            prediction_id=prediction_id,
            model_name="anomaly",
            model_version=self.model_version,
            input_summary={
                "transaction_id": request.transaction_id,
                "channel": request.channel,
                "amount_ngn": request.amount_ngn,
            },
            output_summary={"anomaly_score": anomaly_score, "is_suspicious": is_suspicious},
            requester_id=request.requester_id or "system",
            requires_review=requires_review,
        )

        return AnomalyResponse(
            prediction_id=prediction_id,
            model_name="anomaly",
            model_version=self.model_version,
            confidence_score=anomaly_score,
            requires_human_review=requires_review,
            explanation=[],
            timestamp_wat=format_wat_timestamp(),
            anomaly_score=anomaly_score,
            is_suspicious=bool(is_suspicious),
            anomaly_type=None,
            behavioural_signals=self._behavioural_signals(request),
        )
