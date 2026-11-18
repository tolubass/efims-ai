"""The anomaly model.pkl and category_mappings.json load, and real
inference produces a score inside its valid range.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.api.schemas.anomaly_schema import AnomalyRequest
from src.inference.anomaly_service import AnomalyService

MODEL_PATH = Path("model_registry/anomaly/v1.0/model.pkl")
MAPPINGS_PATH = Path("model_registry/anomaly/v1.0/category_mappings.json")


def test_model_pkl_exists_and_loads():
    assert MODEL_PATH.exists()
    loaded = joblib.load(MODEL_PATH)
    assert loaded is not None


def test_category_mappings_json_exists_and_loads():
    assert MAPPINGS_PATH.exists()
    mappings = json.loads(MAPPINGS_PATH.read_text(encoding="utf-8"))
    for column in ("channel", "transaction_type"):
        assert column in mappings
        assert len(mappings[column]) > 0


def test_real_prediction_is_within_valid_ranges():
    service = AnomalyService()
    assert service.load_model() is True

    request = AnomalyRequest(
        transaction_id="TXN-MODEL-TEST",
        amount_ngn=2500000.0,
        channel="ATM",
        transaction_type="ATM Withdrawal",
        velocity_indicator="High",
        ip_risk_score=0.85,
        device_risk_score=0.72,
        is_reversal=False,
        is_chargeback=False,
        account_risk_rating="High",
    )
    response = service.predict(request)

    assert 0.0 <= response.anomaly_score <= 1.0
    assert isinstance(response.is_suspicious, bool)
