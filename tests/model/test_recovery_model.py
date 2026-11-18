"""The recovery model.pkl and category_mappings.json load, and real
inference produces a probability inside its valid range.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.api.schemas.recovery_schema import RecoveryRequest
from src.inference.recovery_service import RecoveryService

MODEL_PATH = Path("model_registry/recovery/v1.0/model.pkl")
MAPPINGS_PATH = Path("model_registry/recovery/v1.0/category_mappings.json")


def test_model_pkl_exists_and_loads():
    assert MODEL_PATH.exists()
    loaded = joblib.load(MODEL_PATH)
    assert loaded is not None


def test_category_mappings_json_exists_and_loads():
    assert MAPPINGS_PATH.exists()
    mappings = json.loads(MAPPINGS_PATH.read_text(encoding="utf-8"))
    for column in ("classification", "channel"):
        assert column in mappings
        assert len(mappings[column]) > 0


def test_real_prediction_is_within_valid_ranges():
    service = RecoveryService()
    assert service.load_model() is True

    request = RecoveryRequest(
        case_id="CASE-MODEL-TEST",
        gross_actual_loss_ngn=9200000.0,
        severity="S1",
        fraud_type="Account Takeover",
        channel="Mobile Banking",
        days_to_detection=0.5,
        containment_confirmed=True,
    )
    response = service.predict(request)

    assert 0.0 <= response.recovery_probability <= 1.0
    assert isinstance(response.recovery_success_prediction, bool)
