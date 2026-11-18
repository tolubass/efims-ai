"""The severity model.pkl and category_mappings.json load, and real
inference produces values inside their valid ranges.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.api.schemas.severity_schema import SeverityRequest
from src.inference.severity_service import SeverityService

MODEL_PATH = Path("model_registry/severity/v1.0/model.pkl")
MAPPINGS_PATH = Path("model_registry/severity/v1.0/category_mappings.json")


def test_model_pkl_exists_and_loads():
    assert MODEL_PATH.exists()
    loaded = joblib.load(MODEL_PATH)
    assert loaded is not None


def test_category_mappings_json_exists_and_loads():
    assert MAPPINGS_PATH.exists()
    mappings = json.loads(MAPPINGS_PATH.read_text(encoding="utf-8"))
    for column in ("source", "channel", "incident_type", "reported_by_type"):
        assert column in mappings
        assert len(mappings[column]) > 0


def test_real_prediction_is_within_valid_ranges():
    service = SeverityService()
    assert service.load_model() is True

    request = SeverityRequest(
        incident_id="INC-MODEL-TEST",
        financial_exposure_ngn=4800000.0,
        customer_impact_count=3,
        ongoing_threat=True,
        insider_involvement=False,
        data_compromise=False,
        regulatory_relevance=True,
        source="Customer Complaint",
        channel="Mobile Banking",
        incident_type="Account Takeover",
        reported_by_type="Customer",
    )
    response = service.predict(request)

    assert response.severity_prediction in ("S1", "S2", "S3", "S4")
    assert 0.0 <= response.confidence_score <= 1.0
    probs = response.all_class_probabilities
    assert set(probs) == {"S1", "S2", "S3", "S4"}
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 0.01
