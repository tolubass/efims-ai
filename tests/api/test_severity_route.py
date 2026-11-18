"""POST /api/v1/severity/predict — auth, validation, and response shape."""

from __future__ import annotations

from tests.conftest import AUTH_HEADERS, WRONG_AUTH_HEADERS

URL = "/api/v1/severity/predict"

VALID_PAYLOAD: dict = {
    "incident_id": "INC-TEST-0001",
    "financial_exposure_ngn": 4800000.0,
    "customer_impact_count": 3,
    "ongoing_threat": True,
    "insider_involvement": False,
    "data_compromise": False,
    "regulatory_relevance": True,
    "source": "Customer Complaint",
    "channel": "Mobile Banking",
    "incident_type": "Account Takeover",
    "reported_by_type": "Customer",
}


def test_no_api_key_returns_401(client):
    response = client.post(URL, json=VALID_PAYLOAD)
    assert response.status_code == 401


def test_wrong_api_key_returns_401(client):
    response = client.post(URL, json=VALID_PAYLOAD, headers=WRONG_AUTH_HEADERS)
    assert response.status_code == 401


def test_correct_api_key_returns_200(client):
    response = client.post(URL, json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    assert response.status_code == 200


def test_negative_financial_exposure_returns_422(client):
    payload = {**VALID_PAYLOAD, "financial_exposure_ngn": -500.0}
    response = client.post(URL, json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_response_has_required_fields(client):
    body = client.post(URL, json=VALID_PAYLOAD, headers=AUTH_HEADERS).json()
    assert "prediction_id" in body
    assert "confidence_score" in body
    assert "requires_human_review" in body
    assert "explanation" in body
    assert "top_factors" in body
    assert "disclaimer" in body
    assert body["severity_prediction"] in ("S1", "S2", "S3", "S4")


def test_unset_api_key_returns_503(client, monkeypatch):
    monkeypatch.delenv("EFIMS_API_KEY", raising=False)
    response = client.post(URL, json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    assert response.status_code == 503
