"""POST /api/v1/recovery/predict — auth, validation, and response shape."""

from __future__ import annotations

from tests.conftest import AUTH_HEADERS, WRONG_AUTH_HEADERS

URL = "/api/v1/recovery/predict"

VALID_PAYLOAD: dict = {
    "case_id": "CASE-TEST-0001",
    "gross_actual_loss_ngn": 9200000.0,
    "severity": "S1",
    "fraud_type": "Account Takeover",
    "channel": "Mobile Banking",
    "days_to_detection": 0.5,
    "containment_confirmed": True,
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


def test_bad_severity_returns_422(client):
    payload = {**VALID_PAYLOAD, "severity": "S9"}
    response = client.post(URL, json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_response_has_required_fields(client):
    body = client.post(URL, json=VALID_PAYLOAD, headers=AUTH_HEADERS).json()
    assert "prediction_id" in body
    assert "confidence_score" in body
    assert "requires_human_review" in body
    assert "explanation" in body
    assert "key_recovery_factors" in body
    assert "disclaimer" in body
    assert 0.0 <= body["recovery_probability"] <= 1.0


def test_unset_api_key_returns_503(client, monkeypatch):
    monkeypatch.delenv("EFIMS_API_KEY", raising=False)
    response = client.post(URL, json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    assert response.status_code == 503
