"""POST /api/v1/anomaly/score — auth, validation, and response shape."""

from __future__ import annotations

from tests.conftest import AUTH_HEADERS, WRONG_AUTH_HEADERS

URL = "/api/v1/anomaly/score"

VALID_PAYLOAD: dict = {
    "transaction_id": "TXN-TEST-0001",
    "amount_ngn": 2500000.0,
    "channel": "ATM",
    "transaction_type": "ATM Withdrawal",
    "velocity_indicator": "High",
    "ip_risk_score": 0.85,
    "device_risk_score": 0.72,
    "is_reversal": False,
    "is_chargeback": False,
    "account_risk_rating": "High",
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


def test_bad_velocity_indicator_returns_422(client):
    payload = {**VALID_PAYLOAD, "velocity_indicator": "Extreme"}
    response = client.post(URL, json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_response_has_required_fields(client):
    body = client.post(URL, json=VALID_PAYLOAD, headers=AUTH_HEADERS).json()
    assert "prediction_id" in body
    assert "confidence_score" in body
    assert "requires_human_review" in body
    assert "explanation" in body
    assert "disclaimer" in body
    assert 0.0 <= body["anomaly_score"] <= 1.0


def test_unset_api_key_returns_503(client, monkeypatch):
    monkeypatch.delenv("EFIMS_API_KEY", raising=False)
    response = client.post(URL, json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    assert response.status_code == 503
