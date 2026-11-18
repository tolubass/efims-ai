"""GET /api/v1/health — must work without any API key."""

from __future__ import annotations


def test_health_returns_200_without_api_key(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_reports_all_three_models(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "healthy"
    assert set(body["models"]) == {"severity", "anomaly", "recovery"}
    for model_status in body["models"].values():
        assert model_status["status"] == "healthy"
