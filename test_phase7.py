import time
from fastapi.testclient import TestClient
from src.api.main import app

# This version of Starlette only runs FastAPI's lifespan (startup/shutdown)
# when TestClient is used as a context manager. Without `with`, app.state
# never gets severity_service/anomaly_service/recovery_service set, and
# every route below fails with AttributeError. `client = TestClient(app)`
# on its own is no longer enough.
with TestClient(app) as client:

    print("Testing API endpoints...")

    # Test root
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "EFIMS AI Service"
    assert data["status"] == "running"
    print("GET / : OK")

    # Test health
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    health = resp.json()
    assert "status" in health
    assert "models" in health
    assert "severity" in health["models"]
    assert "anomaly" in health["models"]
    assert "recovery" in health["models"]
    print("GET /api/v1/health : OK - status:", health["status"])
    print("  Models:", {
    k: v["status"] for k, v in health["models"].items()
    })

    # Test severity prediction
    resp = client.post("/api/v1/severity/predict", json={
    "incident_id": "INC-TEST-001",
    "financial_exposure_ngn": 4800000.0,
    "customer_impact_count": 3,
    "ongoing_threat": True,
    "insider_involvement": False,
    "data_compromise": False,
    "regulatory_relevance": True,
    "source": "Customer Complaint",
    "channel": "Mobile Banking",
    "incident_type": "Account Takeover",
    "reported_by_type": "Customer"
    })
    assert resp.status_code == 200
    sev = resp.json()
    assert sev["severity_prediction"] in ["S1","S2","S3","S4"]
    assert "confidence_score" in sev
    assert "requires_human_review" in sev
    assert "disclaimer" in sev
    print("POST /api/v1/severity/predict : OK")
    print("  Prediction:", sev["severity_prediction"])
    print("  Confidence:", sev["confidence_score"])
    print("  Requires review:", sev["requires_human_review"])

    # Test anomaly scoring
    resp = client.post("/api/v1/anomaly/score", json={
    "transaction_id": "TXN-TEST-001",
    "amount_ngn": 2500000.0,
    "channel": "ATM",
    "transaction_type": "ATM Withdrawal",
    "velocity_indicator": "High",
    "ip_risk_score": 0.85,
    "device_risk_score": 0.72,
    "is_reversal": False,
    "is_chargeback": False,
    "account_risk_rating": "High"
    })
    assert resp.status_code == 200
    anom = resp.json()
    assert "anomaly_score" in anom
    assert "is_suspicious" in anom
    assert 0.0 <= anom["anomaly_score"] <= 1.0
    print("POST /api/v1/anomaly/score : OK")
    print("  Anomaly score:", anom["anomaly_score"])
    print("  Is suspicious:", anom["is_suspicious"])

    # Test recovery prediction
    resp = client.post("/api/v1/recovery/predict", json={
    "case_id": "CASE-TEST-001",
    "gross_actual_loss_ngn": 9200000.0,
    "severity": "S1",
    "fraud_type": "Account Takeover",
    "channel": "Mobile Banking",
    "days_to_detection": 0.5,
    "containment_confirmed": True
    })
    assert resp.status_code == 200
    rec = resp.json()
    assert "recovery_probability" in rec
    assert "recommended_recovery_actions" in rec
    assert len(rec["recommended_recovery_actions"]) > 0
    print("POST /api/v1/recovery/predict : OK")
    print("  Recovery probability:", rec["recovery_probability"])
    print("  Actions:", len(rec["recommended_recovery_actions"]))

    # Test validation error
    resp = client.post("/api/v1/severity/predict", json={
    "incident_id": "INC-001",
    "financial_exposure_ngn": -500.0
    })
    assert resp.status_code == 422
    print("Validation error test: OK - 422 returned correctly")

    print("ALL PHASE 7 TESTS PASSED")
    print("Run the live server with:")
    print("  uvicorn src.api.main:app --reload --port 8000")
