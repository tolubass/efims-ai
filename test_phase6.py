import pandas as pd
from pathlib import Path
from src.governance.human_review_flag import (
    requires_human_review, get_threshold)
from src.governance.prediction_audit_logger import (
    log_prediction, mask_audit_input)
from src.inference.base_service import BaseInferenceService
from src.inference.severity_service import SeverityService
from src.inference.anomaly_service import AnomalyService
from src.inference.recovery_service import RecoveryService
from src.api.schemas.severity_schema import SeverityRequest
from src.api.schemas.anomaly_schema import AnomalyRequest
from src.api.schemas.recovery_schema import RecoveryRequest

print("Testing governance...")

result = requires_human_review(0.50, "severity")
assert result == True
print("requires_human_review low confidence: OK")

result2 = requires_human_review(0.95, "severity")
assert result2 == False
print("requires_human_review high confidence: OK")

threshold = get_threshold("severity")
assert threshold == 0.70
print("get_threshold severity: OK -", threshold)

masked = mask_audit_input({
    "incident_id": "INC-001",
    "narrative": "Customer 08012345678 called about account 1234567890",
    "financial_exposure_ngn": 500000.0
})
assert "[PHONE_REDACTED]" in masked["narrative"]
assert "[ACCOUNT_REDACTED]" in masked["narrative"]
print("mask_audit_input: OK")

log_prediction(
    prediction_id="PRED-SEV-TEST-001",
    model_name="severity",
    model_version="v1.0",
    input_summary={"incident_id": "INC-001",
                   "channel": "Mobile Banking"},
    output_summary={"severity_prediction": "S2",
                    "confidence_score": 0.84},
    requester_id="test_user",
    requires_review=False
)
audit_files = list(Path("logs").glob("audit_*.jsonl"))
assert len(audit_files) > 0
print("log_prediction: OK - audit file created")

print("Testing inference services...")

sev_service = SeverityService()
loaded = sev_service.load_model()
print("SeverityService load_model:", loaded)

if loaded:
    req = SeverityRequest(
        incident_id="INC-TEST-001",
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
        narrative="Customer reported unauthorised transfers"
    )
    response = sev_service.predict(req)
    assert response.severity_prediction in ["S1","S2","S3","S4"]
    assert 0.0 <= response.confidence_score <= 1.0
    assert isinstance(response.requires_human_review, bool)
    assert "Human review" in response.disclaimer
    print("SeverityService predict: OK")
    print("  Prediction:", response.severity_prediction)
    print("  Confidence:", response.confidence_score)
    print("  Requires review:", response.requires_human_review)

anom_service = AnomalyService()
anom_loaded = anom_service.load_model()
print("AnomalyService load_model:", anom_loaded)

if anom_loaded:
    req2 = AnomalyRequest(
        transaction_id="TXN-TEST-001",
        amount_ngn=2500000.0,
        channel="ATM",
        transaction_type="ATM Withdrawal",
        velocity_indicator="High",
        ip_risk_score=0.85,
        device_risk_score=0.72,
        is_reversal=False,
        is_chargeback=False,
        account_risk_rating="High"
    )
    resp2 = anom_service.predict(req2)
    assert 0.0 <= resp2.anomaly_score <= 1.0
    assert isinstance(resp2.is_suspicious, bool)
    print("AnomalyService predict: OK")
    print("  Anomaly score:", resp2.anomaly_score)
    print("  Is suspicious:", resp2.is_suspicious)

rec_service = RecoveryService()
rec_loaded = rec_service.load_model()
print("RecoveryService load_model:", rec_loaded)

if rec_loaded:
    req3 = RecoveryRequest(
        case_id="CASE-TEST-001",
        gross_actual_loss_ngn=9200000.0,
        severity="S1",
        fraud_type="Account Takeover",
        channel="Mobile Banking",
        days_to_detection=0.5,
        containment_confirmed=True
    )
    resp3 = rec_service.predict(req3)
    assert 0.0 <= resp3.recovery_probability <= 1.0
    assert len(resp3.recommended_recovery_actions) > 0
    print("RecoveryService predict: OK")
    print("  Recovery probability:", resp3.recovery_probability)
    print("  Actions:", len(resp3.recommended_recovery_actions))

health = sev_service.health_check()
assert "status" in health
print("health_check: OK -", health["status"])

print("ALL PHASE 6 TESTS PASSED")
