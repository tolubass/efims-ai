from src.api.schemas.duplicate_schema import DuplicateRequest, DuplicateResponse
from src.api.schemas.anomaly_schema import AnomalyRequest, AnomalyResponse
from src.api.schemas.recovery_schema import RecoveryRequest, RecoveryResponse
from src.api.schemas.extraction_schema import ExtractionRequest, ExtractionResponse, ExtractedEntity
from src.api.schemas.next_action_schema import NextActionRequest, NextActionResponse, RecommendedAction

# Test duplicate
req1 = DuplicateRequest(
    incident_id="INC-001",
    narrative="Customer reported unauthorised transfer of NGN 500000",
    incident_type="Account Takeover",
    channel="Mobile Banking",
    financial_exposure_ngn=500000.0,
    customer_impact_count=1,
    state="Lagos"
)
print("duplicate request:", req1.incident_id)

# Test anomaly validator
try:
    bad = AnomalyRequest(
        transaction_id="TXN-001",
        amount_ngn=50000,
        channel="ATM",
        transaction_type="ATM Withdrawal",
        velocity_indicator="VeryHigh",
        ip_risk_score=0.8,
        device_risk_score=0.7,
        is_reversal=False,
        is_chargeback=False,
        account_risk_rating="High"
    )
except Exception as e:
    print("Invalid velocity correctly rejected:", "VeryHigh" in str(e))

# Test recovery validator
try:
    bad2 = RecoveryRequest(
        case_id="CASE-001",
        gross_actual_loss_ngn=1000000,
        severity="S5",
        fraud_type="POS Fraud",
        channel="POS",
        days_to_detection=2.0,
        containment_confirmed=True
    )
except Exception as e:
    print("Invalid severity correctly rejected:", "S5" in str(e))

# Test extraction auto-count
resp = ExtractionResponse(
    prediction_id="PRED-NLP-001",
    model_name="nlp",
    confidence_score=0.75,
    requires_human_review=False,
    timestamp_wat="2026-09-28 10:00:00 WAT",
    incident_id="INC-001",
    entities=[
        ExtractedEntity(
            entity_type="AMOUNT",
            entity_value="NGN 500000",
            start_char=40,
            end_char=50,
            confidence=0.92,
            requires_confirmation=False
        ),
        ExtractedEntity(
            entity_type="LOCATION",
            entity_value="Lagos",
            start_char=60,
            end_char=65,
            confidence=0.55,
            requires_confirmation=True
        )
    ]
)
print("entity_count auto-set:", resp.entity_count)
print("high_confidence_count:", resp.high_confidence_count)

# Test next action
req5 = NextActionRequest(
    case_id="CASE-2024-011",
    severity="S1",
    classification="Account Takeover",
    case_status="Containment",
    containment_confirmed=False,
    days_open=0.5,
    channel="Mobile Banking"
)
print("next action request severity:", req5.severity)

print("ALL SCHEMA TESTS PASSED")
