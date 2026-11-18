"""Unit tests for src/features: feature builders must never leak target
labels or post-investigation columns into their output, even when the
source table contains them.
"""

from __future__ import annotations

import pandas as pd

from src.features.incident_features import build_incident_features
from src.features.recovery_features import build_recovery_features
from src.features.transaction_features import build_transaction_features

FORBIDDEN_COLUMNS = {
    "fraud_label",
    "severity_label",
    "duplicate_label",
    "recovery_amount_ngn",
    "recovery_ngn",
    "net_loss_ngn",
    "recovery_probability_ground_truth",
    "finding_outcome",
}


def test_build_incident_features_excludes_target_labels():
    incidents = pd.DataFrame(
        {
            "incident_id": ["INC-1"],
            "split": ["train"],
            "source": ["Customer Complaint"],
            "channel": ["Mobile Banking"],
            "incident_type": ["Account Takeover"],
            "reported_by_type": ["Customer"],
            "financial_exposure_ngn": [1000.0],
            "customer_impact_count": [1],
            "ongoing_threat": [True],
            "insider_involvement": [False],
            "data_compromise": [False],
            "regulatory_relevance": [False],
            "narrative_length": [10],
            # received_timestamp must be present (even if the value itself is
            # null downstream): build_incident_features crashes with
            # AttributeError if this column is entirely absent from the
            # input, because pd.to_datetime(None) returns a scalar NaT, not
            # a Series, and .dt then fails. Flagged separately, not fixed here.
            "received_timestamp": ["2026-01-01T09:00:00"],
            # leak columns that must never appear in the output:
            "severity_label": ["S1"],
            "duplicate_label": [0],
        }
    )
    features = build_incident_features(incidents)
    assert FORBIDDEN_COLUMNS.isdisjoint(features.columns)


def test_build_transaction_features_excludes_fraud_label():
    transactions = pd.DataFrame(
        {
            "transaction_id": ["TXN-1"],
            "split": ["train"],
            "amount_ngn": [1000.0],
            "channel": ["ATM"],
            "transaction_type": ["ATM Withdrawal"],
            "velocity_indicator": ["Normal"],
            "ip_risk_score": [0.1],
            "device_risk_score": [0.1],
            "is_reversal": [False],
            "is_chargeback": [False],
            # leak column that must never appear in the output:
            "fraud_label": [1],
        }
    )
    features = build_transaction_features(transactions)
    assert FORBIDDEN_COLUMNS.isdisjoint(features.columns)


def test_build_recovery_features_excludes_post_investigation_columns():
    recoveries = pd.DataFrame({"case_id": ["CASE-1"], "recovery_success": [True]})
    cases = pd.DataFrame(
        {
            "case_id": ["CASE-1"],
            "incident_id": ["INC-1"],
            "severity": ["S1"],
            "classification": ["Account Takeover"],
            "recovery_success": [True],
            # leak columns that must never appear in the output:
            "recovery_amount_ngn": [500.0],
            "net_loss_ngn": [100.0],
            "finding_outcome": ["Confirmed"],
        }
    )
    losses = pd.DataFrame({"case_id": ["CASE-1"], "gross_actual_loss_ngn": [1000.0]})
    incidents = pd.DataFrame(
        {
            "incident_id": ["INC-1"],
            "channel": ["Mobile Banking"],
            "event_timestamp": ["2026-01-01T00:00:00"],
            "discovery_timestamp": ["2026-01-02T00:00:00"],
        }
    )

    features = build_recovery_features(recoveries, cases, losses, incidents)
    assert FORBIDDEN_COLUMNS.isdisjoint(features.columns)
    assert "days_to_detection" in features.columns
    assert features.loc[0, "days_to_detection"] == 1.0
