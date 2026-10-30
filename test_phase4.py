import pandas as pd
import numpy as np
from src.preprocessing.cleaning import (
    clean_incidents, clean_transactions,
    clean_cases, mask_pii_for_training
)
from src.preprocessing.validation import (
    validate_incidents, validate_transactions,
    validate_cases, generate_validation_report
)
from src.preprocessing.split_generator import (
    fix_duplicate_group_splits,
    generate_incident_splits,
    generate_transaction_splits
)
from src.features.incident_features import (
    build_incident_features,
    get_severity_feature_matrix
)
from src.features.transaction_features import (
    build_transaction_features,
    get_anomaly_feature_matrix
)
from src.features.recovery_features import (
    build_recovery_features,
    get_recovery_feature_matrix
)

print("Loading CSVs...")
incidents = pd.read_csv("data/synthetic/10_incidents.csv")
transactions = pd.read_csv("data/synthetic/08_transactions.csv")
cases = pd.read_csv("data/synthetic/11_cases.csv")
losses = pd.read_csv("data/synthetic/16_losses.csv")
recoveries = pd.read_csv("data/synthetic/17_recoveries.csv")

print("Testing cleaning...")
clean_inc = clean_incidents(incidents.copy())
assert "narrative_length" in clean_inc.columns
assert clean_inc["financial_exposure_ngn"].isna().sum() == 0
print("clean_incidents: OK - rows:", len(clean_inc))

clean_tx = clean_transactions(transactions.copy())
assert clean_tx["amount_ngn"].min() >= 0
assert clean_tx["velocity_indicator"].isin(
    ["Normal","Elevated","High"]).all()
print("clean_transactions: OK - rows:", len(clean_tx))

clean_c = clean_cases(cases.copy())
print("clean_cases: OK - rows:", len(clean_c))

masked = mask_pii_for_training(
    clean_inc.copy(), ["narrative"]
)
print("mask_pii: OK")

print("Testing validation...")
inc_report = validate_incidents(clean_inc)
assert inc_report["incident_id_unique"]["passed"]
assert inc_report["severity_labels_valid"]["passed"]
print("validate_incidents: OK")

tx_report = validate_transactions(clean_tx)
assert tx_report["amount_non_negative"]["passed"]
print("validate_transactions: OK")

print("Testing splits...")
split_counts = generate_incident_splits(clean_inc)
assert "train" in split_counts
print("incident splits: OK -", split_counts)

tx_splits = generate_transaction_splits(clean_tx)
assert "train" in tx_splits
print("transaction splits: OK -", tx_splits)

print("Testing feature engineering...")
feat_inc = build_incident_features(clean_inc.copy())
assert "log_financial_exposure" in feat_inc.columns
assert "severity_label" not in feat_inc.columns
assert "duplicate_label" not in feat_inc.columns
print("incident features: OK - shape:", feat_inc.shape)

X_sev, y_sev = get_severity_feature_matrix(clean_inc.copy())
print("severity matrix: OK - X:", X_sev.shape, "y:", y_sev.shape)

feat_tx = build_transaction_features(clean_tx.copy())
assert "fraud_label" not in feat_tx.columns
assert "log_amount" in feat_tx.columns
print("transaction features: OK - shape:", feat_tx.shape)

X_anom = get_anomaly_feature_matrix(clean_tx.copy())
assert "fraud_label" not in X_anom.columns
print("anomaly matrix: OK - shape:", X_anom.shape)

feat_rec = build_recovery_features(
    recoveries.copy(), cases.copy(),
    losses.copy(), incidents.copy()
)
assert "recovery_amount_ngn" not in feat_rec.columns
assert "log_gross_loss" in feat_rec.columns
print("recovery features: OK - shape:", feat_rec.shape)

print("ALL PHASE 4 TESTS PASSED")
