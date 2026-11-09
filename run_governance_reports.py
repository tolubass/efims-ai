"""Generate the real fairness report and model card for each trained model.

Run once after training to refresh model_registry/{model}/v1.0/
fairness_report.json and model_card.md for severity, anomaly, and recovery.
No model is retrained here — every prediction comes from the already-saved
model_registry/{model}/v1.0/model.pkl artifact.

Usage:
    c:\\Users\\USER\\Desktop\\efims-ai\\venv\\Scripts\\python.exe run_governance_reports.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.incident_features import get_severity_feature_matrix
from src.features.transaction_features import get_anomaly_feature_matrix
from src.governance.fairness_evaluator import add_age_group, evaluate_fairness, save_fairness_report
from src.governance.model_card import generate_model_card, save_model_card
from src.preprocessing.cleaning import clean_incidents, clean_transactions
from src.utils.logger import get_logger

logger = get_logger(__name__)

INCIDENTS_PATH = "data/synthetic/10_incidents.csv"
TRANSACTIONS_PATH = "data/synthetic/08_transactions.csv"
CUSTOMERS_PATH = "data/synthetic/01_customers.csv"

# Matches configs/models/severity_config.yaml -> evaluation.fairness_columns.
# "state" is NOT pulled from customers.csv: it comes straight from the
# incident/transaction's own location column, so it isn't duplicated here.
GROUP_COLUMNS = ["gender", "age_group", "customer_segment", "state"]


def _customer_demographics() -> pd.DataFrame:
    """Return customer_id + gender + customer_segment + age_group, one row per customer."""
    customers = pd.read_csv(CUSTOMERS_PATH)
    demo = customers[["customer_id", "gender", "customer_segment", "age"]].copy()
    demo["age_group"] = add_age_group(demo["age"])
    return demo.drop(columns=["age"])


def _count_insufficient(report: dict) -> int:
    """Count how many (column, group) slices were too small to score."""
    if not report:
        return 0
    return sum(
        1
        for column_result in report.get("columns", {}).values()
        for group_result in column_result.get("groups", {}).values()
        if group_result.get("insufficient_data")
    )


def run_severity() -> dict:
    """Fairness report + model card for the severity model, using its own validation split."""
    print("\n=== SEVERITY ===")

    cleaned = clean_incidents(pd.read_csv(INCIDENTS_PATH))
    X, y = get_severity_feature_matrix(cleaned)
    val_mask = cleaned["split"] == "validation"
    X_val, y_val = X[val_mask], y[val_mask]

    # Reuse the already-trained model; never retrain here.
    trainer = joblib.load("model_registry/severity/v1.0/model.pkl")
    y_pred = trainer.predict(X_val)["severity_prediction"]

    sensitive_df = (
        cleaned.loc[val_mask, ["customer_id", "state"]]
        .reset_index(drop=True)
        .merge(_customer_demographics(), on="customer_id", how="left")
    )

    report = evaluate_fairness(
        y_val.to_numpy(),
        np.array(y_pred),
        sensitive_df,
        group_columns=GROUP_COLUMNS,
        model_name="severity",
    )
    fairness_path = save_fairness_report(report, "severity", "v1.0") if report else ""

    n_s1 = int((cleaned["severity_label"] == "S1").sum())
    limitations = [
        f"Only {n_s1} S1 (most critical) examples exist in the full dataset "
        "— S1 performance cannot be trusted on this evidence alone.",
        f"The validation set is tiny ({len(X_val)} rows), so metrics (including "
        "this fairness report) are noisy and can shift a lot with more data.",
        "Trained and evaluated on synthetic data only; not yet validated on real incidents.",
    ]
    card_path = save_model_card(
        "severity",
        "v1.0",
        description="Classifies incoming incidents as S1-S4 (most to least critical) to prioritise triage.",
        training_data=(
            "Synthetic incidents from data/synthetic/10_incidents.csv, cleaned via "
            "src.preprocessing.cleaning.clean_incidents and featurised via "
            "src.features.incident_features."
        ),
        limitations=limitations,
    )

    return {
        "model": "severity",
        "fairness_path": fairness_path,
        "card_path": card_path,
        "flagged": report.get("flagged_columns", []) if report else [],
        "insufficient": _count_insufficient(report),
    }


def run_anomaly() -> dict:
    """Fairness report + model card for the anomaly model, using its own test split."""
    print("\n=== ANOMALY ===")

    cleaned = clean_transactions(pd.read_csv(TRANSACTIONS_PATH))
    test_mask = cleaned["split"] == "test"
    test_rows = cleaned[test_mask].reset_index(drop=True)

    X_test = get_anomaly_feature_matrix(test_rows)

    # Reuse the already-trained model; never retrain here.
    trainer = joblib.load("model_registry/anomaly/v1.0/model.pkl")
    prediction = trainer.predict(X_test)  # thresholded internally against the anomaly threshold
    y_pred = np.array(prediction["is_suspicious"], dtype=int)
    y_true = test_rows["fraud_label"].to_numpy()

    sensitive_df = test_rows[["customer_id", "state"]].merge(
        _customer_demographics(), on="customer_id", how="left"
    )

    report = evaluate_fairness(
        y_true,
        y_pred,
        sensitive_df,
        group_columns=GROUP_COLUMNS,
        model_name="anomaly",
    )
    fairness_path = save_fairness_report(report, "anomaly", "v1.0") if report else ""

    limitations = [
        "Unsupervised model: fraud_label was used only for this evaluation, never during training.",
        "Evaluated against synthetically injected fraud patterns — real-world performance is unverified.",
        "Trained and evaluated on synthetic data only; not yet validated on real transactions.",
    ]
    card_path = save_model_card(
        "anomaly",
        "v1.0",
        description="Scores transactions for anomalous behaviour (0-1; higher means more suspicious).",
        training_data=(
            "Synthetic transactions from data/synthetic/08_transactions.csv, cleaned via "
            "src.preprocessing.cleaning.clean_transactions and featurised via "
            "src.features.transaction_features."
        ),
        limitations=limitations,
    )

    return {
        "model": "anomaly",
        "fairness_path": fairness_path,
        "card_path": card_path,
        "flagged": report.get("flagged_columns", []) if report else [],
        "insufficient": _count_insufficient(report),
    }


def run_recovery() -> dict:
    """Model card only for the recovery model — fairness slicing is skipped.

    Only 31 labelled cases exist in total, well below any group size that
    could be sliced by gender/age/segment/state without the result being
    meaningless, so no fairness_report.json is produced for this model.
    """
    print("\n=== RECOVERY ===")

    metrics_path = Path("model_registry/recovery/v1.0/metrics.json")
    roc_auc = None
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as file:
            roc_auc = json.load(file).get("roc_auc")
    roc_auc_text = f"{roc_auc:.3f}" if isinstance(roc_auc, (int, float)) else "unavailable"

    limitations = [
        f"Cross-validated ROC-AUC is weak ({roc_auc_text}) — only a little better than chance; "
        "not reliable enough to act on without a human reviewing every case.",
        "Only 31 labelled recovery cases exist in total — far too few to slice reliably by "
        "demographic group, which is why no fairness report was generated for this model.",
        "Trained and evaluated on synthetic data only; not yet validated on real recoveries.",
    ]

    # model_card.generate_model_card() always renders "Fairness evaluation not
    # yet run" when no fairness_report.json exists for this version. That
    # function isn't ours to edit (it already has content), so the more
    # specific wording the task asked for is substituted into the generated
    # markdown here, in this script, before writing the file ourselves.
    markdown = generate_model_card(
        "recovery",
        "v1.0",
        description=(
            "Predicts the probability that a confirmed-loss case will recover more than "
            "50% of its gross loss, to help prioritise recovery actions."
        ),
        training_data=(
            "Synthetic recoveries, cases, losses, and incidents joined and cleaned per "
            "src.features.recovery_features.build_recovery_features."
        ),
        limitations=limitations,
    )
    markdown = markdown.replace(
        "Fairness evaluation not yet run",
        "Not run: only 31 labelled cases exist, too few to slice reliably by demographic group.",
    )

    card_dir = Path("model_registry/recovery/v1.0")
    card_dir.mkdir(parents=True, exist_ok=True)
    card_path = card_dir / "model_card.md"
    card_path.write_text(markdown, encoding="utf-8")
    logger.info("run_recovery: saved model card to %s", card_path)

    return {
        "model": "recovery",
        "fairness_path": "",
        "card_path": str(card_path),
        "flagged": [],
        "insufficient": None,
        "note": "fairness skipped — only 31 labelled cases",
    }


def main() -> None:
    results = [run_severity(), run_anomaly(), run_recovery()]

    print("\n=== SUMMARY ===")
    for result in results:
        print(f"{result['model']}:")
        print(f"  fairness_report : {result['fairness_path'] or '(skipped)'}")
        print(f"  model_card      : {result['card_path']}")
        print(f"  flagged columns : {result['flagged']}")
        if result.get("insufficient") is not None:
            print(f"  insufficient-data groups: {result['insufficient']}")
        if result.get("note"):
            print(f"  note: {result['note']}")


if __name__ == "__main__":
    main()
