"""Build the feature matrix for the recovery propensity model.

Joins recoveries -> cases -> incidents -> losses, collapses to one row per
case, and engineers a small feature set. Every post-recovery /
post-investigation column (recovery_amount_ngn, recovery_ngn, net_loss_ngn,
recovery_probability_ground_truth, finding_outcome, ...) is deliberately left
out: the output is built column-by-column from a fixed feature list, so none
of them can ride along by accident.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from src.features.incident_features import encode_categoricals
from src.utils.logger import get_logger

logger = get_logger(__name__)

# S1 = most severe. Unknown/missing severities encode to -1.
SEVERITY_ENCODING: dict[str, int] = {"S1": 3, "S2": 2, "S3": 1, "S4": 0}

# Exact, ordered set of model-facing feature columns. case_id and
# recovery_success are carried alongside for splitting/evaluation.
FEATURE_COLUMNS: tuple[str, ...] = (
    "gross_actual_loss_ngn",
    "log_gross_loss",
    "severity_encoded",
    "classification_encoded",
    "channel_encoded",
    "days_to_detection",
)


def compute_days_to_detection(incidents_df: pd.DataFrame) -> pd.Series:
    """Return days from event_timestamp to discovery_timestamp, by incident_id.

    Missing or unparseable timestamps produce NaN for that incident.
    """
    event = pd.to_datetime(incidents_df.get("event_timestamp"), errors="coerce")
    discovery = pd.to_datetime(incidents_df.get("discovery_timestamp"), errors="coerce")
    days = (discovery - event).dt.total_seconds() / 86_400
    return pd.Series(days.values, index=incidents_df["incident_id"], name="days_to_detection")


def build_recovery_features(
    recoveries_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    losses_df: pd.DataFrame,
    incidents_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join recoveries/cases/incidents/losses and build the recovery feature matrix.

    One row per case_id. Never includes recovery_amount_ngn, recovery_ngn,
    net_loss_ngn, recovery_probability_ground_truth, or finding_outcome.

    Returns:
        A DataFrame with columns ``case_id``, ``recovery_success``, and every
        name in ``FEATURE_COLUMNS``, in that order.
    """
    # recoveries_df is the anchor; its own recovery_success is kept unsuffixed,
    # cases_df's copy of the same column is renamed out of the way.
    merged = recoveries_df.merge(
        cases_df[["case_id", "incident_id", "severity", "classification", "recovery_success"]],
        on="case_id",
        how="left",
        suffixes=("", "_case"),
    )

    days_to_detection = compute_days_to_detection(incidents_df)
    merged["days_to_detection"] = merged["incident_id"].map(days_to_detection)
    merged = merged.merge(
        incidents_df[["incident_id", "channel"]], on="incident_id", how="left"
    )

    merged = merged.merge(
        losses_df[["case_id", "gross_actual_loss_ngn"]], on="case_id", how="left"
    )

    # Collapse to one row per case: every joined field is constant within a
    # case, so taking the first row per group is a no-op for those columns
    # and satisfies "take first recovery_success value per case" for cases
    # where recoveries_df has more than one row per case_id.
    collapsed = merged.groupby("case_id", as_index=False).first()

    output = pd.DataFrame()
    output["case_id"] = collapsed["case_id"]
    output["recovery_success"] = collapsed["recovery_success"]

    gross_loss = pd.to_numeric(collapsed["gross_actual_loss_ngn"], errors="coerce").fillna(0.0)
    output["gross_actual_loss_ngn"] = gross_loss
    output["log_gross_loss"] = np.log1p(gross_loss.clip(lower=0))

    output["severity_encoded"] = (
        collapsed["severity"].astype(str).str.strip().str.upper().map(SEVERITY_ENCODING).fillna(-1).astype(int)
    )

    encoded_classification, _ = encode_categoricals(collapsed, ["classification"])
    output["classification_encoded"] = encoded_classification["classification_encoded"]

    encoded_channel, _ = encode_categoricals(collapsed, ["channel"])
    output["channel_encoded"] = encoded_channel["channel_encoded"]

    output["days_to_detection"] = pd.to_numeric(collapsed["days_to_detection"], errors="coerce")

    output = output[["case_id", "recovery_success", *FEATURE_COLUMNS]]

    logger.info(
        "build_recovery_features: shape=%s, columns=%s", output.shape, list(FEATURE_COLUMNS)
    )
    return output


def get_recovery_feature_matrix(
    recoveries_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    losses_df: pd.DataFrame,
    incidents_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for the recovery model: features and recovery_success."""
    features = build_recovery_features(recoveries_df, cases_df, losses_df, incidents_df)
    X = features[list(FEATURE_COLUMNS)]
    y = features["recovery_success"]

    logger.info(
        "get_recovery_feature_matrix: class distribution:\n%s", y.value_counts(dropna=False).to_string()
    )
    return X, y
