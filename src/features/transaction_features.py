"""Build the feature matrix for the anomaly (Isolation Forest) model.

fraud_label is the evaluation target for the anomaly model, never a training
input, so the feature builder constructs its output column-by-column and
never copies it across from the source table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.incident_features import encode_categoricals
from src.utils.logger import get_logger

logger = get_logger(__name__)

VELOCITY_ENCODING: dict[str, int] = {"Normal": 0, "Elevated": 1, "High": 2}

CATEGORICAL_COLUMNS: tuple[str, ...] = ("channel", "transaction_type")

# Exact, ordered set of model-facing feature columns. transaction_id and
# split are carried alongside these for joining/splitting, but are not
# features, and fraud_label is never included at all.
FEATURE_COLUMNS: tuple[str, ...] = (
    "amount_ngn",
    "log_amount",
    "ip_risk_score",
    "device_risk_score",
    "velocity_encoded",
    "is_reversal",
    "is_chargeback",
    "channel_encoded",
    "transaction_type_encoded",
    "hour_of_day",
    "is_weekend",
)

_TIMESTAMP_COLUMNS: tuple[str, ...] = ("transaction_timestamp", "received_timestamp")


def _timestamp_column(df: pd.DataFrame) -> pd.Series:
    """Return the first available timestamp column, parsed to datetime.

    Prefers transaction_timestamp; falls back to received_timestamp if that
    is the only one present. Returns an all-NaT series if neither exists.
    """
    for col in _TIMESTAMP_COLUMNS:
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def build_transaction_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Build the transaction feature matrix (plus transaction_id and split).

    Expects a cleaned transactions DataFrame (see
    ``src.preprocessing.cleaning.clean_transactions``). Never includes
    fraud_label.

    Returns:
        A DataFrame with columns ``transaction_id``, ``split``, and every
        name in ``FEATURE_COLUMNS``, in that order.
    """
    source = transactions_df
    encoded, _mappings = encode_categoricals(source, list(CATEGORICAL_COLUMNS))

    features = pd.DataFrame(index=source.index)
    features["transaction_id"] = source.get("transaction_id")
    features["split"] = source.get("split")

    amount = pd.to_numeric(source.get("amount_ngn", 0), errors="coerce").fillna(0.0).clip(lower=0.0)
    features["amount_ngn"] = amount
    features["log_amount"] = np.log1p(amount)

    for col in ("ip_risk_score", "device_risk_score"):
        features[col] = (
            pd.to_numeric(source.get(col, 0.0), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        )

    features["velocity_encoded"] = (
        source.get("velocity_indicator", "Normal").map(VELOCITY_ENCODING).fillna(0).astype(int)
    )

    for col in ("is_reversal", "is_chargeback"):
        features[col] = source.get(col, False).astype(bool).astype(int)

    for col in CATEGORICAL_COLUMNS:
        features[f"{col}_encoded"] = encoded[f"{col}_encoded"]

    timestamp = _timestamp_column(source)
    features["hour_of_day"] = timestamp.dt.hour.fillna(-1).astype(int)
    features["is_weekend"] = (timestamp.dt.dayofweek >= 5).fillna(False).astype(int)

    features = features[["transaction_id", "split", *FEATURE_COLUMNS]]

    logger.info("build_transaction_features: shape=%s, columns=%s", features.shape, list(FEATURE_COLUMNS))
    return features


def get_anomaly_feature_matrix(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature-only matrix the Isolation Forest trains on.

    No transaction_id, no split, and no fraud_label.
    """
    features = build_transaction_features(transactions_df)
    return features[list(FEATURE_COLUMNS)]
