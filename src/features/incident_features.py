"""Build the feature matrix for the severity and duplicate models.

Only information available at incident intake time is used — no
post-investigation columns, and no target labels. The returned feature
DataFrame is deliberately built column-by-column rather than copied from the
input, so it is structurally impossible for a leaking column to ride along.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.date_utils import is_business_hours
from src.utils.logger import get_logger

logger = get_logger(__name__)

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "source",
    "channel",
    "incident_type",
    "reported_by_type",
)

# Exact, ordered set of model-facing feature columns. incident_id and split
# are carried alongside these for joining/splitting, but are not features.
FEATURE_COLUMNS: tuple[str, ...] = (
    "financial_exposure_ngn",
    "log_financial_exposure",
    "customer_impact_count",
    "ongoing_threat",
    "insider_involvement",
    "data_compromise",
    "regulatory_relevance",
    "source_encoded",
    "channel_encoded",
    "incident_type_encoded",
    "reported_by_type_encoded",
    "narrative_length",
    "hour_of_day",
    "is_business_hours_flag",
)

UNKNOWN_CATEGORY_CODE: int = -1


def encode_categoricals(df: pd.DataFrame, columns: list) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Label-encode each named column, with a reproducible mapping.

    Each column's distinct values are sorted before encoding, so the same
    data always produces the same codes. An ``{col}_encoded`` column is added
    for every column in ``columns``.

    Args:
        df: Source DataFrame (not modified).
        columns: Column names to encode.

    Returns:
        A tuple of (DataFrame with the new ``{col}_encoded`` columns added,
        dict mapping each column name to its {value: code} dict). At
        inference time, apply a saved mapping with
        ``series.map(mapping).fillna(-1).astype(int)`` so unseen values
        become -1 instead of raising.
    """
    df = df.copy()
    mappings: Dict[str, dict] = {}

    for col in columns:
        values = df[col].astype(str).fillna("")
        uniques = sorted(values.unique())
        mapping = {value: code for code, value in enumerate(uniques)}
        mappings[col] = mapping
        df[f"{col}_encoded"] = (
            values.map(mapping).fillna(UNKNOWN_CATEGORY_CODE).astype(int)
        )

    return df, mappings


def build_incident_features(incidents_df: pd.DataFrame) -> pd.DataFrame:
    """Build the incident feature matrix (plus incident_id and split for joining).

    Expects a cleaned incidents DataFrame (see ``src.preprocessing.cleaning
    .clean_incidents``). Never includes severity_label, duplicate_label, or
    any post_investigation_ column.

    Returns:
        A DataFrame with columns ``incident_id``, ``split``, and every name
        in ``FEATURE_COLUMNS``, in that order.
    """
    source = incidents_df
    encoded, _mappings = encode_categoricals(source, list(CATEGORICAL_COLUMNS))

    features = pd.DataFrame(index=source.index)
    features["incident_id"] = source.get("incident_id")
    features["split"] = source.get("split")

    exposure = pd.to_numeric(source.get("financial_exposure_ngn", 0), errors="coerce").fillna(0.0)
    features["financial_exposure_ngn"] = exposure.astype(float)
    features["log_financial_exposure"] = np.log1p(exposure.clip(lower=0))

    features["customer_impact_count"] = (
        pd.to_numeric(source.get("customer_impact_count", 0), errors="coerce").fillna(0).astype(int)
    )

    for col in ("ongoing_threat", "insider_involvement", "data_compromise", "regulatory_relevance"):
        features[col] = source.get(col, False).astype(bool).astype(int)

    for col in CATEGORICAL_COLUMNS:
        features[f"{col}_encoded"] = encoded[f"{col}_encoded"]

    if "narrative_length" in source.columns:
        features["narrative_length"] = source["narrative_length"].fillna(0).astype(int)
    else:
        narrative = source.get("narrative", "").fillna("").astype(str)
        features["narrative_length"] = narrative.str.len()

    received = pd.to_datetime(source.get("received_timestamp"), errors="coerce")
    features["hour_of_day"] = received.dt.hour.fillna(-1).astype(int)
    features["is_business_hours_flag"] = received.apply(
        lambda ts: int(is_business_hours(ts)) if pd.notna(ts) else 0
    )

    features = features[["incident_id", "split", *FEATURE_COLUMNS]]

    logger.info(
        "build_incident_features: shape=%s, columns=%s",
        features.shape,
        list(FEATURE_COLUMNS),
    )
    return features


def get_severity_feature_matrix(incidents_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for the severity model: features and severity_label."""
    features = build_incident_features(incidents_df)
    X = features[list(FEATURE_COLUMNS)]
    y = incidents_df["severity_label"]

    logger.info("get_severity_feature_matrix: class distribution:\n%s", y.value_counts().to_string())
    return X, y


def get_duplicate_feature_matrix(incidents_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for the duplicate model: features and duplicate_label."""
    features = build_incident_features(incidents_df)
    X = features[list(FEATURE_COLUMNS)]
    y = incidents_df["duplicate_label"]

    logger.info("get_duplicate_feature_matrix: class distribution:\n%s", y.value_counts().to_string())
    return X, y
