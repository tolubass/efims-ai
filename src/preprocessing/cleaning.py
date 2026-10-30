"""Clean raw efims-ai CSVs before validation, splitting, or feature engineering.

Every function here takes a DataFrame and returns a new, cleaned DataFrame —
nothing is modified in place, so callers can always compare the cleaned
result against the original. Cleaning only fixes types, formatting, and
obviously-missing values; it never drops rows or invents business logic.

``mask_pii_for_training`` is separate from the per-table cleaners: it strips
personally identifiable text (phone numbers, account-like numbers, emails)
out of free-text columns and must run on any narrative before it reaches a
training pipeline.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Nigerian mobile prefixes covered by the phone-redaction pattern (0 + prefix
# + 8 digits = 11 digits total). Other valid prefixes (e.g. 091, 071) are not
# in scope per the current redaction spec.
_PHONE_PATTERN = re.compile(r"\b0(?:70|80|81|90)\d{8}\b")
# Any run of 10+ consecutive digits not already caught above (account numbers,
# BVNs, NUBANs, etc). Must run AFTER phone redaction, or 11-digit phone
# numbers would be caught here first and never get the phone-specific label.
_ACCOUNT_PATTERN = re.compile(r"\b\d{10,}\b")
_EMAIL_PATTERN = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")

_BOOL_COLUMNS_INCIDENTS: tuple[str, ...] = (
    "ongoing_threat",
    "insider_involvement",
    "data_compromise",
    "regulatory_relevance",
)
_TITLE_CASE_COLUMNS_INCIDENTS: tuple[str, ...] = ("channel", "incident_type")

_BOOL_COLUMNS_TRANSACTIONS: tuple[str, ...] = ("is_reversal", "is_chargeback")
_RISK_SCORE_COLUMNS: tuple[str, ...] = ("ip_risk_score", "device_risk_score")
_VELOCITY_MAP_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("high", "High"),
    ("elev", "Elevated"),
    ("norm", "Normal"),
)

_CASE_NGN_COLUMNS: tuple[str, ...] = (
    "financial_exposure_ngn",
    "gross_actual_loss_ngn",
    "net_loss_ngn",
    "recovery_ngn",
    "customer_reimbursement_ngn",
)
_ARITHMETIC_TOLERANCE_NGN: float = 1.0


def _string_columns(df: pd.DataFrame) -> list[str]:
    """Return columns holding text, whether stored as object or string dtype."""
    return [
        col
        for col in df.columns
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col])
    ]


def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce a column of mixed True/False/"true"/1/NaN values to bool.

    Missing or unrecognised values become False.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    truthy = {"true", "1", "1.0", "yes", "y", "t"}
    return series.apply(
        lambda v: str(v).strip().lower() in truthy if pd.notna(v) else False
    ).astype(bool)


def clean_incidents(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw incidents table.

    Fills missing narratives, strips string columns, coerces boolean and
    numeric columns, title-cases channel/incident_type, and (re)computes
    narrative_length.
    """
    df = df.copy()
    n_rows = len(df)
    nulls_fixed = 0

    if "narrative" in df.columns:
        nulls_fixed += int(df["narrative"].isna().sum())
        df["narrative"] = df["narrative"].fillna("")

    for col in _string_columns(df):
        df[col] = df[col].astype(str).str.strip()
    # astype(str) turns real NaNs into the literal "nan"; put narrative back
    # (handled above) and restore blanks for any other column that was null.
    for col in _string_columns(df):
        if col == "narrative":
            continue
        df[col] = df[col].replace("nan", "")

    for col in _BOOL_COLUMNS_INCIDENTS:
        if col in df.columns:
            nulls_fixed += int(df[col].isna().sum())
            df[col] = _to_bool(df[col])

    if "financial_exposure_ngn" in df.columns:
        nulls_fixed += int(df["financial_exposure_ngn"].isna().sum())
        df["financial_exposure_ngn"] = (
            pd.to_numeric(df["financial_exposure_ngn"], errors="coerce").fillna(0.0).astype(float)
        )

    if "customer_impact_count" in df.columns:
        nulls_fixed += int(df["customer_impact_count"].isna().sum())
        df["customer_impact_count"] = (
            pd.to_numeric(df["customer_impact_count"], errors="coerce").fillna(0).astype(int)
        )

    for col in _TITLE_CASE_COLUMNS_INCIDENTS:
        if col in df.columns:
            df[col] = df[col].str.title()

    df["narrative_length"] = df.get("narrative", pd.Series([""] * n_rows)).str.len()

    logger.info(
        "clean_incidents: cleaned %d rows, fixed %d null values", n_rows, nulls_fixed
    )
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw transactions table.

    Clips amounts and risk scores to valid ranges, coerces boolean columns,
    and standardises velocity_indicator to exactly Normal/Elevated/High.
    """
    df = df.copy()

    if "amount_ngn" in df.columns:
        df["amount_ngn"] = (
            pd.to_numeric(df["amount_ngn"], errors="coerce").fillna(0.0).clip(lower=0.0)
        )

    for col in _BOOL_COLUMNS_TRANSACTIONS:
        if col in df.columns:
            df[col] = _to_bool(df[col])

    if "velocity_indicator" in df.columns:
        df["velocity_indicator"] = df["velocity_indicator"].apply(_normalise_velocity)

    for col in _RISK_SCORE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0.0, upper=1.0)

    logger.info(
        "clean_transactions: cleaned %d rows | amount_ngn mean=%.2f | "
        "ip_risk_score mean=%.3f | device_risk_score mean=%.3f",
        len(df),
        df["amount_ngn"].mean() if "amount_ngn" in df.columns else float("nan"),
        df["ip_risk_score"].mean() if "ip_risk_score" in df.columns else float("nan"),
        df["device_risk_score"].mean() if "device_risk_score" in df.columns else float("nan"),
    )
    return df


def _normalise_velocity(value: object) -> str:
    """Map any spelling/casing of normal/elevated/high to the canonical label.

    Anything unrecognised (including missing values) becomes "Normal".
    """
    if pd.isna(value):
        return "Normal"
    text = str(value).strip().lower()
    for keyword, label in _VELOCITY_MAP_KEYWORDS:
        if keyword in text:
            return label
    return "Normal"


def clean_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw cases table.

    Coerces NGN amount columns to float, standardises severity to uppercase,
    and logs a warning when net_loss_ngn doesn't reconcile with
    gross_actual_loss_ngn - recovery_ngn beyond a 1 NGN tolerance.
    """
    df = df.copy()

    for col in _CASE_NGN_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    if "severity" in df.columns:
        df["severity"] = df["severity"].astype(str).str.strip().str.upper()

    if {"net_loss_ngn", "gross_actual_loss_ngn", "recovery_ngn"} <= set(df.columns):
        expected = df["gross_actual_loss_ngn"] - df["recovery_ngn"]
        mismatches = (df["net_loss_ngn"] - expected).abs() > _ARITHMETIC_TOLERANCE_NGN
        n_mismatches = int(mismatches.sum())
        if n_mismatches:
            logger.warning(
                "clean_cases: %d of %d rows have net_loss_ngn more than %.0f NGN off "
                "from gross_actual_loss_ngn - recovery_ngn",
                n_mismatches,
                len(df),
                _ARITHMETIC_TOLERANCE_NGN,
            )

    logger.info("clean_cases: processed %d rows", len(df))
    return df


def mask_pii_for_training(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    """Redact phone numbers, account-like numbers, and emails from text columns.

    Must be called on any free-text column (e.g. narrative) before it is
    used in a training pipeline. Returns a new DataFrame; the input is not
    modified.

    Args:
        df: DataFrame containing the text columns to mask.
        text_columns: Names of columns to redact.

    Returns:
        A copy of ``df`` with matches replaced by [PHONE_REDACTED],
        [ACCOUNT_REDACTED], or [EMAIL_REDACTED].
    """
    df = df.copy()
    total_replacements = 0

    for col in text_columns:
        if col not in df.columns:
            logger.warning("mask_pii_for_training: column %r not found, skipping", col)
            continue

        series = df[col].astype(str)
        col_replacements = 0

        def _mask(text: str) -> str:
            nonlocal col_replacements
            text, n1 = _PHONE_PATTERN.subn("[PHONE_REDACTED]", text)
            text, n2 = _ACCOUNT_PATTERN.subn("[ACCOUNT_REDACTED]", text)
            text, n3 = _EMAIL_PATTERN.subn("[EMAIL_REDACTED]", text)
            col_replacements += n1 + n2 + n3
            return text

        df[col] = series.apply(_mask)
        total_replacements += col_replacements

    logger.info(
        "mask_pii_for_training: made %d redactions across %d column(s)",
        total_replacements,
        len(text_columns),
    )
    return df
