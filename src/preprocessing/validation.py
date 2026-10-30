"""Validate cleaned efims-ai data before it is split or trained on.

Each ``validate_*`` function runs a fixed set of logical checks against one
cleaned table and returns a plain dict: one entry per check (pass/fail plus
counts), so the result is always JSON-serialisable and easy to assert on in
tests. ``generate_validation_report`` runs all three and writes the combined
result to ``data/processed/validation_report.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_SEVERITIES: tuple[str, ...] = ("S1", "S2", "S3", "S4")
VALID_DUPLICATE_LABELS: tuple[int, ...] = (0, 1, 2)
VALID_SPLITS: tuple[str, ...] = ("train", "validation", "test")
ARITHMETIC_TOLERANCE_NGN: float = 1.0

DEFAULT_REPORT_PATH: str = "data/processed/validation_report.json"


def _check(passed: bool, **extra: Any) -> Dict[str, Any]:
    """Build one check's result dict, coercing numpy scalars to native types."""
    result: Dict[str, Any] = {"passed": bool(passed)}
    for key, value in extra.items():
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        elif isinstance(value, (np.bool_,)):
            value = bool(value)
        result[key] = value
    return result


def _finalise(report: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Add an overall_passed flag summarising every check in the report."""
    report = dict(report)
    report["overall_passed"] = all(check["passed"] for check in report.values())
    return report


def validate_incidents(df: pd.DataFrame) -> dict:
    """Validate the cleaned incidents table. Returns a report dict."""
    report: Dict[str, Dict[str, Any]] = {}

    null_ids = int(df["incident_id"].isna().sum()) if "incident_id" in df.columns else len(df)
    report["no_null_incident_id"] = _check(null_ids == 0, null_count=null_ids)

    dupes = int(df["incident_id"].duplicated().sum()) if "incident_id" in df.columns else len(df)
    report["incident_id_unique"] = _check(dupes == 0, duplicate_count=dupes)

    if "severity_label" in df.columns:
        invalid = int((~df["severity_label"].isin(VALID_SEVERITIES)).sum())
    else:
        invalid = len(df)
    report["severity_labels_valid"] = _check(invalid == 0, invalid_count=invalid)

    if "duplicate_label" in df.columns:
        invalid = int((~df["duplicate_label"].isin(VALID_DUPLICATE_LABELS)).sum())
    else:
        invalid = len(df)
    report["duplicate_labels_valid"] = _check(invalid == 0, invalid_count=invalid)

    if "financial_exposure_ngn" in df.columns:
        negative = int((df["financial_exposure_ngn"] < 0).sum())
    else:
        negative = len(df)
    report["financial_exposure_non_negative"] = _check(negative == 0, negative_count=negative)

    if "split" in df.columns:
        invalid = int((~df["split"].isin(VALID_SPLITS)).sum())
    else:
        invalid = len(df)
    report["split_values_valid"] = _check(invalid == 0, invalid_count=invalid)

    report = _finalise(report)
    logger.info(
        "validate_incidents: %s (%d rows checked)",
        "PASSED" if report["overall_passed"] else "FAILED",
        len(df),
    )
    return report


def validate_transactions(df: pd.DataFrame) -> dict:
    """Validate the cleaned transactions table. Returns a report dict."""
    report: Dict[str, Dict[str, Any]] = {}

    null_ids = int(df["transaction_id"].isna().sum()) if "transaction_id" in df.columns else len(df)
    report["no_null_transaction_id"] = _check(null_ids == 0, null_count=null_ids)

    dupes = (
        int(df["transaction_id"].duplicated().sum()) if "transaction_id" in df.columns else len(df)
    )
    report["transaction_id_unique"] = _check(dupes == 0, duplicate_count=dupes)

    if "amount_ngn" in df.columns:
        negative = int((df["amount_ngn"] < 0).sum())
    else:
        negative = len(df)
    report["amount_non_negative"] = _check(negative == 0, negative_count=negative)

    if "fraud_label" in df.columns:
        invalid = int((~df["fraud_label"].isin([0, 1])).sum())
    else:
        invalid = len(df)
    report["fraud_label_valid"] = _check(invalid == 0, invalid_count=invalid)

    out_of_range = 0
    for col in ("ip_risk_score", "device_risk_score"):
        if col in df.columns:
            out_of_range += int(((df[col] < 0) | (df[col] > 1)).sum())
        else:
            out_of_range += len(df)
    report["risk_scores_in_range"] = _check(out_of_range == 0, out_of_range_count=out_of_range)

    if "velocity_indicator" in df.columns:
        invalid = int((~df["velocity_indicator"].isin(["Normal", "Elevated", "High"])).sum())
    else:
        invalid = len(df)
    report["velocity_indicator_valid"] = _check(invalid == 0, invalid_count=invalid)

    report = _finalise(report)
    logger.info(
        "validate_transactions: %s (%d rows checked)",
        "PASSED" if report["overall_passed"] else "FAILED",
        len(df),
    )
    return report


def validate_cases(df: pd.DataFrame) -> dict:
    """Validate the cleaned cases table. Returns a report dict."""
    report: Dict[str, Dict[str, Any]] = {}

    null_ids = int(df["case_id"].isna().sum()) if "case_id" in df.columns else len(df)
    report["no_null_case_id"] = _check(null_ids == 0, null_count=null_ids)

    dupes = int(df["case_id"].duplicated().sum()) if "case_id" in df.columns else len(df)
    report["case_id_unique"] = _check(dupes == 0, duplicate_count=dupes)

    if "gross_actual_loss_ngn" in df.columns:
        negative = int((df["gross_actual_loss_ngn"] < 0).sum())
    else:
        negative = len(df)
    report["gross_actual_loss_non_negative"] = _check(negative == 0, negative_count=negative)

    if "net_loss_ngn" in df.columns:
        negative = int((df["net_loss_ngn"] < 0).sum())
    else:
        negative = len(df)
    report["net_loss_non_negative"] = _check(negative == 0, negative_count=negative)

    if "severity" in df.columns:
        invalid = int((~df["severity"].isin(VALID_SEVERITIES)).sum())
    else:
        invalid = len(df)
    report["severity_valid"] = _check(invalid == 0, invalid_count=invalid)

    if {"net_loss_ngn", "gross_actual_loss_ngn", "recovery_ngn"} <= set(df.columns):
        expected = df["gross_actual_loss_ngn"] - df["recovery_ngn"]
        arithmetic_errors = int(
            ((df["net_loss_ngn"] - expected).abs() > ARITHMETIC_TOLERANCE_NGN).sum()
        )
    else:
        arithmetic_errors = len(df)
    report["net_loss_arithmetic_valid"] = _check(
        arithmetic_errors == 0, arithmetic_error_count=arithmetic_errors
    )

    report = _finalise(report)
    logger.info(
        "validate_cases: %s (%d rows checked)",
        "PASSED" if report["overall_passed"] else "FAILED",
        len(df),
    )
    return report


def generate_validation_report(
    incidents_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    output_path: str = DEFAULT_REPORT_PATH,
) -> dict:
    """Run all three validators and save the combined report as JSON.

    Returns the combined report dict (also written to ``output_path``).
    """
    report = {
        "incidents": validate_incidents(incidents_df),
        "transactions": validate_transactions(transactions_df),
        "cases": validate_cases(cases_df),
    }
    report["overall_passed"] = all(
        report[table]["overall_passed"] for table in ("incidents", "transactions", "cases")
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    logger.info(
        "generate_validation_report: overall %s | incidents=%s transactions=%s cases=%s | saved to %s",
        "PASSED" if report["overall_passed"] else "FAILED",
        report["incidents"]["overall_passed"],
        report["transactions"]["overall_passed"],
        report["cases"]["overall_passed"],
        out_path,
    )
    return report
