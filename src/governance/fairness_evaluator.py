"""Check whether a model performs noticeably worse for some customer groups.

This is a report for human reviewers, not a gate: it never blocks
deployment on its own. Sensitive attributes (gender, age group, customer
segment, state, ...) are only ever used here, to slice an already-computed
set of predictions — never as model features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.utils.date_utils import format_wat_timestamp
from src.utils.logger import get_logger

logger = get_logger(__name__)

AGE_BINS: List[float] = [18, 25, 35, 45, 55, 65, 150]
AGE_LABELS: List[str] = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
UNKNOWN_AGE_LABEL: str = "Unknown"

FAIRNESS_NOTE: str = (
    "Small groups can produce unreliable gaps. Treat flags as prompts for "
    "human review, not proof of bias."
)


def _to_native(value: Any) -> Any:
    """Convert a numpy scalar (or plain value) to a native, JSON-safe Python type."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def add_age_group(age: pd.Series) -> pd.Series:
    """Bin ages into 18-24 / 25-34 / 45-54 / 55-64 / 65+ bands.

    Missing values and ages outside [18, 150) become "Unknown".
    """
    numeric_age = pd.to_numeric(age, errors="coerce")
    bands = pd.cut(numeric_age, bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return bands.astype(object).where(bands.notna(), UNKNOWN_AGE_LABEL).astype(str)


def _evaluate_group(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> dict:
    """Return {n, accuracy, f1} for one slice of rows."""
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def evaluate_fairness(
    y_true,
    y_pred,
    sensitive_df: pd.DataFrame,
    group_columns: list,
    model_name: str = "model",
    average: str = "macro",
    disparity_threshold: float = 0.10,
    min_group_size: int = 10,
) -> dict:
    """Slice predictions by each sensitive column and flag large F1 gaps.

    Returns a JSON-serialisable report dict, or ``{}`` (logged as an error)
    if ``y_true``, ``y_pred``, and ``sensitive_df`` aren't the same length.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if not (len(y_true_arr) == len(y_pred_arr) == len(sensitive_df)):
        logger.error(
            "evaluate_fairness: length mismatch (y_true=%d, y_pred=%d, sensitive_df=%d); "
            "returning {}",
            len(y_true_arr),
            len(y_pred_arr),
            len(sensitive_df),
        )
        return {}

    sensitive_df = sensitive_df.reset_index(drop=True)
    positions = np.arange(len(y_true_arr))

    columns_report: dict = {}
    flagged_columns: list = []

    for column in group_columns:
        if column not in sensitive_df.columns:
            logger.warning("evaluate_fairness: column %r not found in sensitive_df, skipping", column)
            continue

        values = sensitive_df[column]
        group_results: dict = {}
        eligible_f1: dict = {}

        for group_value in sorted(values.dropna().unique(), key=str):
            mask = (values == group_value).to_numpy()
            group_positions = positions[mask]
            n = int(len(group_positions))

            if n < min_group_size:
                group_results[str(group_value)] = {"n": n, "insufficient_data": True}
                continue

            metrics = _evaluate_group(
                y_true_arr[group_positions], y_pred_arr[group_positions], average
            )
            metrics["insufficient_data"] = False
            group_results[str(group_value)] = metrics
            eligible_f1[str(group_value)] = metrics["f1"]

        if len(eligible_f1) >= 2:
            best_group = max(eligible_f1, key=eligible_f1.get)
            worst_group = min(eligible_f1, key=eligible_f1.get)
            disparity = eligible_f1[best_group] - eligible_f1[worst_group]
        else:
            best_group = None
            worst_group = None
            disparity = None

        flagged = disparity is not None and disparity > disparity_threshold
        if flagged:
            flagged_columns.append(column)
            logger.warning(
                "evaluate_fairness: %s flagged on %r — best group %r (F1=%.4f), "
                "worst group %r (F1=%.4f), disparity=%.4f",
                model_name,
                column,
                best_group,
                eligible_f1[best_group],
                worst_group,
                eligible_f1[worst_group],
                disparity,
            )

        columns_report[column] = {
            "groups": group_results,
            "best_group": best_group,
            "worst_group": worst_group,
            "disparity": disparity,
            "flagged": flagged,
        }

    report = {
        "model_name": model_name,
        "generated_at": format_wat_timestamp(),
        "disparity_threshold": disparity_threshold,
        "min_group_size": min_group_size,
        "columns": columns_report,
        "flagged_columns": flagged_columns,
        "overall_flagged": len(flagged_columns) > 0,
        "note": FAIRNESS_NOTE,
    }

    return json.loads(json.dumps(report, default=_to_native))


def save_fairness_report(report: dict, model_name: str, version: str = "v1.0") -> str:
    """Save ``report`` to model_registry/{model_name}/{version}/fairness_report.json.

    Returns the path as a string, or "" (logged as an error) on failure.
    """
    try:
        out_path = Path("model_registry") / model_name / version / "fairness_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, default=_to_native)
        logger.info("save_fairness_report: saved to %s", out_path)
        return str(out_path)
    except Exception as error:
        logger.error("save_fairness_report: failed to save report for %s %s: %s", model_name, version, error)
        return ""
