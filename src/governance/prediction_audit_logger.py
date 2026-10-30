"""Append-only audit trail of every prediction, for CBN / internal audit.

Writes one JSON object per line to ``logs/audit_YYYYMMDD.jsonl``. Never
raises: a broken audit log must never take down a live prediction, so every
failure here is caught and logged instead.

Raw PII (phone numbers, account-like numbers, emails) is never written to the
log; ``mask_audit_input`` strips it before anything is serialised.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.utils.date_utils import format_wat_timestamp, now_wat
from src.utils.id_generator import generate_audit_id
from src.utils.logger import get_logger

logger = get_logger(__name__)

# prediction_audit_logger.py -> parents[2] is the project root (same depth as
# src/utils/logger.py), so the audit log always lands in the same place
# regardless of the caller's current working directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
LOG_DIR: Path = PROJECT_ROOT / "logs"

_PHONE_PATTERN = re.compile(r"\b0(?:70|80|81|90)\d{8}\b")
_ACCOUNT_PATTERN = re.compile(r"\b\d{10,}\b")
_EMAIL_PATTERN = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")

_MAX_FIELD_LENGTH: int = 100
_TRUNCATION_SUFFIX: str = "...[truncated for audit]"


def mask_audit_input(input_dict: dict) -> dict:
    """Return a redacted copy of ``input_dict``; the original is never modified.

    String values have phone numbers, account-like numbers, and emails
    replaced with redaction tags, and are truncated to 100 characters (plus
    a truncation marker) if longer. Non-string values pass through unchanged.
    """
    masked: dict = {}
    for key, value in input_dict.items():
        if isinstance(value, str):
            text = _PHONE_PATTERN.sub("[PHONE_REDACTED]", value)
            text = _ACCOUNT_PATTERN.sub("[ACCOUNT_REDACTED]", text)
            text = _EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
            if len(text) > _MAX_FIELD_LENGTH:
                text = text[:_MAX_FIELD_LENGTH] + _TRUNCATION_SUFFIX
            masked[key] = text
        else:
            masked[key] = value
    return masked


def log_prediction(
    prediction_id: str,
    model_name: str,
    model_version: str,
    input_summary: dict,
    output_summary: dict,
    requester_id: str = "system",
    requires_review: bool = False,
) -> None:
    """Append one audit entry for a prediction to today's JSONL audit log.

    Never raises: any failure (masking, serialisation, or disk I/O) is
    logged as an error and swallowed, so a logging problem can never break a
    live prediction.
    """
    try:
        entry: dict[str, Any] = {
            "audit_id": generate_audit_id(),
            "prediction_id": prediction_id,
            "model_name": model_name,
            "model_version": model_version,
            "timestamp_wat": format_wat_timestamp(),
            "requester_id": requester_id,
            "input_summary": mask_audit_input(input_summary),
            "output_summary": output_summary,
            "requires_human_review": requires_review,
        }

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"audit_{now_wat():%Y%m%d}.jsonl"

        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, default=str) + "\n")
    except Exception as error:  # audit logging must never crash a prediction
        logger.error("log_prediction: failed to write audit entry: %s", error)
