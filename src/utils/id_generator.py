"""Generate unique identifiers used across efims-ai.

Every prediction, audit log entry, and training run is tagged with an ID from
this module. All dates and times are in West Africa Time (WAT, UTC+1).

    PRED-SEV-20260923-A3F2B1               generate_prediction_id("severity")
    AUDIT-550e8400e29b41d4a716446655440000 generate_audit_id()
    RUN-SEV-20260923-141100                generate_run_id("severity")
    CASE-2026-004821                       generate_case_ref()

Pure standard library, no logging, no side effects.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

# Nigeria observes WAT (UTC+1) year-round with no daylight saving.
WAT: timezone = timezone(timedelta(hours=1), name="WAT")


def _now_wat() -> datetime:
    """Return the current time in WAT."""
    return datetime.now(WAT)


def _model_code(model_name: str) -> str:
    """Return the first 3 characters of ``model_name``, uppercased."""
    code = model_name.strip()[:3].upper()
    if not code:
        raise ValueError("model_name must be a non-empty string")
    return code


def generate_prediction_id(model_name: str) -> str:
    """Return a prediction ID: ``PRED-{MODEL}-{YYYYMMDD}-{6 hex chars}``.

    Example: ``generate_prediction_id("severity")`` -> ``PRED-SEV-20260923-A3F2B1``
    """
    suffix = uuid.uuid4().hex[:6].upper()
    return f"PRED-{_model_code(model_name)}-{_now_wat():%Y%m%d}-{suffix}"


def generate_audit_id() -> str:
    """Return an audit ID: ``AUDIT-{32-char UUID4 hex}``.

    Example: ``AUDIT-550e8400e29b41d4a716446655440000``
    """
    return f"AUDIT-{uuid.uuid4().hex}"


def generate_run_id(model_name: str) -> str:
    """Return a training run ID: ``RUN-{MODEL}-{YYYYMMDD}-{HHMMSS}``.

    Example: ``generate_run_id("severity")`` -> ``RUN-SEV-20260923-141100``

    Note: IDs have one-second resolution, so two runs of the same model started
    within the same second will collide.
    """
    return f"RUN-{_model_code(model_name)}-{_now_wat():%Y%m%d-%H%M%S}"


def generate_case_ref(prefix: str = "CASE") -> str:
    """Return a case reference: ``{PREFIX}-{YEAR}-{6-digit number}``.

    Example: ``generate_case_ref()`` -> ``CASE-2026-004821``

    For demo and test data only: with 1,000,000 possible numbers per year,
    collisions are likely once thousands of refs are generated.
    """
    number = uuid.uuid4().int % 1_000_000
    return f"{prefix.upper()}-{_now_wat():%Y}-{number:06d}"
