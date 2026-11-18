"""Unit tests for src/governance: human_review_flag, prediction_audit_logger."""

from __future__ import annotations

import json

from src.governance import human_review_flag, prediction_audit_logger


# ---- human_review_flag ----
# configs/base/thresholds.yaml: global=0.60, severity=0.70.

def test_severity_below_threshold_requires_review():
    assert human_review_flag.requires_human_review(0.69, "severity") is True


def test_severity_above_threshold_does_not_require_review():
    assert human_review_flag.requires_human_review(0.95, "severity") is False


def test_unknown_model_falls_back_to_global_threshold():
    # global=0.60: 0.59 is below it, 0.61 is at/above it.
    assert human_review_flag.requires_human_review(0.59, "not-a-real-model") is True
    assert human_review_flag.requires_human_review(0.61, "not-a-real-model") is False


def test_get_threshold_returns_severity_confidence_threshold():
    assert human_review_flag.get_threshold("severity") == 0.70


# ---- prediction_audit_logger.mask_audit_input ----

def test_mask_audit_input_redacts_phone_number():
    masked = prediction_audit_logger.mask_audit_input(
        {"narrative": "Customer called from 08012345678 about the incident."}
    )
    assert "08012345678" not in masked["narrative"]
    assert "[PHONE_REDACTED]" in masked["narrative"]


def test_mask_audit_input_redacts_account_number():
    masked = prediction_audit_logger.mask_audit_input({"note": "Account 1234567890 flagged."})
    assert "1234567890" not in masked["note"]
    assert "[ACCOUNT_REDACTED]" in masked["note"]


def test_mask_audit_input_redacts_email():
    masked = prediction_audit_logger.mask_audit_input({"note": "Reach customer at jane.doe@example.com."})
    assert "jane.doe@example.com" not in masked["note"]
    assert "[EMAIL_REDACTED]" in masked["note"]


def test_mask_audit_input_truncates_long_strings():
    long_text = "x" * 150
    masked = prediction_audit_logger.mask_audit_input({"narrative": long_text})
    assert len(masked["narrative"]) <= 100 + len("...[truncated for audit]")
    assert masked["narrative"].endswith("...[truncated for audit]")


def test_mask_audit_input_leaves_non_string_values_unchanged():
    masked = prediction_audit_logger.mask_audit_input({"amount": 4800000.0, "count": 3})
    assert masked == {"amount": 4800000.0, "count": 3}


def test_mask_audit_input_does_not_mutate_original():
    original = {"narrative": "Call 08012345678 now."}
    prediction_audit_logger.mask_audit_input(original)
    assert original["narrative"] == "Call 08012345678 now."


# ---- prediction_audit_logger.log_prediction (writes to the patched LOG_DIR) ----

def test_log_prediction_writes_masked_entry_to_redirected_log_dir():
    prediction_audit_logger.log_prediction(
        prediction_id="PRED-TEST-0001",
        model_name="severity",
        model_version="v1.0",
        input_summary={"narrative": "Customer at 08012345678 reported fraud."},
        output_summary={"severity_prediction": "S2"},
        requester_id="test-suite",
        requires_review=False,
    )

    log_files = list(prediction_audit_logger.LOG_DIR.glob("audit_*.jsonl"))
    assert len(log_files) == 1

    lines = log_files[0].read_text(encoding="utf-8").strip().splitlines()
    entry = json.loads(lines[-1])
    assert entry["prediction_id"] == "PRED-TEST-0001"
    assert "08012345678" not in entry["input_summary"]["narrative"]
    assert "[PHONE_REDACTED]" in entry["input_summary"]["narrative"]


def test_log_prediction_never_raises_on_bad_input():
    # output_summary with a non-JSON-serialisable-by-default value (a set)
    # is still handled: json.dumps uses default=str, so this must not raise.
    prediction_audit_logger.log_prediction(
        prediction_id="PRED-TEST-0002",
        model_name="anomaly",
        model_version="v1.0",
        input_summary={"note": "fine"},
        output_summary={"weird": {1, 2, 3}},
    )
