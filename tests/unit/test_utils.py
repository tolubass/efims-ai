"""Unit tests for src/utils: id_generator, date_utils, metrics."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from src.utils import config_loader, date_utils, id_generator, metrics


# ---- id_generator ----

def test_generate_prediction_id_format():
    pred_id = id_generator.generate_prediction_id("severity")
    assert re.fullmatch(r"PRED-SEV-\d{8}-[0-9A-F]{6}", pred_id)


def test_generate_audit_id_format():
    audit_id = id_generator.generate_audit_id()
    assert re.fullmatch(r"AUDIT-[0-9a-f]{32}", audit_id)


def test_generate_run_id_format():
    run_id = id_generator.generate_run_id("anomaly")
    assert re.fullmatch(r"RUN-ANO-\d{8}-\d{6}", run_id)


def test_generate_case_ref_format():
    case_ref = id_generator.generate_case_ref()
    assert re.fullmatch(r"CASE-\d{4}-\d{6}", case_ref)


def test_model_code_rejects_empty_model_name():
    with pytest.raises(ValueError):
        id_generator.generate_prediction_id("")


# ---- date_utils ----

def test_is_business_hours_true_on_weekday_morning():
    # Monday 2026-09-21, 10:00 WAT
    dt = datetime(2026, 9, 21, 10, 0, tzinfo=date_utils.WAT)
    assert date_utils.is_business_hours(dt) is True


def test_is_business_hours_false_outside_window():
    # Monday 2026-09-21, 18:00 WAT (after 17:00 close)
    dt = datetime(2026, 9, 21, 18, 0, tzinfo=date_utils.WAT)
    assert date_utils.is_business_hours(dt) is False


def test_is_business_hours_false_on_weekend():
    # Saturday 2026-09-19, 10:00 WAT
    dt = datetime(2026, 9, 19, 10, 0, tzinfo=date_utils.WAT)
    assert date_utils.is_business_hours(dt) is False


def test_is_public_holiday_true_on_independence_day():
    dt = datetime(2026, 10, 1, 9, 0, tzinfo=date_utils.WAT)
    assert date_utils.is_public_holiday(dt) is True


def test_is_public_holiday_false_on_ordinary_day():
    dt = datetime(2026, 3, 15, 9, 0, tzinfo=date_utils.WAT)
    assert date_utils.is_public_holiday(dt) is False


def test_to_wat_treats_naive_datetime_as_utc():
    naive = datetime(2026, 1, 1, 12, 0)  # no tzinfo
    converted = date_utils.to_wat(naive)
    assert converted.hour == 13  # UTC+1
    assert converted.tzinfo is not None


def test_format_wat_timestamp_shape():
    dt = datetime(2026, 9, 23, 14, 52, 15, tzinfo=timezone.utc)
    formatted = date_utils.format_wat_timestamp(dt)
    assert formatted.endswith(" WAT")
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} WAT", formatted)


# ---- metrics ----

def test_false_positive_rate_basic():
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 1]
    # FP=2, TN=1 -> FPR = 2/3
    assert metrics.false_positive_rate(y_true, y_pred) == pytest.approx(2 / 3, abs=1e-4)


def test_false_negative_rate_basic():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 0]
    # FN=2, TP=1 -> FNR = 2/3
    assert metrics.false_negative_rate(y_true, y_pred) == pytest.approx(2 / 3, abs=1e-4)


def test_false_positive_rate_empty_input_returns_zero():
    assert metrics.false_positive_rate([], []) == 0.0


def test_false_positive_rate_rejects_non_binary_labels():
    assert metrics.false_positive_rate([0, 1, 2], [0, 1, 2]) == 0.0


def test_precision_at_k_basic():
    y_true = [1, 0, 1, 0, 0]
    y_scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    # top 2 by score: indices 0,1 -> true labels 1,0 -> precision = 0.5
    assert metrics.precision_at_k(y_true, y_scores, k=2) == pytest.approx(0.5)


def test_precision_at_k_k_exceeds_length_uses_all_items():
    y_true = [1, 0]
    y_scores = [0.9, 0.1]
    assert metrics.precision_at_k(y_true, y_scores, k=10) == pytest.approx(0.5)


def test_precision_at_k_invalid_k_returns_zero():
    assert metrics.precision_at_k([1, 0], [0.9, 0.1], k=0) == 0.0


def test_classification_metrics_empty_input_returns_empty_dict():
    assert metrics.classification_metrics([], []) == {}


# ---- config_loader ----

def test_load_config_missing_file_raises_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        config_loader.load_config("models/not_a_real_config.yaml")


def test_load_model_config_rejects_unknown_model_name():
    with pytest.raises(ValueError):
        config_loader.load_model_config("not-a-real-model")


def test_load_thresholds_returns_known_severity_threshold():
    thresholds = config_loader.load_thresholds()
    assert thresholds["severity"]["confidence_threshold"] == 0.70
