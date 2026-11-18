"""Unit tests for src/preprocessing/cleaning.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.cleaning import (
    clean_cases,
    clean_incidents,
    clean_transactions,
    mask_pii_for_training,
)


def test_clean_incidents_fills_missing_narrative_and_computes_length():
    df = pd.DataFrame({"narrative": [None, "short"], "source": ["Customer Complaint", None]})
    cleaned = clean_incidents(df)
    assert cleaned["narrative"].tolist() == ["", "short"]
    assert cleaned["narrative_length"].tolist() == [0, 5]


def test_clean_incidents_coerces_boolean_columns():
    df = pd.DataFrame({"ongoing_threat": ["true", "0", None, "yes"]})
    cleaned = clean_incidents(df)
    assert cleaned["ongoing_threat"].tolist() == [True, False, False, True]


def test_clean_incidents_title_cases_channel():
    df = pd.DataFrame({"channel": ["mobile banking", "ATM"]})
    cleaned = clean_incidents(df)
    assert cleaned["channel"].tolist() == ["Mobile Banking", "Atm"]


def test_clean_transactions_clips_amount_and_risk_scores():
    df = pd.DataFrame(
        {
            "amount_ngn": [-100.0, 500.0],
            "ip_risk_score": [1.5, -0.2],
            "device_risk_score": [0.5, 2.0],
        }
    )
    cleaned = clean_transactions(df)
    assert cleaned["amount_ngn"].tolist() == [0.0, 500.0]
    assert cleaned["ip_risk_score"].tolist() == [1.0, 0.0]
    assert cleaned["device_risk_score"].tolist() == [0.5, 1.0]


def test_clean_transactions_normalises_velocity_indicator():
    df = pd.DataFrame({"velocity_indicator": ["HIGH", "elevated", "weird-value", None]})
    cleaned = clean_transactions(df)
    assert cleaned["velocity_indicator"].tolist() == ["High", "Elevated", "Normal", "Normal"]


def test_clean_cases_uppercases_severity():
    df = pd.DataFrame({"severity": [" s1 ", "s4"]})
    cleaned = clean_cases(df)
    assert cleaned["severity"].tolist() == ["S1", "S4"]


def test_clean_cases_coerces_ngn_columns_to_float():
    df = pd.DataFrame({"gross_actual_loss_ngn": ["1000.5", None]})
    cleaned = clean_cases(df)
    assert cleaned["gross_actual_loss_ngn"].tolist() == [1000.5, 0.0]


def test_mask_pii_for_training_redacts_phone_account_and_email():
    df = pd.DataFrame(
        {
            "narrative": [
                "Call 08012345678 or email fraud@bank.com about account 1234567890."
            ]
        }
    )
    masked = mask_pii_for_training(df, ["narrative"])
    text = masked.loc[0, "narrative"]
    assert "08012345678" not in text
    assert "fraud@bank.com" not in text
    assert "1234567890" not in text
    assert "[PHONE_REDACTED]" in text
    assert "[EMAIL_REDACTED]" in text
    assert "[ACCOUNT_REDACTED]" in text


def test_mask_pii_for_training_does_not_mutate_original():
    df = pd.DataFrame({"narrative": ["Call 08012345678."]})
    mask_pii_for_training(df, ["narrative"])
    assert df.loc[0, "narrative"] == "Call 08012345678."


def test_mask_pii_for_training_skips_missing_column_without_raising():
    df = pd.DataFrame({"narrative": ["fine"]})
    masked = mask_pii_for_training(df, ["not_a_real_column"])
    assert masked.loc[0, "narrative"] == "fine"
