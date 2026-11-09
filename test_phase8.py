import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.governance.fairness_evaluator import (
    add_age_group, evaluate_fairness, save_fairness_report)
from src.governance.model_card import generate_model_card, save_model_card

MODEL_NAME = "demo_test"
VERSION = "v1.0"
REGISTRY_DIR = Path("model_registry") / MODEL_NAME

print("Testing add_age_group...")
ages = pd.Series([30, None])
bands = add_age_group(ages)
assert bands.iloc[0] == "25-34", bands.iloc[0]
assert bands.iloc[1] == "Unknown", bands.iloc[1]
print("add_age_group: OK")

print("Building fake fairness data...")
rng = np.random.default_rng(42)
n = 120

# "Female": 60 rows, predictions mostly correct.
# "Male": 51 rows, predictions much worse (clear disparity).
# "Other": 9 rows, deliberately below min_group_size=10.
gender = np.array(["Female"] * 60 + ["Male"] * 51 + ["Other"] * 9)
state = np.array(["Lagos"] * 60 + ["Kano"] * 60)

y_true = rng.integers(0, 2, size=n)
y_pred = y_true.copy()

female_idx = np.where(gender == "Female")[0]
male_idx = np.where(gender == "Male")[0]
other_idx = np.where(gender == "Other")[0]

# Flip a few Female predictions (small error rate).
flip_female = rng.choice(female_idx, size=3, replace=False)
y_pred[flip_female] = 1 - y_pred[flip_female]

# Flip most Male predictions (large error rate -> much worse F1).
flip_male = rng.choice(male_idx, size=40, replace=False)
y_pred[flip_male] = 1 - y_pred[flip_male]

# Other group: leave as-is (perfect), but it's too small to count anyway.

sensitive_df = pd.DataFrame({"gender": gender, "state": state})

report = evaluate_fairness(
    y_true, y_pred, sensitive_df,
    group_columns=["gender", "state"],
    model_name=MODEL_NAME,
    disparity_threshold=0.10,
    min_group_size=10,
)

assert "gender" in report["flagged_columns"], report["flagged_columns"]
assert report["columns"]["gender"]["groups"]["Other"]["insufficient_data"] is True
print("evaluate_fairness: OK - flagged:", report["flagged_columns"])

print("Testing model card generation...")
if REGISTRY_DIR.exists():
    shutil.rmtree(REGISTRY_DIR)

metrics_dir = REGISTRY_DIR / VERSION
metrics_dir.mkdir(parents=True, exist_ok=True)
with (metrics_dir / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump({"accuracy": 0.81, "f1": 0.77, "confusion_matrix": [[1, 2], [3, 4]]}, f)

fairness_path = save_fairness_report(report, MODEL_NAME, VERSION)
assert fairness_path != ""
print("save_fairness_report: OK -", fairness_path)

card_path = save_model_card(
    MODEL_NAME, VERSION,
    description="Demo card for phase 8 testing.",
    training_data="Synthetic fake data generated in test_phase8.py.",
    limitations=["Synthetic data only.", "Tiny sample size."],
)
assert card_path != ""
print("save_model_card: OK -", card_path)

markdown = Path(card_path).read_text(encoding="utf-8")
for required in ("Purpose", "Performance", "Fairness", "Limitations", "SYNTHETIC"):
    assert required in markdown, f"missing {required!r} in model card"
print("model card content: OK")

print("Cleaning up demo_test folder...")
shutil.rmtree(REGISTRY_DIR)
assert not REGISTRY_DIR.exists()
print("cleanup: OK")

print("ALL PHASE 8 TESTS PASSED")
