import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.models.base_model import BaseModel
from src.models.severity.trainer import (
    SeverityTrainer, run_severity_training)
from src.models.anomaly.trainer import (
    AnomalyTrainer, run_anomaly_training)
from src.models.recovery.trainer import (
    RecoveryTrainer, run_recovery_training)

print("Testing BaseModel interface...")
assert hasattr(BaseModel, "save")
assert hasattr(BaseModel, "load")
assert hasattr(BaseModel, "get_champion_version")
print("BaseModel: OK")

print("Running severity training...")
sev_metrics = run_severity_training()
assert "accuracy" in sev_metrics or "f1" in sev_metrics
assert Path("model_registry/severity/v1.0/model.pkl").exists()
assert Path("model_registry/severity/v1.0/metrics.json").exists()
assert Path("model_registry/severity/champion.txt").exists()
print("Severity training: OK - metrics:", {
    k: v for k, v in sev_metrics.items()
    if k in ["accuracy", "f1", "model_name"]
})

print("Running anomaly training...")
anom_metrics = run_anomaly_training()
assert Path("model_registry/anomaly/v1.0/model.pkl").exists()
assert Path("model_registry/anomaly/v1.0/metrics.json").exists()
print("Anomaly training: OK - metrics:", {
    k: v for k, v in anom_metrics.items()
    if k in ["roc_auc", "precision_at_k"]
})

print("Running recovery training...")
rec_metrics = run_recovery_training()
assert Path("model_registry/recovery/v1.0/model.pkl").exists()
assert Path("model_registry/recovery/v1.0/metrics.json").exists()
print("Recovery training: OK - metrics:", {
    k: v for k, v in rec_metrics.items()
    if k in ["roc_auc", "f1", "brier_score"]
})

print("ALL PHASE 5 TESTS PASSED")
