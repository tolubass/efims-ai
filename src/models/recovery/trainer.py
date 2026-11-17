"""Train the recovery propensity model: predicts whether a fraud case will
recover more than 50% of its gross loss.

Only 31 labelled cases exist today, and the provided test split holds just
one of them, so stratified 5-fold cross-validation is used for evaluation
instead of a fixed holdout. The final deployed model is then refit on every
labelled row.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.features.recovery_features import get_recovery_feature_matrix
from src.models.base_model import BaseModel
from src.preprocessing.cleaning import clean_cases, clean_incidents
from src.utils.config_loader import load_model_config
from src.utils.id_generator import generate_run_id
from src.utils.logger import get_logger
from src.utils.metrics import classification_metrics

logger = get_logger(__name__)

MLFLOW_EXPERIMENT: str = "efims-recovery"
CV_FOLDS: int = 5


class RecoveryTrainer(BaseModel):
    """LightGBM binary classifier for recovery propensity."""

    def __init__(self) -> None:
        super().__init__("recovery", "v1.0")
        self.config: dict = load_model_config("recovery")

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the final LightGBM classifier on the full labelled dataset."""
        params = dict(self.config["hyperparameters"])
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)

        self.model = model
        self.feature_columns = list(X.columns)
        self.is_trained = True

        logger.info("RecoveryTrainer: training complete | rows=%d", len(X))

    def predict(self, X: pd.DataFrame) -> dict:
        """Predict recovery success for one or more cases."""
        if not self.is_trained:
            raise RuntimeError("RecoveryTrainer: model is not trained yet")

        X_aligned = X[self.feature_columns]
        probabilities = self.model.predict_proba(X_aligned)[:, 1]
        predictions = probabilities >= 0.5

        if len(X_aligned) == 1:
            return {
                "recovery_success_prediction": bool(predictions[0]),
                "recovery_probability": float(probabilities[0]),
            }

        return {
            "recovery_success_prediction": [bool(p) for p in predictions],
            "recovery_probability": [float(p) for p in probabilities],
        }

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate via stratified 5-fold cross-validation.

        Fits an independent fold model per split (it does not require
        ``train`` to have been called first) and pools out-of-fold
        predictions before scoring.
        """
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True).astype(int)

        params = dict(self.config["hyperparameters"])
        random_state = params.get("random_state", 42)
        splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)

        oof_pred = np.zeros(len(y), dtype=int)
        oof_prob = np.zeros(len(y), dtype=float)

        for train_idx, test_idx in splitter.split(X, y):
            fold_model = lgb.LGBMClassifier(**params)
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof_prob[test_idx] = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
            oof_pred[test_idx] = fold_model.predict(X.iloc[test_idx])

        metrics = classification_metrics(
            y, oof_pred, y_prob=oof_prob, average="binary", model_name="recovery"
        )
        metrics["cv_folds"] = CV_FOLDS

        self.metrics = metrics
        return metrics


def _log_to_mlflow(trainer: "RecoveryTrainer", metrics: dict) -> None:
    """Best-effort MLflow logging; never lets a tracking failure fail training.

    Imports mlflow here, not at module level: this module is re-imported by
    joblib.load() when unpickling a saved RecoveryTrainer at inference time,
    and inference has no need for mlflow at all — only an actual training
    run (which calls this function) does.
    """
    try:
        import mlflow

        # mlflow>=3 refuses to write to a filesystem tracking store (the
        # default here) unless this is set. See:
        # https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=generate_run_id("recovery")):
            mlflow.log_params(
                {k: v for k, v in trainer.config["hyperparameters"].items() if isinstance(v, (str, int, float, bool))}
            )
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            )
    except Exception as error:
        logger.warning("RecoveryTrainer: MLflow logging failed (continuing without it): %s", error)


def run_recovery_training(
    recoveries_path: str = "data/synthetic/17_recoveries.csv",
    cases_path: str = "data/synthetic/11_cases.csv",
    losses_path: str = "data/synthetic/16_losses.csv",
    incidents_path: str = "data/synthetic/10_incidents.csv",
) -> dict:
    """End-to-end: load, clean, cross-validate, refit, save, and register the recovery model.

    Returns the cross-validation metrics dict.
    """
    logger.info("run_recovery_training: loading recoveries/cases/losses/incidents")
    recoveries = pd.read_csv(recoveries_path)
    cases = clean_cases(pd.read_csv(cases_path))
    losses = pd.read_csv(losses_path)
    incidents = clean_incidents(pd.read_csv(incidents_path))

    X, y = get_recovery_feature_matrix(recoveries, cases, losses, incidents)

    X = X.fillna(X.median(numeric_only=True))
    valid = y.notna()
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True).astype(int)

    trainer = RecoveryTrainer()
    metrics = trainer.evaluate(X, y)
    trainer.train(X, y)

    output = trainer.config["output"]
    trainer.save(output["model_path"])

    metrics_path = Path(output["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=str)

    registry_dir = str(Path(output["model_path"]).parent.parent)
    trainer.update_champion(registry_dir, "v1.0")

    _log_to_mlflow(trainer, metrics)

    logger.info("run_recovery_training: done | roc_auc=%s", metrics.get("roc_auc"))
    return metrics
