"""Train the anomaly detection model: an unsupervised Isolation Forest.

fraud_label is never seen during training. It is only used afterwards, to
evaluate how well the anomaly scores line up with confirmed fraud.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from src.features.transaction_features import get_anomaly_feature_matrix
from src.models.base_model import BaseModel
from src.preprocessing.cleaning import clean_transactions
from src.utils.config_loader import load_model_config, load_thresholds
from src.utils.id_generator import generate_run_id
from src.utils.logger import get_logger
from src.utils.metrics import false_positive_rate, precision_at_k

logger = get_logger(__name__)

MLFLOW_EXPERIMENT: str = "efims-anomaly"
PRECISION_AT_K: int = 50


class AnomalyTrainer(BaseModel):
    """Unsupervised Isolation Forest for transaction anomaly scoring."""

    def __init__(self) -> None:
        super().__init__("anomaly", "v1.0")
        self.config: dict = load_model_config("anomaly")
        self.contamination: float = self.config["training"]["contamination"]
        self.scaler: MinMaxScaler | None = None

    def train(self, X: pd.DataFrame) -> None:
        """Fit the Isolation Forest, with no labels at all.

        Also fits a MinMaxScaler on the negated raw scores, so that later
        scores can be rescaled to 0-1 with higher meaning more anomalous.
        """
        params = dict(self.config["hyperparameters"])
        model = IsolationForest(**params)
        model.fit(X)

        raw_scores = model.decision_function(X)
        scaler = MinMaxScaler()
        scaler.fit((-raw_scores).reshape(-1, 1))

        self.model = model
        self.scaler = scaler
        self.feature_columns = list(X.columns)
        self.is_trained = True

        logger.info("AnomalyTrainer: training complete | rows=%d | contamination=%.3f", len(X), self.contamination)

    def predict(self, X: pd.DataFrame) -> dict:
        """Score transactions. Higher anomaly_scores (0-1) mean more suspicious."""
        if not self.is_trained:
            raise RuntimeError("AnomalyTrainer: model is not trained yet")

        X_aligned = X[self.feature_columns]
        raw_scores = self.model.decision_function(X_aligned)
        scaled = self.scaler.transform((-raw_scores).reshape(-1, 1)).flatten()
        scaled = np.clip(scaled, 0.0, 1.0)

        threshold = load_thresholds()["anomaly"]["score_threshold"]
        is_suspicious = scaled >= threshold

        return {
            "anomaly_scores": scaled.tolist(),
            "is_suspicious": is_suspicious.tolist(),
        }

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> dict:
        """Evaluate scores against fraud_label: ROC-AUC, precision@k, and FPR."""
        if not self.is_trained:
            raise RuntimeError("AnomalyTrainer: model is not trained yet")

        result = self.predict(X)
        scores = np.array(result["anomaly_scores"])
        is_suspicious = np.array(result["is_suspicious"], dtype=int)
        y_true_arr = np.asarray(y_true)

        threshold = load_thresholds()["anomaly"]["score_threshold"]

        if len(np.unique(y_true_arr)) > 1:
            from sklearn.metrics import roc_auc_score

            roc_auc = float(roc_auc_score(y_true_arr, scores))
        else:
            logger.warning("AnomalyTrainer: only one class present in y_true, roc_auc set to 0.0")
            roc_auc = 0.0

        metrics = {
            "model_name": "anomaly",
            "roc_auc": roc_auc,
            "precision_at_k": precision_at_k(y_true_arr, scores, k=PRECISION_AT_K),
            "k_value": PRECISION_AT_K,
            "false_positive_rate": false_positive_rate(y_true_arr, is_suspicious),
            "threshold_used": threshold,
        }

        self.metrics = metrics
        return metrics


def _log_to_mlflow(trainer: "AnomalyTrainer", metrics: dict) -> None:
    """Best-effort MLflow logging; never lets a tracking failure fail training."""
    try:
        # mlflow>=3 refuses to write to a filesystem tracking store (the
        # default here) unless this is set. See:
        # https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=generate_run_id("anomaly")):
            mlflow.log_params(
                {k: v for k, v in trainer.config["hyperparameters"].items() if isinstance(v, (str, int, float, bool))}
            )
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            )
    except Exception as error:
        logger.warning("AnomalyTrainer: MLflow logging failed (continuing without it): %s", error)


def run_anomaly_training(data_path: str = "data/synthetic/08_transactions.csv") -> dict:
    """End-to-end: load, clean, train, evaluate, save, and register the anomaly model.

    Returns the test-split metrics dict.
    """
    logger.info("run_anomaly_training: loading %s", data_path)
    raw = pd.read_csv(data_path)
    cleaned = clean_transactions(raw)

    X = get_anomaly_feature_matrix(cleaned)
    split = cleaned["split"]
    fraud_label = cleaned["fraud_label"]

    X_train = X[split == "train"]
    X_test = X[split == "test"]
    y_test = fraud_label[split == "test"]

    trainer = AnomalyTrainer()
    trainer.train(X_train)
    metrics = trainer.evaluate(X_test, y_test)

    output = trainer.config["output"]
    trainer.save(output["model_path"])

    metrics_path = Path(output["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=str)

    registry_dir = str(Path(output["model_path"]).parent.parent)
    trainer.update_champion(registry_dir, "v1.0")

    _log_to_mlflow(trainer, metrics)

    logger.info("run_anomaly_training: done | roc_auc=%.4f", metrics.get("roc_auc", float("nan")))
    return metrics
