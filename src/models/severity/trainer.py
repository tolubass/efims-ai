"""Train the severity classification model (S1/S2/S3/S4, LightGBM multiclass).

S1 is the most critical, rarest class; macro-averaged metrics are used
throughout so S1 performance isn't hidden by the far more common S3/S4 cases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import f1_score

from src.features.incident_features import get_severity_feature_matrix
from src.models.base_model import BaseModel
from src.preprocessing.cleaning import clean_incidents
from src.utils.config_loader import load_base_config, load_model_config
from src.utils.id_generator import generate_run_id
from src.utils.logger import get_logger
from src.utils.metrics import classification_metrics

logger = get_logger(__name__)

SEVERITY_LABELS: Tuple[str, ...] = ("S1", "S2", "S3", "S4")
LABEL_TO_INDEX: dict = {label: index for index, label in enumerate(SEVERITY_LABELS)}
INDEX_TO_LABEL: dict = {index: label for label, index in LABEL_TO_INDEX.items()}

MLFLOW_EXPERIMENT: str = "efims-severity"


class SeverityTrainer(BaseModel):
    """LightGBM multiclass classifier for incident severity (S1-S4)."""

    def __init__(self) -> None:
        super().__init__("severity", "v1.0")
        self.config: dict = load_model_config("severity")
        self.base_config: dict = load_base_config()
        self.random_seed: int = self.base_config["random_seed"]

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the LightGBM classifier on S1-S4 labels encoded as 0-3."""
        y_encoded = y.map(LABEL_TO_INDEX)

        params = dict(self.config["hyperparameters"])
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y_encoded)

        self.model = model
        self.is_trained = True
        self.feature_columns = list(X.columns)

        logger.info(
            "SeverityTrainer: training complete | n_estimators=%d | rows=%d",
            params.get("n_estimators", -1),
            len(X),
        )

    def predict(self, X: pd.DataFrame) -> dict:
        """Predict severity for one or more incidents.

        Returns a single-row dict when ``X`` has one row, or a dict of
        parallel lists when it has more than one.
        """
        if not self.is_trained:
            raise RuntimeError("SeverityTrainer: model is not trained yet")

        X_aligned = X[self.feature_columns]
        probabilities = self.model.predict_proba(X_aligned)
        class_indices = probabilities.argmax(axis=1)

        if len(X_aligned) == 1:
            idx = int(class_indices[0])
            probs = probabilities[0]
            return {
                "severity_prediction": INDEX_TO_LABEL[idx],
                "confidence_score": float(probs[idx]),
                "all_class_probabilities": {
                    SEVERITY_LABELS[i]: float(p) for i, p in enumerate(probs)
                },
            }

        return {
            "severity_prediction": [INDEX_TO_LABEL[int(i)] for i in class_indices],
            "confidence_score": [float(row[i]) for row, i in zip(probabilities, class_indices)],
            "all_class_probabilities": [
                {SEVERITY_LABELS[i]: float(p) for i, p in enumerate(row)} for row in probabilities
            ],
        }

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate on held-out data: macro metrics plus per-class F1."""
        if not self.is_trained:
            raise RuntimeError("SeverityTrainer: model is not trained yet")

        X_aligned = X[self.feature_columns]
        y_encoded = y.map(LABEL_TO_INDEX)
        y_pred = self.model.predict(X_aligned)
        y_prob = self.model.predict_proba(X_aligned)

        metrics = classification_metrics(
            y_encoded, y_pred, y_prob=y_prob, average="macro", model_name="severity"
        )

        per_class = f1_score(y_encoded, y_pred, average=None, labels=list(range(4)), zero_division=0)
        metrics["per_class_f1"] = {SEVERITY_LABELS[i]: float(score) for i, score in enumerate(per_class)}

        self.metrics = metrics
        return metrics


def _log_to_mlflow(trainer: "SeverityTrainer", metrics: dict) -> None:
    """Best-effort MLflow logging; never lets a tracking failure fail training."""
    try:
        # mlflow>=3 refuses to write to a filesystem tracking store (the kind
        # this project's config points at) unless this is set. See:
        # https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(trainer.base_config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=generate_run_id("severity")):
            mlflow.log_params(
                {k: v for k, v in trainer.config["hyperparameters"].items() if isinstance(v, (str, int, float, bool))}
            )
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            )
    except Exception as error:
        logger.warning("SeverityTrainer: MLflow logging failed (continuing without it): %s", error)


def run_severity_training(data_path: str = "data/synthetic/10_incidents.csv") -> dict:
    """End-to-end: load, clean, train, evaluate, save, and register the severity model.

    Returns the validation metrics dict.
    """
    logger.info("run_severity_training: loading %s", data_path)
    raw = pd.read_csv(data_path)
    cleaned = clean_incidents(raw)

    X, y = get_severity_feature_matrix(cleaned)
    split = cleaned["split"]
    X_train, y_train = X[split == "train"], y[split == "train"]
    X_val, y_val = X[split == "validation"], y[split == "validation"]

    trainer = SeverityTrainer()
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_val, y_val)

    output = trainer.config["output"]
    trainer.save(output["model_path"])

    metrics_path = Path(output["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=str)

    if output.get("features_path"):
        features_path = Path(output["features_path"])
        features_path.parent.mkdir(parents=True, exist_ok=True)
        with features_path.open("w", encoding="utf-8") as file:
            json.dump({"feature_columns": trainer.feature_columns}, file, indent=2)

    registry_dir = str(Path(output["model_path"]).parent.parent)
    trainer.update_champion(registry_dir, "v1.0")

    _log_to_mlflow(trainer, metrics)

    logger.info("run_severity_training: done | %s", {k: v for k, v in metrics.items() if k in ("accuracy", "f1")})
    return metrics
