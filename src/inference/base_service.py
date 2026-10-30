"""Abstract base class for every live inference service.

Loads models defensively: a missing or corrupt model file must never crash
the API, so ``load_model`` catches everything and reports success/failure
through a return value instead.

Compatibility note: ``src.models.base_model.BaseModel.save()`` persists the
*entire* trainer instance via joblib, not a bare estimator. ``joblib.load()``
on one of our model files therefore returns a ``BaseModel`` subclass (with
its own ``.model``, ``.feature_columns``, and, for the anomaly trainer,
``.scaler``) rather than a raw scikit-learn/LightGBM object.
``_unwrap_model_artifact`` detects that shape and unwraps it, so subclasses
can treat ``self.model`` as a plain estimator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY_DIR: str = "model_registry"
DEFAULT_CHAMPION_VERSION: str = "v1.0"


class BaseInferenceService(ABC):
    """Common load/health-check behaviour for every model's inference service."""

    def __init__(self, model_name: str, model_version: str = "v1.0") -> None:
        self.model_name: str = model_name
        self.model_version: str = model_version
        self.model: Any = None
        self.is_loaded: bool = False
        self.registry_path: Path = Path(MODEL_REGISTRY_DIR) / model_name
        self.feature_columns: list = []

    @abstractmethod
    def predict(self, request) -> dict:
        """Return a prediction for ``request``. Implemented per model."""
        raise NotImplementedError

    @staticmethod
    def _unwrap_model_artifact(loaded: Any) -> Tuple[Any, list, Optional[Any]]:
        """Unwrap a loaded BaseModel instance into (estimator, feature_columns, scaler).

        If ``loaded`` doesn't look like one of our BaseModel subclasses, it is
        returned as-is (treated as an already-bare estimator).
        """
        if hasattr(loaded, "model") and hasattr(loaded, "is_trained"):
            return loaded.model, list(getattr(loaded, "feature_columns", [])), getattr(loaded, "scaler", None)
        return loaded, [], None

    def load_model(self) -> bool:
        """Load the champion model version from the registry.

        Returns True on success. On any failure, logs the error, sets
        ``is_loaded = False``, and returns False — never raises.
        """
        try:
            champion_file = self.registry_path / "champion.txt"
            version = (
                champion_file.read_text(encoding="utf-8").strip()
                if champion_file.exists()
                else DEFAULT_CHAMPION_VERSION
            )
            model_path = self.registry_path / version / "model.pkl"

            loaded = joblib.load(model_path)
            self.model, self.feature_columns, _scaler = self._unwrap_model_artifact(loaded)
            self.model_version = version
            self.is_loaded = True

            logger.info("%s: loaded model version %s from %s", self.model_name, version, model_path)
            return True
        except Exception as error:
            logger.error("%s: failed to load model: %s", self.model_name, error)
            self.is_loaded = False
            return False

    def health_check(self) -> dict:
        """Return a small status dict suitable for a health endpoint."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "is_loaded": self.is_loaded,
            "status": "healthy" if self.is_loaded else "model_not_loaded",
        }
