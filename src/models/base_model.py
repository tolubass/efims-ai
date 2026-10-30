"""Abstract base class every efims-ai model inherits from.

Defines one interface (``train`` / ``predict`` / ``evaluate``) so inference
services can call any model the same way without knowing which algorithm is
underneath, plus shared, concrete persistence and champion-version helpers
that every subclass gets for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CHAMPION_VERSION: str = "v1.0"


class BaseModel(ABC):
    """Common interface and shared behaviour for every efims-ai model.

    Subclasses must implement ``train``, ``predict``, and ``evaluate``.
    ``save``/``load`` (via joblib) and the champion-version helpers are
    implemented here and inherited as-is.
    """

    def __init__(self, model_name: str, version: str) -> None:
        self.model_name: str = model_name
        self.version: str = version
        self.model: Any = None
        self.is_trained: bool = False
        self.feature_columns: list = []
        self.metrics: dict = {}

    @abstractmethod
    def train(self, X, y) -> None:
        """Fit the model. Unsupervised models may override this to drop ``y``."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X) -> dict:
        """Return a prediction (or predictions) for ``X`` as a plain dict."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, X, y) -> dict:
        """Return an evaluation metrics dict for ``X``/``y``."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save this entire model instance (including its trained state) via joblib."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        logger.info("%s: saved to %s", self.model_name, out_path)

    def load(self, path: str) -> None:
        """Load a previously saved instance from ``path`` into this object."""
        in_path = Path(path)
        loaded = joblib.load(in_path)
        self.__dict__.update(loaded.__dict__)
        logger.info("%s: loaded from %s", self.model_name, in_path)

    def get_champion_version(self, registry_path: str) -> str:
        """Return the version recorded in ``champion.txt``, or "v1.0" if absent."""
        champion_file = Path(registry_path) / "champion.txt"
        if not champion_file.exists():
            return DEFAULT_CHAMPION_VERSION
        return champion_file.read_text(encoding="utf-8").strip()

    def update_champion(self, registry_path: str, version: str) -> None:
        """Write ``version`` to ``champion.txt`` in ``registry_path``."""
        registry_dir = Path(registry_path)
        registry_dir.mkdir(parents=True, exist_ok=True)
        champion_file = registry_dir / "champion.txt"
        champion_file.write_text(version, encoding="utf-8")
        logger.info("%s: champion updated to %s at %s", self.model_name, version, champion_file)
