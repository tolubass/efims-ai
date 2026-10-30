"""Load YAML configuration files from the project's ``configs/`` directory.

This is the single entry point for reading configuration in efims-ai. Models,
pipelines, and services must load settings through these functions rather than
opening YAML files themselves.

    from src.utils.config_loader import load_config, load_model_config

    severity_cfg = load_model_config("severity")
    env_cfg = load_config("environments/development.yaml")

Paths are resolved relative to ``<project_root>/configs`` (derived from this
file's location), so loading works regardless of the current working directory.
An empty YAML file loads as an empty dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# src/utils/config_loader.py -> parents[2] is the project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"

MODEL_NAMES: tuple[str, ...] = ("severity", "duplicate", "anomaly", "recovery", "nlp")


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML file from the ``configs/`` directory.

    Args:
        config_path: Path relative to ``configs/``,
            e.g. ``"models/severity_config.yaml"``.

    Returns:
        The parsed YAML as a dict. An empty file returns ``{}``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid YAML, or its top level is not a
            mapping (e.g. a bare list or string).
    """
    full_path = CONFIG_DIR / config_path

    if not full_path.is_file():
        logger.error("Config file not found: %s", full_path)
        raise FileNotFoundError(
            f"Config file '{config_path}' not found. Expected it at: {full_path}"
        )

    try:
        with full_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        logger.error("Invalid YAML in %s: %s", full_path, error)
        raise ValueError(f"Config file '{config_path}' is not valid YAML: {error}") from error

    if data is None:
        logger.warning("Config file is empty, returning {}: %s", full_path)
        return {}

    if not isinstance(data, dict):
        logger.error(
            "Config file %s must contain a mapping at the top level, got %s",
            full_path,
            type(data).__name__,
        )
        raise ValueError(
            f"Config file '{config_path}' must contain key/value pairs at the "
            f"top level, got {type(data).__name__}."
        )

    logger.info("Loaded config: %s", config_path)
    return data


def load_model_config(model_name: str) -> dict[str, Any]:
    """Load ``configs/models/{model_name}_config.yaml``.

    Args:
        model_name: One of ``severity``, ``duplicate``, ``anomaly``,
            ``recovery``, ``nlp``.

    Returns:
        The parsed model config as a dict.

    Raises:
        ValueError: If ``model_name`` is not a known model.
        FileNotFoundError: If the model's config file does not exist.
    """
    if model_name not in MODEL_NAMES:
        logger.error("Unknown model name: %r", model_name)
        raise ValueError(
            f"Unknown model name '{model_name}'. Expected one of: {', '.join(MODEL_NAMES)}"
        )
    return load_config(f"models/{model_name}_config.yaml")


def load_thresholds() -> dict[str, Any]:
    """Load ``configs/base/thresholds.yaml``.

    These thresholds decide when a prediction is flagged for human review
    (used by ``governance/human_review_flag.py``).
    """
    return load_config("base/thresholds.yaml")


def load_base_config() -> dict[str, Any]:
    """Load ``configs/base/model_config.yaml``.

    Contains shared settings such as data paths, random seed, and date ranges.
    """
    return load_config("base/model_config.yaml")
