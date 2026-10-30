"""Decide whether a prediction must be flagged for mandatory human review.

Nigerian banking AI regulation requires that no model prediction trigger an
action on its own: below a model's confidence threshold (or the global
floor), a human must review the suggestion first. See
``configs/base/thresholds.yaml`` for the actual threshold values.
"""

from __future__ import annotations

from src.utils.config_loader import load_thresholds
from src.utils.logger import get_logger

logger = get_logger(__name__)

# The anomaly model's threshold key is "score_threshold" (higher = more
# suspicious); every other model uses "confidence_threshold".
_SCORE_KEYED_MODELS: tuple[str, ...] = ("anomaly",)
_DEFAULT_GLOBAL_THRESHOLD: float = 0.60


def _threshold_key(model_name: str) -> str:
    """Return the threshold key to read for this model."""
    return "score_threshold" if model_name in _SCORE_KEYED_MODELS else "confidence_threshold"


def get_threshold(model_name: str) -> float:
    """Return the review threshold for ``model_name``.

    Falls back to the global human-review threshold if the model is unknown
    or doesn't define its own threshold key.
    """
    thresholds = load_thresholds()
    global_threshold = thresholds.get("global", {}).get(
        "human_review_threshold", _DEFAULT_GLOBAL_THRESHOLD
    )

    model_config = thresholds.get(model_name)
    if model_config is None:
        logger.warning(
            "get_threshold: unknown model_name %r, falling back to global threshold %.2f",
            model_name,
            global_threshold,
        )
        return global_threshold

    return model_config.get(_threshold_key(model_name), global_threshold)


def requires_human_review(confidence: float, model_name: str) -> bool:
    """Return True if ``confidence`` falls below the model's threshold OR the
    global human-review threshold, whichever is stricter.

    An unknown ``model_name`` falls back to the global threshold alone.
    """
    thresholds = load_thresholds()
    global_threshold = thresholds.get("global", {}).get(
        "human_review_threshold", _DEFAULT_GLOBAL_THRESHOLD
    )

    model_config = thresholds.get(model_name)
    if model_config is None:
        logger.warning(
            "requires_human_review: unknown model_name %r, using only the global threshold",
            model_name,
        )
        model_threshold = global_threshold
    else:
        model_threshold = model_config.get(_threshold_key(model_name), global_threshold)

    needs_review = confidence < model_threshold or confidence < global_threshold

    if needs_review:
        logger.info(
            "requires_human_review: %s prediction at confidence %.4f flagged for review "
            "(model_threshold=%.4f, global_threshold=%.4f)",
            model_name,
            confidence,
            model_threshold,
            global_threshold,
        )

    return needs_review
