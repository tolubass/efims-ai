"""Model evaluation metrics shared by every efims-ai evaluator.

All evaluators compute metrics through this module so results are consistent
and comparable across models.

Metric priorities for EFIMS:

* False positive rate -- must be minimised for fraud models, because falsely
  flagging innocent customers causes real harm.
* False negative rate -- missed fraud is the primary risk, so it is always
  tracked alongside FPR.
* Macro F1 -- the primary metric for severity classification, because S1
  cases are rare but the most critical; macro averaging weights every class
  equally instead of letting the majority class dominate.
* Precision at K -- used to evaluate anomaly detection: of the K most
  anomalous transactions, how many are actually fraud?

Functions never raise on bad input. They log a warning and return 0.0 (or an
empty dict) instead, so a metrics failure cannot crash a training pipeline.
Binary functions expect labels encoded as 0 (negative) and 1 (positive).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

ArrayLike = Union[List[Any], np.ndarray]


def _binary_confusion(y_true: ArrayLike, y_pred: ArrayLike, metric: str) -> Optional[tuple[int, int, int, int]]:
    """Return (tn, fp, fn, tp) for 0/1 labels, or None (with a warning) if unusable."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.size == 0 or true.shape != pred.shape:
        logger.warning("%s: y_true and y_pred must be non-empty and the same length", metric)
        return None
    labels = set(np.unique(np.concatenate([true, pred])).tolist())
    if not labels <= {0, 1}:
        logger.warning("%s requires binary 0/1 labels, got %s", metric, sorted(labels, key=str))
        return None
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_prob: Optional[ArrayLike] = None,
    average: str = "macro",
    model_name: str = "model",
) -> Dict[str, Any]:
    """Compute standard classification metrics as a flat, JSON-serialisable dict.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Optional predicted probabilities. For binary problems, either a
            1-D array of positive-class probabilities or an (n, 2) array. For
            multiclass, an (n, n_classes) array with columns in sorted label order.
        average: Averaging for precision/recall/F1 (``"macro"``, ``"weighted"``,
            ``"micro"``, or ``"binary"``).
        model_name: Name used in the log summary.

    Returns:
        Dict with ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``confusion_matrix`` (nested list), ``classification_report`` (str),
        ``average`` and ``model_name``. When ``y_prob`` is given and the metric is
        defined, also ``roc_auc`` (binary or one-vs-rest multiclass), and for
        binary problems ``pr_auc`` and ``brier_score``. Returns ``{}`` on error.
    """
    try:
        true = np.asarray(y_true)
        pred = np.asarray(y_pred)
        if true.size == 0:
            logger.warning("classification_metrics(%s): empty y_true, returning {}", model_name)
            return {}
        if true.shape != pred.shape:
            logger.warning(
                "classification_metrics(%s): y_true %s and y_pred %s differ in shape",
                model_name, true.shape, pred.shape,
            )
            return {}

        labels = np.unique(np.concatenate([true, pred]))
        metrics: Dict[str, Any] = {
            "model_name": model_name,
            "average": average,
            "accuracy": round(float(accuracy_score(true, pred)), 4),
            "precision": round(float(precision_score(true, pred, average=average, zero_division=0)), 4),
            "recall": round(float(recall_score(true, pred, average=average, zero_division=0)), 4),
            "f1": round(float(f1_score(true, pred, average=average, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist(),
            "classification_report": classification_report(true, pred, zero_division=0),
        }

        if y_prob is not None:
            metrics.update(_probability_metrics(true, np.asarray(y_prob, dtype=float), model_name))

        logger.info(summarise_metrics(metrics, model_name))
        return metrics
    except Exception as error:  # never crash a pipeline over metrics
        logger.warning("classification_metrics(%s) failed: %s", model_name, error)
        return {}


def _probability_metrics(true: np.ndarray, prob: np.ndarray, model_name: str) -> Dict[str, float]:
    """Compute ROC-AUC / PR-AUC / Brier score where defined; skip with a warning otherwise."""
    classes = np.unique(true)
    if len(classes) < 2:
        logger.warning(
            "classification_metrics(%s): only one class in y_true, skipping probability metrics",
            model_name,
        )
        return {}

    result: Dict[str, float] = {}
    try:
        if len(classes) == 2:
            if not set(classes.tolist()) <= {0, 1}:
                logger.warning(
                    "classification_metrics(%s): binary probability metrics need 0/1 labels",
                    model_name,
                )
                return {}
            pos_prob = prob[:, 1] if prob.ndim == 2 else prob
            result["roc_auc"] = round(float(roc_auc_score(true, pos_prob)), 4)
            result["pr_auc"] = round(float(average_precision_score(true, pos_prob)), 4)
            result["brier_score"] = round(float(brier_score_loss(true, pos_prob)), 4)
        else:
            result["roc_auc"] = round(float(roc_auc_score(true, prob, multi_class="ovr")), 4)
    except Exception as error:
        logger.warning("classification_metrics(%s): probability metrics failed: %s", model_name, error)
    return result


def false_positive_rate(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return FPR = FP / (FP + TN) for binary 0/1 labels, rounded to 4 dp.

    Returns 0.0 (with a warning) for non-binary or invalid input, and 0.0 if
    there are no actual negatives.
    """
    counts = _binary_confusion(y_true, y_pred, "false_positive_rate")
    if counts is None:
        return 0.0
    tn, fp, _, _ = counts
    return round(fp / (fp + tn), 4) if (fp + tn) else 0.0


def false_negative_rate(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return FNR = FN / (FN + TP) for binary 0/1 labels, rounded to 4 dp.

    Returns 0.0 (with a warning) for non-binary or invalid input, and 0.0 if
    there are no actual positives.
    """
    counts = _binary_confusion(y_true, y_pred, "false_negative_rate")
    if counts is None:
        return 0.0
    _, _, fn, tp = counts
    return round(fn / (fn + tp), 4) if (fn + tp) else 0.0


def precision_at_k(y_true: ArrayLike, y_scores: ArrayLike, k: int) -> float:
    """Return the fraction of the top-``k`` scored items that are positive (label 1).

    Items are ranked by ``y_scores`` descending (ties keep their original
    order). If ``k`` exceeds the number of items, all items are used. Returns
    0.0 (with a warning) for empty input, mismatched lengths, or ``k < 1``.
    Rounded to 4 dp.
    """
    try:
        true = np.asarray(y_true)
        scores = np.asarray(y_scores, dtype=float)
        if true.size == 0 or true.shape != scores.shape:
            logger.warning("precision_at_k: y_true and y_scores must be non-empty and the same length")
            return 0.0
        if k < 1:
            logger.warning("precision_at_k: k must be >= 1, got %s", k)
            return 0.0
        if k > true.size:
            logger.warning("precision_at_k: k=%d exceeds %d items, using all items", k, true.size)
            k = true.size

        top_k = np.argsort(-scores, kind="stable")[:k]
        return round(float(np.mean(true[top_k] == 1)), 4)
    except Exception as error:
        logger.warning("precision_at_k failed: %s", error)
        return 0.0


def summarise_metrics(metrics_dict: Dict[str, Any], model_name: str) -> str:
    """Return a one-line, human-readable summary of ``classification_metrics()`` output.

    Example: ``"Model: severity | Accuracy: 0.8421 | F1 (macro): 0.7932 | ROC-AUC: 0.9104"``

    Metrics absent from ``metrics_dict`` are omitted.
    """
    parts = [f"Model: {model_name}"]
    if not metrics_dict:
        return f"{parts[0]} | no metrics available"

    average = metrics_dict.get("average", "macro")
    labelled = [
        ("accuracy", "Accuracy"),
        ("f1", f"F1 ({average})"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
    ]
    for key, label in labelled:
        value = metrics_dict.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{label}: {value:.4f}")
    return " | ".join(parts)
