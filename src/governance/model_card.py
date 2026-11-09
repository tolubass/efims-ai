"""Generate a markdown model card for one model version.

Every number in the "Performance" section comes straight from the saved
``metrics.json`` — never typed by hand — so the card can't drift from what
was actually measured. Meant for the bank's risk and compliance team to
review before (and during) a pilot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from src.utils.date_utils import format_wat_timestamp
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_HUMAN_OVERSIGHT: str = (
    "All outputs are advisory. A human must review and approve before any "
    "action is taken on a case."
)
STATUS_NOTE: str = (
    "Trained on SYNTHETIC data only. Not validated on real bank data. "
    "Retraining on real data is required during the pilot."
)
METRICS_NOT_AVAILABLE: str = "Metrics not available"
FAIRNESS_NOT_RUN: str = "Fairness evaluation not yet run"


def _load_json(path: Path) -> Optional[dict]:
    """Return the parsed JSON at ``path``, or None if missing/unreadable."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        logger.warning("model_card: could not read %s (%s)", path, error)
        return None


def _performance_section(metrics: Optional[dict]) -> str:
    """Build the Performance section as a markdown table of numeric metrics."""
    if not metrics:
        return METRICS_NOT_AVAILABLE

    skip = {"confusion_matrix", "classification_report"}
    rows = [
        (key, value)
        for key, value in metrics.items()
        if key not in skip and isinstance(value, (int, float)) and not isinstance(value, bool)
    ]

    if not rows:
        return METRICS_NOT_AVAILABLE

    lines = ["| Metric | Value |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in rows]
    return "\n".join(lines)


def _fairness_section(fairness_report: Optional[dict]) -> str:
    """Build the Fairness section from a saved fairness_report.json, if any."""
    if not fairness_report:
        return FAIRNESS_NOT_RUN

    flagged = fairness_report.get("flagged_columns", [])
    note = fairness_report.get("note", "")

    if flagged:
        lines = [f"Flagged columns: {', '.join(flagged)}"]
    else:
        lines = ["No columns flagged."]
    if note:
        lines.append(f"\n> {note}")
    return "\n".join(lines)


def _limitations_section(limitations: Optional[list]) -> str:
    """Build the Limitations section as a bullet list."""
    if not limitations:
        return "- None recorded."
    return "\n".join(f"- {item}" for item in limitations)


def generate_model_card(
    model_name: str,
    version: str = "v1.0",
    description: str = "",
    training_data: str = "",
    limitations: List[str] = None,
    human_oversight: str = "",
) -> str:
    """Build and return the model card markdown for one model version.

    Reads ``metrics.json`` and ``fairness_report.json`` (if present) from
    ``model_registry/{model_name}/{version}/``; never raises on missing or
    unreadable files.
    """
    registry_dir = Path("model_registry") / model_name / version
    metrics = _load_json(registry_dir / "metrics.json")
    fairness_report = _load_json(registry_dir / "fairness_report.json")

    sections = [
        f"# Model Card: {model_name} ({version})",
        f"Generated {format_wat_timestamp()}",
        "## Purpose",
        description or "Not provided.",
        "## Training data",
        training_data or "Not provided.",
        "## Performance",
        _performance_section(metrics),
        "## Fairness",
        _fairness_section(fairness_report),
        "## Limitations",
        _limitations_section(limitations),
        "## Human oversight",
        human_oversight or DEFAULT_HUMAN_OVERSIGHT,
        "## Status",
        STATUS_NOTE,
    ]

    return "\n\n".join(sections) + "\n"


def save_model_card(model_name: str, version: str = "v1.0", **kwargs) -> str:
    """Generate and save the model card to model_registry/{model_name}/{version}/model_card.md.

    Returns the path as a string, or "" (logged as an error) on failure.
    """
    try:
        markdown = generate_model_card(model_name, version, **kwargs)
        out_path = Path("model_registry") / model_name / version / "model_card.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        logger.info("save_model_card: saved to %s", out_path)
        return str(out_path)
    except Exception as error:
        logger.error("save_model_card: failed to save card for %s %s: %s", model_name, version, error)
        return ""
