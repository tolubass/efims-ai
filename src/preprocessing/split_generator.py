"""Generate train/validation/test parquet files from each cleaned table.

Reads the existing ``split`` column rather than re-splitting the data, with
one exception: for incidents, a ``duplicate_group_id`` must never be spread
across more than one split (that would leak the same real-world incident
into both train and test). ``fix_duplicate_group_splits`` repairs that before
anything is written.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_SPLITS: tuple[str, ...] = ("train", "validation", "test")
DEFAULT_OUTPUT_DIR: str = "data/splits"

# Filename suffix used on disk for each split value.
_SPLIT_FILE_SUFFIX: dict[str, str] = {"train": "train", "validation": "val", "test": "test"}


def fix_duplicate_group_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Reassign every member of a duplicate_group_id to one shared split.

    For each duplicate_group_id whose incidents currently span more than one
    split, every member is moved to that group's majority split (ties broken
    alphabetically, for a deterministic result). Rows with no
    duplicate_group_id are left untouched.
    """
    df = df.copy()
    if "duplicate_group_id" not in df.columns or "split" not in df.columns:
        logger.warning("fix_duplicate_group_splits: missing duplicate_group_id or split column")
        return df

    grouped = df[df["duplicate_group_id"].notna()].groupby("duplicate_group_id")["split"]
    groups_fixed = 0
    rows_moved = 0

    for group_id, splits in grouped:
        if splits.nunique() <= 1:
            continue

        counts = splits.value_counts()
        top_count = counts.iloc[0]
        majority = sorted(counts[counts == top_count].index)[0]

        mask = df["duplicate_group_id"] == group_id
        rows_moved += int((df.loc[mask, "split"] != majority).sum())
        df.loc[mask, "split"] = majority
        groups_fixed += 1

    logger.info(
        "fix_duplicate_group_splits: fixed %d group(s), moved %d row(s) to a shared split",
        groups_fixed,
        rows_moved,
    )
    return df


def _write_splits(df: pd.DataFrame, name: str, output_dir: str) -> dict:
    """Split df by its split column and write one parquet file per split."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split_value in VALID_SPLITS:
        subset = df[df["split"] == split_value]
        suffix = _SPLIT_FILE_SUFFIX[split_value]
        path = out_dir / f"{name}_{suffix}.parquet"
        subset.to_parquet(path, index=False)
        counts[split_value] = len(subset)

    logger.info("%s splits written to %s: %s", name, out_dir, counts)
    return counts


def generate_incident_splits(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    """Fix duplicate-group splits, then write incidents_{train,val,test}.parquet.

    Returns a dict of row counts per split.
    """
    df = fix_duplicate_group_splits(df)
    return _write_splits(df, "incidents", output_dir)


def generate_transaction_splits(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    """Write transactions_{train,val,test}.parquet. Returns row counts per split."""
    return _write_splits(df, "transactions", output_dir)


def generate_case_splits(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    """Write cases_{train,val,test}.parquet. Returns row counts per split."""
    return _write_splits(df, "cases", output_dir)


def save_split_metadata(splits_info: dict, output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    """Save split_metadata.json recording row counts and when it was generated.

    Args:
        splits_info: Dict of per-dataset split counts, e.g.
            ``{"incidents": {...}, "transactions": {...}, "cases": {...},
            "duplicate_groups_fixed": 2}``. Whatever is passed is written
            as-is, alongside a generation timestamp.
        output_dir: Directory to write split_metadata.json into.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **splits_info,
    }

    path = out_dir / "split_metadata.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    logger.info("save_split_metadata: wrote %s", path)
