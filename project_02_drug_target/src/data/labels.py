"""Create pre-specified binary label variants from Davis Kd values."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


class LabelingError(ValueError):
    """Raised when binary interaction labels cannot be created safely."""


AFFINITY_COLUMN = "affinity_kd_nM"
DEFAULT_THRESHOLDS_NM = (1000.0, 100.0)


@dataclass(frozen=True)
class BinaryLabelSummary:
    """Summary for one pre-specified binary-label definition."""

    label_column: str
    kd_threshold_nM: float
    pKd_threshold: float
    row_count: int
    positive_count: int
    negative_count: int
    positive_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise_thresholds(thresholds_nM: Iterable[float]) -> tuple[float, ...]:
    """Validate and preserve the requested threshold order."""
    try:
        values = tuple(float(value) for value in thresholds_nM)
    except (TypeError, ValueError) as error:
        raise LabelingError("All Kd thresholds must be numeric.") from error

    if not values:
        raise LabelingError("At least one Kd threshold is required.")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise LabelingError("All Kd thresholds must be finite and greater than zero.")
    if len(set(values)) != len(values):
        raise LabelingError("Kd thresholds must not contain duplicates.")

    return values


def label_column_name(threshold_nM: float) -> str:
    """Return a stable, readable column name for one Kd threshold."""
    threshold = _normalise_thresholds([threshold_nM])[0]
    return f"interaction_kd_le_{threshold:g}_nM"


def _validated_affinities(table: pd.DataFrame) -> pd.Series:
    """Return finite, positive Kd values suitable for thresholding."""
    if AFFINITY_COLUMN not in table.columns:
        raise LabelingError(
            f"Interaction table is missing required column: {AFFINITY_COLUMN}"
        )
    if table.empty:
        raise LabelingError("Interaction table is empty.")

    affinities = pd.to_numeric(table[AFFINITY_COLUMN], errors="coerce")
    if affinities.isna().any():
        raise LabelingError("Affinity values must be present and numeric.")
    if not np.isfinite(affinities.to_numpy(dtype=float)).all():
        raise LabelingError("Affinity values must be finite.")
    if (affinities <= 0).any():
        raise LabelingError("Affinity values must be greater than zero.")

    return affinities


def add_binary_interaction_labels(
    table: pd.DataFrame,
    thresholds_nM: Iterable[float] = DEFAULT_THRESHOLDS_NM,
) -> pd.DataFrame:
    """Return a copy with one binary-label column per pre-specified threshold."""
    thresholds = _normalise_thresholds(thresholds_nM)
    affinities = _validated_affinities(table)

    labeled = table.copy()
    for threshold in thresholds:
        labeled[label_column_name(threshold)] = (
            affinities <= threshold
        ).astype("int8")

    return labeled


def summarize_binary_labels(
    labeled_table: pd.DataFrame,
    thresholds_nM: Iterable[float] = DEFAULT_THRESHOLDS_NM,
) -> list[BinaryLabelSummary]:
    """Summarize prevalence for each binary-label variant."""
    thresholds = _normalise_thresholds(thresholds_nM)
    row_count = int(len(labeled_table))

    if row_count == 0:
        raise LabelingError("Labeled interaction table is empty.")

    summaries: list[BinaryLabelSummary] = []
    for threshold in thresholds:
        column = label_column_name(threshold)
        if column not in labeled_table.columns:
            raise LabelingError(f"Labeled interaction table is missing column: {column}")

        values = labeled_table[column]
        if values.isna().any():
            raise LabelingError(f"Label column contains missing values: {column}")

        invalid_values = set(values.unique()).difference({0, 1})
        if invalid_values:
            raise LabelingError(
                f"Label column must contain only 0 and 1 values: {column}"
            )

        positive_count = int(values.sum())
        negative_count = row_count - positive_count

        summaries.append(
            BinaryLabelSummary(
                label_column=column,
                kd_threshold_nM=float(threshold),
                pKd_threshold=float(-math.log10(threshold / 1e9)),
                row_count=row_count,
                positive_count=positive_count,
                negative_count=negative_count,
                positive_rate=float(positive_count / row_count),
            )
        )

    return summaries


def write_labeled_table(table: pd.DataFrame, output_path: str | Path) -> Path:
    """Write local, ignored labelled interactions for later split construction."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return destination


def write_label_summary(
    summaries: list[BinaryLabelSummary],
    output_path: str | Path,
) -> Path:
    """Write a compact version-controlled summary without raw interaction data."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "continuous_outcome": "pKd = -log10(Kd_nM / 1e9)",
        "binary_label_variants": [summary.to_dict() for summary in summaries],
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create pre-specified Davis binary-label variants."
    )
    parser.add_argument(
        "--input-table",
        type=Path,
        default=Path("data/interim/davis_interactions.csv"),
        help="Local interaction table created by src.data.interaction_table.",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("data/interim/davis_interactions_labeled.csv"),
        help="Local labelled interaction-table destination.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/davis_binary_label_summary.json"),
        help="Version-controlled label-summary destination.",
    )
    parser.add_argument(
        "--thresholds-nm",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS_NM,
        help="Pre-specified Kd thresholds in nM, in reporting order.",
    )
    args = parser.parse_args(argv)

    try:
        table = pd.read_csv(args.input_table)
        labeled = add_binary_interaction_labels(table, args.thresholds_nm)
        summaries = summarize_binary_labels(labeled, args.thresholds_nm)
        table_path = write_labeled_table(labeled, args.table_output)
        summary_path = write_label_summary(summaries, args.summary_output)
    except (LabelingError, OSError, pd.errors.ParserError) as error:
        print(f"Binary-label construction failed: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            [summary.to_dict() for summary in summaries],
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Labeled interaction table written to: {table_path}")
    print(f"Label summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())