"""Build a leakage-auditable interaction table from Davis benchmark files."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .validate_davis import DavisValidationError, validate_davis_dataset


class InteractionTableError(DavisValidationError):
    """Raised when a Davis interaction table cannot be constructed safely."""


TABLE_COLUMNS = [
    "observed_pair_index",
    "drug_matrix_index",
    "target_matrix_index",
    "matrix_flat_index",
    "drug_id",
    "target_id",
    "affinity_kd_nM",
    "pKd",
]


@dataclass(frozen=True)
class InteractionTableSummary:
    row_count: int
    unique_drug_count: int
    unique_target_count: int
    affinity_kd_min_nM: float
    affinity_kd_max_nM: float
    pKd_min: float
    pKd_max: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_mapping(path: Path, label: str) -> dict[str, str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise InteractionTableError(f"Could not load {label}: {path}") from error

    if not isinstance(payload, dict) or not payload:
        raise InteractionTableError(f"{label} must be a non-empty JSON object.")

    if any(
        not isinstance(identifier, str)
        or not identifier.strip()
        or not isinstance(value, str)
        or not value.strip()
        for identifier, value in payload.items()
    ):
        raise InteractionTableError(
            f"{label} contains an empty or invalid identifier/representation."
        )

    return payload


def _load_affinity_matrix(path: Path) -> np.ndarray:
    # Only unpickle checksum-verified benchmark data from a trusted source.
    try:
        with path.open("rb") as handle:
            matrix = np.asarray(pickle.load(handle, encoding="latin1"))
    except (OSError, pickle.UnpicklingError, EOFError) as error:
        raise InteractionTableError(
            f"Could not load affinity matrix: {path}"
        ) from error

    return matrix


def build_davis_interaction_table(data_dir: str | Path) -> pd.DataFrame:
    """Return one row per observed Davis drug-target affinity measurement.

    Mapping order is intentionally preserved from the DeepDTA JSON files because
    that order corresponds to affinity-matrix rows and columns. Do not sort IDs.
    """
    directory = Path(data_dir)
    validate_davis_dataset(directory)

    ligands = _load_mapping(directory / "ligands_can.txt", "ligands_can.txt")
    proteins = _load_mapping(directory / "proteins.txt", "proteins.txt")
    affinity = _load_affinity_matrix(directory / "Y")

    drug_ids = list(ligands.keys())
    target_ids = list(proteins.keys())
    observed_mask = ~np.isnan(affinity)
    drug_rows, target_columns = np.where(observed_mask)
    observed_affinities = affinity[drug_rows, target_columns].astype(float)

    if observed_affinities.size == 0:
        raise InteractionTableError("No observed affinity values were found.")
    if np.any(observed_affinities <= 0):
        raise InteractionTableError(
            "pKd cannot be calculated because one or more observed Kd values are "
            "zero or negative."
        )

    table = pd.DataFrame(
        {
            "observed_pair_index": np.arange(
                observed_affinities.size, dtype=np.int64
            ),
            "drug_matrix_index": drug_rows.astype(np.int64),
            "target_matrix_index": target_columns.astype(np.int64),
            "matrix_flat_index": (
                drug_rows * affinity.shape[1] + target_columns
            ).astype(np.int64),
            "drug_id": [drug_ids[index] for index in drug_rows],
            "target_id": [target_ids[index] for index in target_columns],
            "affinity_kd_nM": observed_affinities,
        }
    )
    table["pKd"] = -np.log10(table["affinity_kd_nM"] / 1e9)

    if table.duplicated(["drug_id", "target_id"]).any():
        raise InteractionTableError("Interaction table contains duplicate pairs.")

    return table[TABLE_COLUMNS]


def summarize_interaction_table(table: pd.DataFrame) -> InteractionTableSummary:
    """Create a compact, committed summary without redistributing raw data."""
    missing_columns = set(TABLE_COLUMNS).difference(table.columns)
    if missing_columns:
        raise InteractionTableError(
            f"Interaction table is missing columns: {sorted(missing_columns)}"
        )
    if table.empty:
        raise InteractionTableError("Interaction table is empty.")

    return InteractionTableSummary(
        row_count=int(len(table)),
        unique_drug_count=int(table["drug_id"].nunique()),
        unique_target_count=int(table["target_id"].nunique()),
        affinity_kd_min_nM=float(table["affinity_kd_nM"].min()),
        affinity_kd_max_nM=float(table["affinity_kd_nM"].max()),
        pKd_min=float(table["pKd"].min()),
        pKd_max=float(table["pKd"].max()),
    )


def write_interaction_table(table: pd.DataFrame, output_path: str | Path) -> Path:
    """Write derived, ignored interaction data for local reproducibility."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return destination


def write_interaction_summary(
    summary: InteractionTableSummary, output_path: str | Path
) -> Path:
    """Write a small, version-controlled interaction-table summary."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a leakage-auditable Davis interaction table."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/davis"),
        help="Path to the Davis raw-data directory.",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("data/interim/davis_interactions.csv"),
        help="Local CSV destination for the derived interaction table.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/davis_interaction_table_summary.json"),
        help="Version-controlled JSON summary destination.",
    )
    args = parser.parse_args(argv)

    try:
        table = build_davis_interaction_table(args.data_dir)
        summary = summarize_interaction_table(table)
        table_path = write_interaction_table(table, args.table_output)
        summary_path = write_interaction_summary(summary, args.summary_output)
    except DavisValidationError as error:
        print(f"Interaction-table construction failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print(f"Interaction table written to: {table_path}")
    print(f"Interaction summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())