"""Structural validation for trusted DeepDTA-format Davis benchmark data."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


class DavisValidationError(ValueError):
    """Raised when a Davis benchmark structural check fails."""


@dataclass(frozen=True)
class DavisValidationReport:
    data_directory: str
    drug_count: int
    target_count: int
    affinity_shape: tuple[int, int]
    affinity_dtype: str
    affinity_missing_values: int
    affinity_non_missing_values: int
    total_pair_count: int
    train_index_count: int
    test_index_count: int
    fold_coverage_count: int
    fold_overlap_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_file(data_dir: Path, relative_path: str) -> Path:
    path = data_dir / relative_path
    if not path.is_file():
        raise DavisValidationError(f"Required Davis file is missing: {path}")
    return path


def _load_json(path: Path, label: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise DavisValidationError(f"{label} is not valid JSON: {path}") from error


def _validate_representation_mapping(
    payload: Any, label: str, representation_name: str
) -> dict[str, str]:
    if not isinstance(payload, dict) or not payload:
        raise DavisValidationError(f"{label} must be a non-empty JSON object.")

    invalid_ids = [
        identifier
        for identifier in payload
        if not isinstance(identifier, str) or not identifier.strip()
    ]
    if invalid_ids:
        raise DavisValidationError(f"{label} contains an empty or invalid identifier.")

    empty_values = [
        identifier
        for identifier, value in payload.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if empty_values:
        preview = ", ".join(empty_values[:3])
        raise DavisValidationError(
            f"{label} contains empty {representation_name} values for: {preview}"
        )

    return payload


def _load_affinity_matrix(path: Path) -> np.ndarray:
    # Only unpickle data obtained from a trusted, checksum-verified source.
    try:
        with path.open("rb") as handle:
            matrix = np.asarray(pickle.load(handle, encoding="latin1"))
    except (OSError, pickle.UnpicklingError, EOFError) as error:
        raise DavisValidationError(f"Could not load affinity matrix: {path}") from error

    if matrix.ndim != 2:
        raise DavisValidationError(
            f"Affinity matrix must be two-dimensional; received shape {matrix.shape}."
        )
    if not np.issubdtype(matrix.dtype, np.number):
        raise DavisValidationError(
            f"Affinity matrix must be numeric; received dtype {matrix.dtype}."
        )
    if np.isinf(matrix).any():
        raise DavisValidationError("Affinity matrix contains infinite values.")

    return matrix


def _flatten_indices(value: Any, label: str) -> list[int]:
    if isinstance(value, bool):
        raise DavisValidationError(f"{label} contains a Boolean, not an integer index.")
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        flattened: list[int] = []
        for item in value:
            flattened.extend(_flatten_indices(item, label))
        return flattened
    raise DavisValidationError(f"{label} contains an unsupported value: {value!r}")


def _load_fold_indices(path: Path, label: str) -> list[int]:
    indices = _flatten_indices(_load_json(path, label), label)
    if not indices:
        raise DavisValidationError(f"{label} contains no indices.")
    return indices


def _validate_fold_indices(indices: list[int], label: str, pair_count: int) -> None:
    invalid = [index for index in indices if index < 0 or index >= pair_count]
    if invalid:
        raise DavisValidationError(
            f"{label} contains out-of-range indices; first invalid value: {invalid[0]}"
        )

    if len(indices) != len(set(indices)):
        raise DavisValidationError(f"{label} contains duplicate interaction indices.")


def validate_davis_dataset(data_dir: str | Path) -> DavisValidationReport:
    """Validate the core DeepDTA-style Davis files and return a summary report."""
    directory = Path(data_dir)
    if not directory.is_dir():
        raise DavisValidationError(f"Davis data directory does not exist: {directory}")

    ligands_path = _require_file(directory, "ligands_can.txt")
    proteins_path = _require_file(directory, "proteins.txt")
    affinity_path = _require_file(directory, "Y")
    train_fold_path = _require_file(directory, "folds/train_fold_setting1.txt")
    test_fold_path = _require_file(directory, "folds/test_fold_setting1.txt")

    ligands = _validate_representation_mapping(
        _load_json(ligands_path, "ligands_can.txt"), "ligands_can.txt", "SMILES"
    )
    proteins = _validate_representation_mapping(
        _load_json(proteins_path, "proteins.txt"), "proteins.txt", "protein sequence"
    )
    affinity = _load_affinity_matrix(affinity_path)

    expected_shape = (len(ligands), len(proteins))
    if affinity.shape != expected_shape:
        raise DavisValidationError(
            "Affinity matrix shape does not match the ligand/target counts: "
            f"expected {expected_shape}, received {affinity.shape}."
        )

    pair_count = int(affinity.size)
    train_indices = _load_fold_indices(train_fold_path, "train_fold_setting1.txt")
    test_indices = _load_fold_indices(test_fold_path, "test_fold_setting1.txt")

    _validate_fold_indices(train_indices, "train_fold_setting1.txt", pair_count)
    _validate_fold_indices(test_indices, "test_fold_setting1.txt", pair_count)

    overlap = set(train_indices).intersection(test_indices)
    if overlap:
        raise DavisValidationError(
            "Train and test folds overlap; first overlapping index: "
            f"{min(overlap)}"
        )

    missing_values = int(np.isnan(affinity).sum())
    non_missing_values = pair_count - missing_values
    if non_missing_values == 0:
        raise DavisValidationError("Affinity matrix contains no observed values.")

    return DavisValidationReport(
        data_directory=str(directory),
        drug_count=len(ligands),
        target_count=len(proteins),
        affinity_shape=tuple(int(value) for value in affinity.shape),
        affinity_dtype=str(affinity.dtype),
        affinity_missing_values=missing_values,
        affinity_non_missing_values=non_missing_values,
        total_pair_count=pair_count,
        train_index_count=len(train_indices),
        test_index_count=len(test_indices),
        fold_coverage_count=len(set(train_indices).union(test_indices)),
        fold_overlap_count=len(overlap),
    )


def write_validation_report(
    report: DavisValidationReport, output_path: str | Path
) -> Path:
    """Write a machine-readable validation report."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a DeepDTA-format Davis benchmark directory."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/davis"),
        help="Path to the Davis raw-data directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/davis_structural_validation.json"),
        help="Destination for the JSON validation report.",
    )
    args = parser.parse_args(argv)

    try:
        report = validate_davis_dataset(args.data_dir)
        output_path = write_validation_report(report, args.output)
    except DavisValidationError as error:
        print(f"Davis validation failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    print(f"Validation report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())