"""Deterministic loader and provenance checks for the Davis kinase dataset.

The loader expects the canonical DeepDTA-style directory containing
``ligands_can.txt``, ``proteins.txt``, and ``Y``.  The first two are JSON
objects and ``Y`` is a rectangular JSON affinity matrix in nanomolar units.
No network download occurs in this module: the dataset must be acquired and
its file hashes approved by the researcher before use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite, log10
from pathlib import Path
import pickle
from typing import Any, Mapping


CANONICAL_DAVIS_FILES = ("ligands_can.txt", "proteins.txt", "Y")
DEFAULT_BINDER_PKD_CUTOFF = 7.0  # Kd <= 100 nM


@dataclass(frozen=True)
class DavisSourceReport:
    root: str
    compound_count: int
    target_count: int
    affinity_rows: int
    affinity_columns: int
    record_count: int
    is_valid: bool
    issues: tuple[str, ...]
    sha256: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DavisDataset:
    """Canonical Davis data with source-order identifiers preserved."""

    compound_ids: tuple[str, ...]
    compounds: dict[str, str]
    target_ids: tuple[str, ...]
    targets: dict[str, str]
    affinity_nm: tuple[tuple[float, ...], ...]
    source_report: DavisSourceReport


def sha256_file(path: str | Path) -> str:
    """Return a stable SHA-256 digest without loading the whole file at once."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_hashes(actual: Mapping[str, str], expected: Mapping[str, str]) -> None:
    """Require an exact approved SHA-256 value for every canonical source file."""
    missing = [name for name in CANONICAL_DAVIS_FILES if name not in expected]
    placeholders = [name for name in CANONICAL_DAVIS_FILES
                    if str(expected.get(name, "")).startswith("REPLACE_")]
    mismatches = [name for name in CANONICAL_DAVIS_FILES
                  if name in expected and str(expected[name]).lower() != actual[name].lower()]
    if missing or placeholders or mismatches:
        parts: list[str] = []
        if missing:
            parts.append("missing approved hash for " + ", ".join(missing))
        if placeholders:
            parts.append("placeholder hash for " + ", ".join(placeholders))
        if mismatches:
            parts.append("hash mismatch for " + ", ".join(mismatches))
        raise ValueError("Davis source is not approved: " + "; ".join(parts))


def _read_json_object(path: Path, *, label: str) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse {label} JSON: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty JSON object.")
    normalised = {str(key): str(item) for key, item in value.items()}
    if any(not key.strip() or not item.strip() for key, item in normalised.items()):
        raise ValueError(f"{label} contains an empty identifier or sequence.")
    return normalised


def _read_affinity_matrix(path: Path) -> tuple[tuple[float, ...], ...]:
    """Load the canonical DeepDTA binary-pickle affinity matrix.

    Python pickles are unsafe when their provenance is unknown. Callers must
    only use the pinned and hash-reviewed source specified in the v0.2 manifest.
    """
    try:
        with path.open("rb") as handle:
            # DeepDTA's canonical loader specifies latin1 for legacy pickles.
            value = pickle.load(handle, encoding="latin1")
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as exc:
        raise ValueError(f"Cannot parse binary-pickle Y affinity matrix: {exc}") from exc
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(row, (list, tuple)) and row for row in value):
        raise ValueError("Y must be a non-empty rectangular affinity matrix.")
    matrix = [list(row) for row in value]
    width = len(matrix[0])
    rows: list[tuple[float, ...]] = []
    for row in matrix:
        if len(row) != width:
            raise ValueError("Y must be rectangular.")
        converted = tuple(float(item) for item in row)
        if any(not isfinite(item) or item <= 0.0 for item in converted):
            raise ValueError("Y must contain finite, strictly positive Kd values in nM.")
        rows.append(converted)
    return tuple(rows)


def load_davis_dataset(root: str | Path, *, expected_sha256: Mapping[str, str] | None = None) -> DavisDataset:
    """Load and validate a local canonical Davis data directory.

    The matrix convention is compounds by rows and targets by columns, matching
    the insertion order of ``ligands_can.txt`` and ``proteins.txt``.
    """
    directory = Path(root)
    missing = [name for name in CANONICAL_DAVIS_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Davis source is missing required file(s): {', '.join(missing)}")
    hashes = {name: sha256_file(directory / name) for name in CANONICAL_DAVIS_FILES}
    if expected_sha256 is not None:
        verify_source_hashes(hashes, expected_sha256)
    compounds = _read_json_object(directory / "ligands_can.txt", label="ligands_can.txt")
    targets = _read_json_object(directory / "proteins.txt", label="proteins.txt")
    affinities = _read_affinity_matrix(directory / "Y")
    issues: list[str] = []
    if len(affinities) != len(compounds):
        issues.append("affinity row count does not equal compound count")
    if any(len(row) != len(targets) for row in affinities):
        issues.append("affinity column count does not equal target count")
    report = DavisSourceReport(
        root=str(directory),
        compound_count=len(compounds),
        target_count=len(targets),
        affinity_rows=len(affinities),
        affinity_columns=len(affinities[0]),
        record_count=len(compounds) * len(targets),
        is_valid=not issues,
        issues=tuple(issues),
        sha256=hashes,
    )
    if not report.is_valid:
        raise ValueError("Invalid Davis dimensions: " + "; ".join(report.issues))
    return DavisDataset(
        compound_ids=tuple(compounds), compounds=compounds,
        target_ids=tuple(targets), targets=targets,
        affinity_nm=affinities, source_report=report,
    )


def affinity_nm_to_pkd(affinity_nm: float) -> float:
    """Convert Kd in nM to pKd: ``-log10(Kd[M])``."""
    value = float(affinity_nm)
    if not isfinite(value) or value <= 0.0:
        raise ValueError("Affinity must be finite and strictly positive in nM.")
    return 9.0 - log10(value)


def davis_records(dataset: DavisDataset, *, binder_pkd_cutoff: float = DEFAULT_BINDER_PKD_CUTOFF) -> list[dict[str, Any]]:
    """Return deterministic long-form DTI records retaining raw and derived values."""
    cutoff = float(binder_pkd_cutoff)
    if not isfinite(cutoff):
        raise ValueError("binder_pkd_cutoff must be finite.")
    records: list[dict[str, Any]] = []
    for drug_index, drug_id in enumerate(dataset.compound_ids):
        for target_index, target_id in enumerate(dataset.target_ids):
            kd_nm = dataset.affinity_nm[drug_index][target_index]
            pkd = affinity_nm_to_pkd(kd_nm)
            records.append({
                "dataset": "Davis",
                "drug_id": drug_id,
                "target_id": target_id,
                "smiles": dataset.compounds[drug_id],
                "protein_sequence": dataset.targets[target_id],
                "affinity_nm": kd_nm,
                "pkd": pkd,
                "label": int(pkd >= cutoff),
                "binder_pkd_cutoff": cutoff,
            })
    return records
