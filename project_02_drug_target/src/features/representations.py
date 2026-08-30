"""Build transparent, dependency-free Davis drug-target feature tables."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class FeatureRepresentationError(ValueError):
    """Raised when transparent features cannot be constructed safely."""


CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
TARGET_REFERENCE_RESIDUE = "Y"
UNKNOWN_AMINO_ACID = "X"

TARGET_AMINO_ACID_FEATURES = tuple(
    residue
    for residue in CANONICAL_AMINO_ACIDS
    if residue != TARGET_REFERENCE_RESIDUE
)
SMILES_ELEMENT_FEATURES = ("C", "N", "O", "S", "P")
SMILES_HALOGENS = ("F", "Cl", "Br", "I")

DRUG_FEATURE_COLUMNS = (
    "drug_smiles_length",
    "drug_atom_count",
    "drug_carbon_fraction",
    "drug_nitrogen_fraction",
    "drug_oxygen_fraction",
    "drug_sulfur_fraction",
    "drug_phosphorus_fraction",
    "drug_halogen_fraction",
    "drug_other_atom_fraction",
    "drug_aromatic_atom_fraction",
    "drug_ring_marker_count",
    "drug_branch_count",
    "drug_double_bond_count",
    "drug_triple_bond_count",
    "drug_charge_marker_count",
)

TARGET_FEATURE_COLUMNS = (
    "target_sequence_length",
    *tuple(
        f"target_aa_fraction_{residue}"
        for residue in TARGET_AMINO_ACID_FEATURES
    ),
    "target_unknown_residue_fraction",
)

FEATURE_COLUMNS = DRUG_FEATURE_COLUMNS + TARGET_FEATURE_COLUMNS


@dataclass(frozen=True)
class FeatureSummary:
    """Compact feature-quality summary safe to commit to the repository."""

    row_count: int
    unique_drug_count: int
    unique_target_count: int
    drug_feature_count: int
    target_feature_count: int
    feature_column_count: int
    missing_feature_value_count: int
    nonfinite_feature_value_count: int
    drug_smiles_length_min: float
    drug_smiles_length_max: float
    target_sequence_length_min: float
    target_sequence_length_max: float
    target_reference_residue: str
    targets_with_unknown_residues: int
    total_unknown_residue_count: int
    maximum_unknown_residue_fraction: float
    feature_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_mapping(path: Path, label: str) -> dict[str, str]:
    """Load a non-empty string-to-string JSON representation mapping."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureRepresentationError(f"Could not load {label}: {path}") from error

    if not isinstance(payload, dict) or not payload:
        raise FeatureRepresentationError(f"{label} must be a non-empty JSON object.")

    if any(
        not isinstance(identifier, str)
        or not identifier.strip()
        or not isinstance(representation, str)
        or not representation.strip()
        for identifier, representation in payload.items()
    ):
        raise FeatureRepresentationError(
            f"{label} contains an empty or invalid identifier/representation."
        )

    return payload

def _normalise_identifier_columns(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    table_label: str,
) -> pd.DataFrame:
    """Return a copy with non-empty identifier columns normalized to strings."""
    missing_columns = set(columns).difference(table.columns)
    if missing_columns:
        raise FeatureRepresentationError(
            f"{table_label} is missing columns: {sorted(missing_columns)}"
        )

    normalized = table.copy()
    for column in columns:
        if normalized[column].isna().any():
            raise FeatureRepresentationError(
                f"{table_label} contains missing identifiers in {column}."
            )

        normalized[column] = normalized[column].astype(str).str.strip()
        if normalized[column].eq("").any():
            raise FeatureRepresentationError(
                f"{table_label} contains empty identifiers in {column}."
            )

    return normalized

def _smiles_atoms(smiles: str) -> tuple[list[str], int]:
    """Extract lightweight atom tokens and aromatic-token count from SMILES."""
    tokens: list[str] = []
    aromatic_count = 0
    index = 0

    while index < len(smiles):
        character = smiles[index]

        if character == "[":
            closing_index = smiles.find("]", index + 1)
            if closing_index == -1:
                raise FeatureRepresentationError(
                    f"SMILES contains an unmatched bracket: {smiles}"
                )

            bracket_content = smiles[index + 1 : closing_index]
            match = re.match(r"([A-Z][a-z]?|[cnops])", bracket_content)
            if match is None:
                raise FeatureRepresentationError(
                    f"SMILES bracket does not begin with an atom token: {smiles}"
                )

            token = match.group(1)
            index = closing_index + 1

        elif smiles.startswith(("Cl", "Br", "Si", "Se"), index):
            token = smiles[index : index + 2]
            index += 2

        elif character in "BCNOFPSI":
            token = character
            index += 1

        elif character in "cnops":
            token = character
            index += 1

        else:
            index += 1
            continue

        if token in "cnops":
            aromatic_count += 1
            token = token.upper()

        tokens.append(token)

    if not tokens:
        raise FeatureRepresentationError(f"SMILES contains no atom tokens: {smiles}")

    return tokens, aromatic_count


def smiles_feature_dict(smiles: str) -> dict[str, float]:
    """Create transparent fixed-size descriptors from one canonical SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise FeatureRepresentationError("SMILES must be a non-empty string.")

    tokens, aromatic_count = _smiles_atoms(smiles)
    atom_count = len(tokens)

    element_counts = {
        element: tokens.count(element)
        for element in SMILES_ELEMENT_FEATURES
    }
    halogen_count = sum(tokens.count(element) for element in SMILES_HALOGENS)
    known_count = sum(element_counts.values()) + halogen_count

    return {
        "drug_smiles_length": float(len(smiles)),
        "drug_atom_count": float(atom_count),
        "drug_carbon_fraction": float(element_counts["C"] / atom_count),
        "drug_nitrogen_fraction": float(element_counts["N"] / atom_count),
        "drug_oxygen_fraction": float(element_counts["O"] / atom_count),
        "drug_sulfur_fraction": float(element_counts["S"] / atom_count),
        "drug_phosphorus_fraction": float(element_counts["P"] / atom_count),
        "drug_halogen_fraction": float(halogen_count / atom_count),
        "drug_other_atom_fraction": float(
            (atom_count - known_count) / atom_count
        ),
        "drug_aromatic_atom_fraction": float(aromatic_count / atom_count),
        "drug_ring_marker_count": float(sum(char.isdigit() for char in smiles)),
        "drug_branch_count": float(smiles.count("(")),
        "drug_double_bond_count": float(smiles.count("=")),
        "drug_triple_bond_count": float(smiles.count("#")),
        "drug_charge_marker_count": float(
            smiles.count("+") + smiles.count("-")
        ),
    }


def target_feature_dict(sequence: str) -> dict[str, float]:
    """Create length and composition features from one protein sequence."""
    if not isinstance(sequence, str) or not sequence.strip():
        raise FeatureRepresentationError(
            "Protein sequence must be a non-empty string."
        )

    normalized_sequence = sequence.upper()
    allowed_residues = set(CANONICAL_AMINO_ACIDS + UNKNOWN_AMINO_ACID)
    invalid_residues = sorted(
        set(normalized_sequence).difference(allowed_residues)
    )

    if invalid_residues:
        raise FeatureRepresentationError(
            "Protein sequence contains non-canonical residues: "
            f"{invalid_residues}"
        )

    sequence_length = len(normalized_sequence)
    features = {"target_sequence_length": float(sequence_length)}
    features.update(
        {
            f"target_aa_fraction_{residue}": float(
                normalized_sequence.count(residue) / sequence_length
            )
            for residue in TARGET_AMINO_ACID_FEATURES
        }
    )
    features["target_unknown_residue_fraction"] = float(
        normalized_sequence.count(UNKNOWN_AMINO_ACID) / sequence_length
    )

    return features


def build_drug_feature_table(ligands: dict[str, str]) -> pd.DataFrame:
    """Return one deterministic descriptor row for each Davis drug."""
    if not ligands:
        raise FeatureRepresentationError("Ligand mapping is empty.")

    rows = [
        {"drug_id": drug_id, **smiles_feature_dict(smiles)}
        for drug_id, smiles in ligands.items()
    ]
    return pd.DataFrame(rows, columns=["drug_id", *DRUG_FEATURE_COLUMNS])


def build_target_feature_table(proteins: dict[str, str]) -> pd.DataFrame:
    """Return one deterministic descriptor row for each Davis target."""
    if not proteins:
        raise FeatureRepresentationError("Protein mapping is empty.")

    rows = [
        {"target_id": target_id, **target_feature_dict(sequence)}
        for target_id, sequence in proteins.items()
    ]
    return pd.DataFrame(rows, columns=["target_id", *TARGET_FEATURE_COLUMNS])


def build_pair_feature_table(
    interactions: pd.DataFrame,
    drug_features: pd.DataFrame,
    target_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join deterministic entity features without using outcome values."""
    required_columns = {"observed_pair_index", "drug_id", "target_id"}
    missing_columns = required_columns.difference(interactions.columns)

    interactions = _normalise_identifier_columns(
        interactions,
        ("drug_id", "target_id"),
        "Interaction table",
    )
    drug_features = _normalise_identifier_columns(
        drug_features,
        ("drug_id",),
        "Drug feature table",
    )
    target_features = _normalise_identifier_columns(
        target_features,
        ("target_id",),
        "Target feature table",
    )

    if missing_columns:
        raise FeatureRepresentationError(
            f"Interactions are missing columns: {sorted(missing_columns)}"
        )
    if interactions.empty:
        raise FeatureRepresentationError("Interaction table is empty.")
    if interactions["observed_pair_index"].duplicated().any():
        raise FeatureRepresentationError(
            "observed_pair_index values must be unique."
        )
    if drug_features["drug_id"].duplicated().any():
        raise FeatureRepresentationError(
            "Drug feature table contains duplicate IDs."
        )
    if target_features["target_id"].duplicated().any():
        raise FeatureRepresentationError(
            "Target feature table contains duplicate IDs."
        )

    paired = interactions.merge(
        drug_features,
        on="drug_id",
        how="left",
        validate="many_to_one",
        sort=False,
    ).merge(
        target_features,
        on="target_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if len(paired) != len(interactions):
        raise FeatureRepresentationError(
            "Feature join changed the interaction-row count."
        )
    if paired.loc[:, FEATURE_COLUMNS].isna().any().any():
        raise FeatureRepresentationError(
            "At least one interaction lacks a drug or target feature row."
        )

    values = paired.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise FeatureRepresentationError(
            "Feature table contains non-finite values."
        )

    return paired


def summarize_pair_features(pair_features: pd.DataFrame) -> FeatureSummary:
    """Return feature-quality evidence without storing raw representations."""
    missing_columns = set(FEATURE_COLUMNS).difference(pair_features.columns)
    if missing_columns:
        raise FeatureRepresentationError(
            f"Pair feature table is missing columns: {sorted(missing_columns)}"
        )
    if pair_features.empty:
        raise FeatureRepresentationError("Pair feature table is empty.")

    values = pair_features.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    unique_targets = pair_features.drop_duplicates("target_id")
    unknown_fractions = unique_targets["target_unknown_residue_fraction"]

    return FeatureSummary(
        row_count=int(len(pair_features)),
        unique_drug_count=int(pair_features["drug_id"].nunique()),
        unique_target_count=int(pair_features["target_id"].nunique()),
        drug_feature_count=len(DRUG_FEATURE_COLUMNS),
        target_feature_count=len(TARGET_FEATURE_COLUMNS),
        feature_column_count=len(FEATURE_COLUMNS),
        missing_feature_value_count=int(
            pair_features.loc[:, FEATURE_COLUMNS].isna().sum().sum()
        ),
        nonfinite_feature_value_count=int((~np.isfinite(values)).sum()),
        drug_smiles_length_min=float(pair_features["drug_smiles_length"].min()),
        drug_smiles_length_max=float(pair_features["drug_smiles_length"].max()),
        target_sequence_length_min=float(
            pair_features["target_sequence_length"].min()
        ),
        target_sequence_length_max=float(
            pair_features["target_sequence_length"].max()
        ),
        target_reference_residue=TARGET_REFERENCE_RESIDUE,
        targets_with_unknown_residues=int((unknown_fractions > 0).sum()),
        total_unknown_residue_count=int(
            round(
                (
                    unknown_fractions
                    * unique_targets["target_sequence_length"]
                ).sum()
            )
        ),
        maximum_unknown_residue_fraction=float(unknown_fractions.max()),
        feature_columns=FEATURE_COLUMNS,
    )


def write_pair_features(table: pd.DataFrame, output_path: str | Path) -> Path:
    """Write local ignored pair features for model development."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return destination


def write_feature_summary(
    summary: FeatureSummary,
    output_path: str | Path,
) -> Path:
    """Write compact version-controlled feature-quality evidence."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build transparent, dependency-free Davis feature tables."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/davis"),
        help="Directory containing ligands_can.txt and proteins.txt.",
    )
    parser.add_argument(
        "--interaction-table",
        type=Path,
        default=Path("data/interim/davis_interactions_labeled.csv"),
        help="Local labelled Davis interaction table.",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("data/processed/davis_pair_features.csv"),
        help="Local pair-feature CSV destination.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/davis_feature_summary.json"),
        help="Version-controlled feature-summary JSON destination.",
    )
    args = parser.parse_args(argv)

    try:
        interactions = pd.read_csv(
            args.interaction_table,
            dtype={"drug_id": str, "target_id": str},
        )
        ligands = _load_mapping(
            args.data_dir / "ligands_can.txt",
            "ligands_can.txt",
        )
        proteins = _load_mapping(
            args.data_dir / "proteins.txt",
            "proteins.txt",
        )
        drug_features = build_drug_feature_table(ligands)
        target_features = build_target_feature_table(proteins)
        pair_features = build_pair_feature_table(
            interactions,
            drug_features,
            target_features,
        )
        summary = summarize_pair_features(pair_features)
        table_path = write_pair_features(pair_features, args.table_output)
        summary_path = write_feature_summary(summary, args.summary_output)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(f"Feature construction failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print(f"Pair features written to: {table_path}")
    print(f"Feature summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())