"""Transparent fixed features for a reproducible DTI baseline experiment."""

from __future__ import annotations

from math import isfinite
from typing import Any, Iterable, Mapping

import numpy as np


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DRUG_FEATURE_NAMES = (
    "smiles_length", "smiles_upper_c", "smiles_n", "smiles_o", "smiles_s",
    "smiles_halogen", "smiles_aromatic", "smiles_ring_digit", "smiles_branch",
    "smiles_double_bond", "smiles_triple_bond",
)
PROTEIN_FEATURE_NAMES = ("protein_length",) + tuple(f"protein_fraction_{aa}" for aa in AMINO_ACIDS)
FEATURE_NAMES = DRUG_FEATURE_NAMES + PROTEIN_FEATURE_NAMES


def _ratio(text: str, character: str) -> float:
    return text.count(character) / len(text) if text else 0.0


def dti_feature_vector(record: Mapping[str, Any]) -> list[float]:
    """Create fixed, interpretable SMILES and sequence-composition features."""
    smiles = str(record.get("smiles", ""))
    sequence = str(record.get("protein_sequence", "")).upper()
    if not smiles or not sequence:
        raise ValueError("DTI record requires non-empty smiles and protein_sequence.")
    drug_features = [
        float(len(smiles)), _ratio(smiles, "C"), _ratio(smiles, "N"), _ratio(smiles, "O"), _ratio(smiles, "S"),
        (smiles.count("F") + smiles.count("Cl") + smiles.count("Br")) / len(smiles),
        sum(smiles.count(token) for token in ("c", "n", "o", "s")) / len(smiles),
        sum(character.isdigit() for character in smiles) / len(smiles),
        smiles.count("(") / len(smiles), smiles.count("=") / len(smiles), smiles.count("#") / len(smiles),
    ]
    protein_features = [float(len(sequence))] + [_ratio(sequence, aa) for aa in AMINO_ACIDS]
    values = drug_features + protein_features
    if not all(isfinite(value) for value in values):
        raise ValueError("Feature construction produced a non-finite value.")
    return values


def feature_matrix(records: Iterable[Mapping[str, Any]]) -> np.ndarray:
    rows = [dti_feature_vector(record) for record in records]
    if not rows:
        raise ValueError("At least one DTI record is required.")
    return np.asarray(rows, dtype=float)
