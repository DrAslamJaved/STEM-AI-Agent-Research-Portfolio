import json
from pathlib import Path

import pandas as pd
import pytest

from src.features.representations import (
    FEATURE_COLUMNS,
    FeatureRepresentationError,
    build_drug_feature_table,
    build_pair_feature_table,
    build_target_feature_table,
    smiles_feature_dict,
    summarize_pair_features,
    target_feature_dict,
    write_feature_summary,
    write_pair_features,
)


def test_smiles_features_handle_aromatic_atoms_and_halogens() -> None:
    features = smiles_feature_dict("Clc1[nH]cc(Br)cc1C#N")

    assert features["drug_atom_count"] == 10.0
    assert features["drug_carbon_fraction"] == pytest.approx(0.6)
    assert features["drug_nitrogen_fraction"] == pytest.approx(0.2)
    assert features["drug_halogen_fraction"] == pytest.approx(0.2)
    assert features["drug_aromatic_atom_fraction"] == pytest.approx(0.6)
    assert features["drug_ring_marker_count"] == 2.0
    assert features["drug_branch_count"] == 1.0
    assert features["drug_triple_bond_count"] == 1.0


def test_target_features_use_y_reference_and_retain_x() -> None:
    features = target_feature_dict("ACXY")

    assert features["target_sequence_length"] == 4.0
    assert features["target_aa_fraction_A"] == pytest.approx(0.25)
    assert features["target_aa_fraction_C"] == pytest.approx(0.25)
    assert features["target_unknown_residue_fraction"] == pytest.approx(0.25)
    assert "target_aa_fraction_Y" not in features


def test_target_features_reject_unsupported_residues() -> None:
    with pytest.raises(FeatureRepresentationError, match="non-canonical"):
        target_feature_dict("ACZ")


def test_pair_feature_join_preserves_rows_and_excludes_matrix_indices() -> None:
    interactions = pd.DataFrame(
        {
            "observed_pair_index": [0, 1],
            "drug_id": ["drug_a", "drug_b"],
            "target_id": ["target_a", "target_a"],
            "drug_matrix_index": [0, 1],
            "target_matrix_index": [0, 0],
            "interaction_kd_le_1000_nM": [1, 0],
        }
    )
    drugs = build_drug_feature_table({"drug_a": "CCO", "drug_b": "C#N"})
    targets = build_target_feature_table({"target_a": "ACDY"})

    paired = build_pair_feature_table(interactions, drugs, targets)

    assert paired["observed_pair_index"].tolist() == [0, 1]
    assert len(paired) == len(interactions)
    assert "drug_matrix_index" not in FEATURE_COLUMNS
    assert "target_matrix_index" not in FEATURE_COLUMNS
    assert not paired.loc[:, FEATURE_COLUMNS].isna().any().any()


def test_pair_feature_join_rejects_missing_entity_features() -> None:
    interactions = pd.DataFrame(
        {
            "observed_pair_index": [0],
            "drug_id": ["missing_drug"],
            "target_id": ["target_a"],
        }
    )
    drugs = build_drug_feature_table({"drug_a": "CCO"})
    targets = build_target_feature_table({"target_a": "ACDY"})

    with pytest.raises(FeatureRepresentationError, match="lacks a drug or target"):
        build_pair_feature_table(interactions, drugs, targets)


def test_feature_writers_create_reproducible_outputs(tmp_path: Path) -> None:
    interactions = pd.DataFrame(
        {
            "observed_pair_index": [0, 1],
            "drug_id": ["drug_a", "drug_b"],
            "target_id": ["target_a", "target_a"],
        }
    )
    paired = build_pair_feature_table(
        interactions,
        build_drug_feature_table({"drug_a": "CCO", "drug_b": "C#N"}),
        build_target_feature_table({"target_a": "ACXY"}),
    )
    summary = summarize_pair_features(paired)

    table_path = write_pair_features(paired, tmp_path / "features.csv")
    summary_path = write_feature_summary(summary, tmp_path / "summary.json")

    reloaded = pd.read_csv(table_path)
    report = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(reloaded) == 2
    assert report["feature_column_count"] == len(FEATURE_COLUMNS)
    assert report["missing_feature_value_count"] == 0
    assert report["targets_with_unknown_residues"] == 1
    assert report["total_unknown_residue_count"] == 1

def test_pair_feature_join_normalizes_numeric_csv_identifiers() -> None:
    interactions = pd.DataFrame(
        {
            "observed_pair_index": [0, 1],
            "drug_id": [101, 102],
            "target_id": [1, 1],
        }
    )
    drugs = build_drug_feature_table({"101": "CCO", "102": "C#N"})
    targets = build_target_feature_table({"1": "ACDY"})

    paired = build_pair_feature_table(interactions, drugs, targets)

    assert paired["drug_id"].tolist() == ["101", "102"]
    assert paired["target_id"].tolist() == ["1", "1"]
    assert not paired.loc[:, FEATURE_COLUMNS].isna().any().any()