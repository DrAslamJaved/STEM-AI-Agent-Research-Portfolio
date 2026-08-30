"""Tests for the label-free transparent feature-collision audit."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.features.collision_audit import (
    ENTITY_PAIR_COLUMNS,
    CollisionAuditError,
    audit_entity_representations,
    load_representation_mapping,
    run_collision_audit,
    write_collision_audit_summary,
    write_collision_pairs,
)
from src.features.representations import (
    DRUG_FEATURE_COLUMNS,
    build_drug_feature_table,
)


def synthetic_representations() -> tuple[dict[str, str], dict[str, str]]:
    """Return raw duplicates and order-insensitive descriptor collisions."""
    ligands = {
        "drug_a": "CCO",
        "drug_b": "CCO",
        "drug_c": "COC",
        "drug_d": "CN",
    }
    proteins = {
        "target_a": "ACDE",
        "target_b": "ACDE",
        "target_c": "ACED",
        "target_d": "AAAA",
    }
    return ligands, proteins


def test_audit_distinguishes_raw_duplicates_from_feature_collisions() -> None:
    ligands, proteins = synthetic_representations()

    run = run_collision_audit(ligands, proteins, top_n=3)

    drug_summary = run.report["drug_audit"]
    target_summary = run.report["target_audit"]
    assert drug_summary["raw_duplicate_group_count"] == 1
    assert drug_summary["raw_duplicate_pair_count"] == 1
    assert drug_summary["exact_feature_collision_pair_count"] == 3
    assert drug_summary["distinct_raw_feature_collision_pair_count"] == 2
    assert target_summary["raw_duplicate_group_count"] == 1
    assert target_summary["distinct_raw_feature_collision_pair_count"] == 2


def test_nearest_distinct_raw_pair_can_be_an_exact_descriptor_collision() -> None:
    ligands, proteins = synthetic_representations()

    run = run_collision_audit(ligands, proteins, top_n=3)
    pair = run.pair_table.loc[
        (run.pair_table["entity_type"] == "drug")
        & (run.pair_table["entity_id_a"] == "drug_a")
        & (run.pair_table["entity_id_b"] == "drug_c")
    ].iloc[0]

    assert bool(pair["raw_representation_equal"]) is False
    assert bool(pair["exact_feature_vector_equal"]) is True
    assert bool(pair["included_as_nearest_pair"]) is True
    assert pair["range_normalized_mean_absolute_difference"] == 0.0
    assert pair["range_normalized_max_absolute_difference"] == 0.0


def test_report_excludes_labels_predictions_and_raw_representation_strings() -> None:
    ligands, proteins = synthetic_representations()

    report = run_collision_audit(ligands, proteins, top_n=2).report
    serialized = json.dumps(report, sort_keys=True)

    assert report["audit_scope"] == "unsupervised_raw_and_feature_representation_only"
    assert report["outcome_values_used"] is False
    assert report["model_predictions_used"] is False
    assert report["outer_holdout_raw_representations_included"] is True
    assert "CCO" not in serialized
    assert "ACDE" not in serialized


def test_writers_produce_the_frozen_json_and_csv_contracts(tmp_path) -> None:
    ligands, proteins = synthetic_representations()
    run = run_collision_audit(ligands, proteins, top_n=2)

    summary_path = write_collision_audit_summary(
        run,
        tmp_path / "collision_audit.json",
    )
    pairs_path = write_collision_pairs(
        run.pair_table,
        tmp_path / "collision_pairs.csv",
    )

    saved_report = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_pairs = pd.read_csv(pairs_path)
    assert saved_report["top_n"] == 2
    assert tuple(saved_pairs.columns) == ENTITY_PAIR_COLUMNS
    assert len(saved_pairs) == len(run.pair_table)


def test_mapping_loader_and_audit_input_validation_fail_loudly(tmp_path) -> None:
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("[]", encoding="utf-8")

    with pytest.raises(CollisionAuditError, match="non-empty JSON object"):
        load_representation_mapping(malformed_path, "synthetic mapping")

    ligands, _ = synthetic_representations()
    feature_table = build_drug_feature_table(ligands)
    inconsistent_table = feature_table.iloc[:-1].copy()

    with pytest.raises(CollisionAuditError, match="identifiers do not match"):
        audit_entity_representations(
            entity_type="drug",
            representations=ligands,
            feature_table=inconsistent_table,
            id_column="drug_id",
            feature_columns=DRUG_FEATURE_COLUMNS,
        )

    with pytest.raises(CollisionAuditError, match="at least one"):
        run_collision_audit(ligands, {"target": "ACDE"}, top_n=0)
