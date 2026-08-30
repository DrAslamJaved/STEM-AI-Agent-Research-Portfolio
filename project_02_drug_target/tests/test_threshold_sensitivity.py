"""Tests for frozen-fold Davis affinity-threshold sensitivity analysis."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

import src.models.threshold_sensitivity as sensitivity
from src.models.threshold_sensitivity import (
    SENSITIVITY_OOF_COLUMNS,
    FixedCandidate,
    ThresholdSensitivityError,
    build_frozen_fold_dataset,
    extract_frozen_fold_assignments,
    run_threshold_sensitivity,
    write_threshold_sensitivity_oof_predictions,
    write_threshold_sensitivity_summary,
)


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return two fixed drug folds with both classes for both thresholds."""
    fold_by_drug = {
        "drug_0": 0,
        "drug_1": 0,
        "drug_2": 1,
        "drug_3": 1,
    }
    feature_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    observed_pair_index = 0

    for drug_id, fold_index in fold_by_drug.items():
        for target_index in range(4):
            feature_rows.append(
                {
                    "observed_pair_index": observed_pair_index,
                    "drug_id": drug_id,
                    "target_id": f"target_{target_index}",
                    "interaction_kd_le_1000_nM": int(target_index % 2 == 0),
                    "interaction_kd_le_100_nM": int(
                        target_index in (0, 3)
                    ),
                    "feature_a": float(target_index),
                    "feature_b": float(observed_pair_index % 3),
                }
            )
            oof_rows.append(
                {
                    "model_id": "dummy_prior",
                    "fold_index": fold_index,
                    "observed_pair_index": observed_pair_index,
                    "drug_id": drug_id,
                    "target_id": f"target_{target_index}",
                    "y_true": 99,
                    "positive_probability": 0.99,
                }
            )
            observed_pair_index += 1

    summary: dict[str, object] = {
        "outer_policy": "cold_drug",
        "cv_scope": "frozen_outer_training_partition_only",
        "outer_test_partition_used": False,
        "label_column": "interaction_kd_le_1000_nM",
        "n_splits": 2,
        "random_state": 7,
        "input_pair_count": len(feature_rows),
        "input_drug_count": len(fold_by_drug),
    }
    return (
        pd.DataFrame(feature_rows),
        pd.DataFrame(oof_rows),
        summary,
    )


def patch_minimal_feature_and_model_contract(monkeypatch) -> None:
    """Keep tests fast while exercising frozen-fold orchestration fully."""
    monkeypatch.setattr(
        sensitivity,
        "FEATURE_COLUMNS",
        ("feature_a", "feature_b"),
    )
    monkeypatch.setattr(
        sensitivity,
        "_candidate_models",
        lambda: (
            FixedCandidate(
                model_id="test_prior",
                model_name="DummyClassifier",
                model_parameters={"strategy": "prior"},
                builder=lambda state: DummyClassifier(
                    strategy="prior",
                    random_state=state,
                ),
            ),
        ),
    )


def test_frozen_dataset_uses_only_primary_oof_membership(monkeypatch) -> None:
    feature_table, oof_table, summary = synthetic_inputs()
    patch_minimal_feature_and_model_contract(monkeypatch)

    dataset = build_frozen_fold_dataset(
        feature_table,
        oof_table,
        summary,
        n_splits=2,
        random_state=7,
    )

    assert len(dataset.features) == 16
    assert dataset.metadata["drug_id"].nunique() == 4
    assert tuple(dataset.feature_columns) == ("feature_a", "feature_b")
    assert set(dataset.fold_indices) == {0, 1}
    assert dataset.labels["interaction_kd_le_1000_nM"].sum() == 8
    assert dataset.labels["interaction_kd_le_100_nM"].sum() == 8
    assert dataset.metadata.groupby("drug_id")["observed_pair_index"].count().eq(4).all()


def test_run_uses_same_folds_for_both_pre_specified_labels(monkeypatch) -> None:
    feature_table, oof_table, summary = synthetic_inputs()
    patch_minimal_feature_and_model_contract(monkeypatch)

    run = run_threshold_sensitivity(
        feature_table,
        oof_table,
        summary,
        n_splits=2,
        random_state=7,
    )

    assert run.report["outer_test_partition_used"] is False
    assert run.report["outer_test_outcomes_selected"] is False
    assert run.report["model_selection"]["selection_reopened"] is False
    assert len(run.report["variants"]) == 2
    assert len(run.oof_predictions) == 32
    assert tuple(run.oof_predictions.columns) == SENSITIVITY_OOF_COLUMNS
    fold_counts = run.oof_predictions.groupby(
        ["threshold_variant", "observed_pair_index"]
    )["fold_index"].nunique()
    assert fold_counts.eq(1).all()
    assert set(run.oof_predictions["label_column"]) == {
        "interaction_kd_le_1000_nM",
        "interaction_kd_le_100_nM",
    }


def test_reference_extraction_does_not_require_prior_labels_or_scores() -> None:
    _, oof_table, _ = synthetic_inputs()
    structural_only = oof_table.loc[
        :,
        [
            "model_id",
            "fold_index",
            "observed_pair_index",
            "drug_id",
            "target_id",
        ],
    ]

    assignments = extract_frozen_fold_assignments(structural_only)

    assert len(assignments) == 16
    assert tuple(assignments.columns) == (
        "fold_index",
        "observed_pair_index",
        "drug_id",
        "target_id",
    )


def test_invalid_frozen_contract_and_duplicate_reference_rows_fail(monkeypatch) -> None:
    feature_table, oof_table, summary = synthetic_inputs()
    patch_minimal_feature_and_model_contract(monkeypatch)

    invalid_summary = {**summary, "outer_test_partition_used": True}
    with pytest.raises(ThresholdSensitivityError, match="outer_test_partition_used"):
        build_frozen_fold_dataset(
            feature_table,
            oof_table,
            invalid_summary,
            n_splits=2,
            random_state=7,
        )

    duplicate_oof = pd.concat([oof_table, oof_table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ThresholdSensitivityError, match="duplicate"):
        extract_frozen_fold_assignments(duplicate_oof)


def test_writers_preserve_json_and_oof_contracts(monkeypatch, tmp_path) -> None:
    feature_table, oof_table, summary = synthetic_inputs()
    patch_minimal_feature_and_model_contract(monkeypatch)
    run = run_threshold_sensitivity(
        feature_table,
        oof_table,
        summary,
        n_splits=2,
        random_state=7,
    )

    summary_path = write_threshold_sensitivity_summary(
        run,
        tmp_path / "threshold_sensitivity.json",
    )
    oof_path = write_threshold_sensitivity_oof_predictions(
        run.oof_predictions,
        tmp_path / "threshold_sensitivity_oof.csv",
    )

    saved_report = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_predictions = pd.read_csv(oof_path)
    assert saved_report["n_splits"] == 2
    assert tuple(saved_predictions.columns) == SENSITIVITY_OOF_COLUMNS
    assert len(saved_predictions) == len(run.oof_predictions)
