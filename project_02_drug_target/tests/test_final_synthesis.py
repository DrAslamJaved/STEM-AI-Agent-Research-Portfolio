"""Tests for the final, leakage-aware Davis evidence synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.reporting.final_synthesis import (
    EXPECTED_MODEL_IDS,
    FinalSynthesisError,
    build_final_synthesis,
    main,
    read_json_report,
    write_final_synthesis_markdown,
    write_final_synthesis_summary,
)


def _metric_summary(mean: float) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": mean,
            "standard_deviation": 0.01,
            "minimum": mean - 0.01,
            "maximum": mean + 0.01,
        }
        for metric in (
            "average_precision",
            "roc_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
        )
    }


def _pooled_metrics(mean: float) -> dict[str, float | int]:
    return {
        "average_precision": mean,
        "roc_auc": mean,
        "accuracy": mean,
        "precision": mean,
        "recall": mean,
        "f1": mean,
        "positive_rate": 0.2,
        "sample_count": 16,
        "decision_threshold": 0.5,
        "true_negative": 8,
        "false_positive": 2,
        "false_negative": 3,
        "true_positive": 3,
    }


def _model_result(
    model_id: str, average_precision: float, pooled_key: str
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_name": model_id,
        "model_parameters": {"fixed": True},
        "oof_prediction_count": 16,
        "fold_metric_summary": _metric_summary(average_precision),
        pooled_key: _pooled_metrics(average_precision),
        "fold_results": [
            {"drug_overlap_count": 0},
            {"drug_overlap_count": 0},
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def synthetic_inputs(tmp_path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    primary_scores = {
        "dummy_prior": 0.20,
        "logistic_regression_balanced": 0.30,
        "random_forest_balanced": 0.40,
        "hist_gradient_boosting_balanced": 0.38,
    }
    inner_cv = {
        "outer_policy": "cold_drug",
        "cv_scope": "frozen_outer_training_partition_only",
        "outer_test_partition_used": False,
        "label_column": "interaction_kd_le_1000_nM",
        "n_splits": 2,
        "random_state": 7,
        "group_column": "drug_id",
        "input_pair_count": 16,
        "input_drug_count": 4,
        "input_feature_count": 2,
        "primary_comparison_metric": "average_precision",
        "model_results": [
            _model_result(model_id, primary_scores[model_id], "pooled_oof_metrics")
            for model_id in EXPECTED_MODEL_IDS
        ],
    }
    threshold_variants = []
    variant_inputs = (
        (
            "primary_kd_le_1000_nM",
            "interaction_kd_le_1000_nM",
            1000.0,
            6.0,
            0.20,
            primary_scores,
        ),
        (
            "sensitivity_kd_le_100_nM",
            "interaction_kd_le_100_nM",
            100.0,
            7.0,
            0.10,
            {
                "dummy_prior": 0.10,
                "logistic_regression_balanced": 0.22,
                "random_forest_balanced": 0.25,
                "hist_gradient_boosting_balanced": 0.29,
            },
        ),
    )
    for variant_id, label, kd, p_kd, rate, scores in variant_inputs:
        threshold_variants.append(
            {
                "variant_id": variant_id,
                "label_column": label,
                "kd_threshold_nM": kd,
                "pKd_threshold": p_kd,
                "positive_count": int(16 * rate),
                "negative_count": 16 - int(16 * rate),
                "positive_rate": rate,
                "model_results": [
                    _model_result(
                        model_id,
                        scores[model_id],
                        "pooled_oof_metrics_descriptive_only",
                    )
                    for model_id in EXPECTED_MODEL_IDS
                ],
            }
        )
    threshold = {
        "analysis_scope": "frozen_outer_training_partition_fixed_inner_cold_drug_folds",
        "outer_policy": "cold_drug",
        "outer_test_partition_used": False,
        "outer_test_outcomes_selected": False,
        "n_splits": 2,
        "random_state": 7,
        "input_pair_count": 16,
        "input_drug_count": 4,
        "model_selection": {
            "selection_reopened": False,
            "hyperparameter_tuning_performed": False,
            "selection_metric": "average_precision",
            "primary_selected_model_id": "random_forest_balanced",
        },
        "variants": threshold_variants,
    }
    split = {
        "split_policies": [
            {
                "policy": "cold_drug",
                "drug_overlap_count": 0,
                "target_overlap_count": 6,
                "reference_label_column": "interaction_kd_le_1000_nM",
                "splitter_name": "StratifiedGroupKFold",
                "fold_index": 1,
                "random_state": 7,
                "train_pair_count": 16,
                "test_pair_count": 8,
                "train_drug_count": 4,
                "test_drug_count": 2,
                "train_positive_rate": 0.2,
                "test_positive_rate": 0.25,
            }
        ]
    }
    collision = {
        "audit_scope": "unsupervised_raw_and_feature_representation_only",
        "model_predictions_used": False,
        "outcome_values_used": False,
        "drug_audit": {
            "entity_count": 6,
            "raw_duplicate_group_count": 0,
            "exact_feature_collision_group_count": 0,
            "distinct_raw_feature_collision_pair_count": 0,
        },
        "target_audit": {
            "entity_count": 6,
            "raw_duplicate_group_count": 1,
            "exact_feature_collision_group_count": 1,
            "distinct_raw_feature_collision_pair_count": 0,
        },
        "interpretation_limits": ["Descriptor limitations apply."],
    }
    paths = {
        "inner_cv_report": _write_json(tmp_path / "inner_cv.json", inner_cv),
        "threshold_sensitivity_report": _write_json(
            tmp_path / "threshold.json", threshold
        ),
        "split_audit_report": _write_json(tmp_path / "split.json", split),
        "collision_audit_report": _write_json(tmp_path / "collision.json", collision),
        "dataset_provenance": tmp_path / "dataset_provenance.md",
        "requirements_file": tmp_path / "requirements.txt",
    }
    paths["dataset_provenance"].write_text(
        "# Provenance\n\nPinned commit: a546a8433a6822e958f36171c4356ad6f414d623\n",
        encoding="utf-8",
    )
    paths["requirements_file"].write_text(
        "numpy==2.5.2\nscikit-learn==1.9.0\n", encoding="utf-8"
    )
    return {
        "inner_cv": inner_cv,
        "threshold": threshold,
        "split": split,
        "collision": collision,
    }, paths


def test_builds_noncausal_final_synthesis(tmp_path: Path) -> None:
    payloads, paths = synthetic_inputs(tmp_path)

    run = build_final_synthesis(
        inner_cv_report=payloads["inner_cv"],
        threshold_report=payloads["threshold"],
        split_audit=payloads["split"],
        collision_report=payloads["collision"],
        input_paths=paths,
        source_git_commit="abc123",
    )

    selection = run.summary["primary_model_comparison"]["pre_specified_selection"]
    assert selection["selected_model_id"] == "random_forest_balanced"
    assert selection["runner_up_model_id"] == "hist_gradient_boosting_balanced"
    assert selection["mean_average_precision_margin"] == pytest.approx(0.02)
    assert run.summary["threshold_sensitivity"]["outer_test_partition_used"] is False
    assert len(run.summary["threshold_sensitivity"]["variants"]) == 2
    assert "does not train a new model" in run.markdown
    assert "Causal claims" in run.markdown


def test_rejects_reopened_threshold_model_selection(tmp_path: Path) -> None:
    payloads, paths = synthetic_inputs(tmp_path)
    payloads["threshold"]["model_selection"]["selection_reopened"] = True

    with pytest.raises(FinalSynthesisError, match="selection_reopened"):
        build_final_synthesis(
            inner_cv_report=payloads["inner_cv"],
            threshold_report=payloads["threshold"],
            split_audit=payloads["split"],
            collision_report=payloads["collision"],
            input_paths=paths,
            source_git_commit="abc123",
        )


def test_writers_and_command_line_entrypoint(tmp_path: Path) -> None:
    payloads, paths = synthetic_inputs(tmp_path)
    run = build_final_synthesis(
        inner_cv_report=payloads["inner_cv"],
        threshold_report=payloads["threshold"],
        split_audit=payloads["split"],
        collision_report=payloads["collision"],
        input_paths=paths,
        source_git_commit="abc123",
    )
    json_path = write_final_synthesis_summary(run, tmp_path / "summary.json")
    markdown_path = write_final_synthesis_markdown(run, tmp_path / "summary.md")

    assert read_json_report(json_path)["schema_version"] == 1
    assert "Final Evidence Synthesis" in markdown_path.read_text(encoding="utf-8")

    cli_json = tmp_path / "cli_summary.json"
    cli_markdown = tmp_path / "cli_summary.md"
    exit_code = main(
        [
            "--inner-cv-report",
            str(paths["inner_cv_report"]),
            "--threshold-sensitivity-report",
            str(paths["threshold_sensitivity_report"]),
            "--split-audit-report",
            str(paths["split_audit_report"]),
            "--collision-audit-report",
            str(paths["collision_audit_report"]),
            "--dataset-provenance",
            str(paths["dataset_provenance"]),
            "--requirements-file",
            str(paths["requirements_file"]),
            "--source-git-commit",
            "abc123",
            "--summary-output",
            str(cli_json),
            "--markdown-output",
            str(cli_markdown),
        ]
    )

    assert exit_code == 0
    assert cli_json.exists()
    assert cli_markdown.exists()
