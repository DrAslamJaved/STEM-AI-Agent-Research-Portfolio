"""Tests for the frozen Phase 8 UNSW-NB15 contract."""

from __future__ import annotations

from pathlib import Path

import yaml


ATTACK_CATEGORY_ORDER = [
    "Normal",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]


def test_phase_eight_configuration_contract(
) -> None:
    configuration = yaml.safe_load(
        Path(
            "configs/baseline.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    contract = configuration[
        "unsw_evaluation"
    ]

    assert contract["phase"] == 8
    assert contract["dataset_name"] == (
        "UNSW-NB15"
    )
    assert contract["model_input"] == (
        "phase_07_standardized_partitions"
    )
    assert contract["pca_fit_split"] == (
        "normal_fit_only"
    )
    assert contract[
        "component_selection_split"
    ] == "normal_fit_only"
    assert contract[
        "threshold_calibration_split"
    ] == "normal_calibration_only"
    assert contract["prediction_split"] == (
        "official_test_features_only"
    )
    assert contract["label_access"] == (
        "after_predictions_frozen"
    )
    assert contract[
        "attack_category_access"
    ] == "after_predictions_frozen"
    assert contract["alignment_key"] == [
        "source_partition",
        "id",
    ]
    assert contract[
        "test_source_partition"
    ] == "unsw_testing"
    assert (
        contract["post_evaluation_tuning"]
        is False
    )
    assert contract[
        "attack_category_order"
    ] == ATTACK_CATEGORY_ORDER
    assert contract["figure_dpi"] == 150
    assert contract["table_format"] == "csv"
    assert contract["figure_format"] == "png"

    assert contract["output_paths"] == {
        "summary_json": (
            "results/"
            "unsw_nb15_evaluation.json"
        ),
        "predictions_csv": (
            "results/"
            "unsw_nb15_predictions.csv"
        ),
        "metrics_csv": (
            "reports/tables/"
            "unsw_nb15_metrics.csv"
        ),
        "attack_category_metrics_csv": (
            "reports/tables/"
            "unsw_nb15_attack_category_metrics.csv"
        ),
        "confusion_matrix_figure": (
            "reports/figures/"
            "unsw_nb15_confusion_matrix.png"
        ),
        "reconstruction_error_figure": (
            "reports/figures/"
            "unsw_nb15_reconstruction_errors.png"
        ),
        "scree_plot_figure": (
            "reports/figures/"
            "unsw_nb15_scree_plot.png"
        ),
        "attack_category_rates_figure": (
            "reports/figures/"
            "unsw_nb15_attack_category_rates.png"
        ),
    }

    for output_path in (
        contract["output_paths"].values()
    ):
        assert not Path(
            output_path
        ).is_relative_to(
            Path("data/raw")
        )

    assert (
        configuration["pca"][
            "explained_variance_target"
        ]
        == 0.95
    )
    assert (
        configuration["threshold"][
            "quantile"
        ]
        == 0.99
    )
    assert (
        configuration["threshold"][
            "comparison"
        ]
        == "strictly_greater_than"
    )

def test_phase_eight_public_exports() -> None:
    from cyber_pca import (
        UNSWDetectionResult,
        UNSWEvaluationArtifacts,
        align_unsw_evaluation_data,
        build_unsw_evaluation_summary,
        compute_unsw_reconstruction_errors,
        evaluate_unsw_attack_categories,
        fit_unsw_normal_pca,
        resolve_unsw_evaluation_artifacts,
        run_unsw_detection,
        write_unsw_evaluation_artifacts,
    )

    assert UNSWDetectionResult is not None
    assert UNSWEvaluationArtifacts is not None

    for function in (
        align_unsw_evaluation_data,
        build_unsw_evaluation_summary,
        compute_unsw_reconstruction_errors,
        evaluate_unsw_attack_categories,
        fit_unsw_normal_pca,
        resolve_unsw_evaluation_artifacts,
        run_unsw_detection,
        write_unsw_evaluation_artifacts,
    ):
        assert callable(function)
