"""Tests for synthetic evaluation reporting."""

from dataclasses import fields, is_dataclass

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from cyber_pca import (
    AnomalyThresholdResult,
    BinaryEvaluationResult,
    ManualPCA,
    PCAFitResult,
)

from cyber_pca.reporting import (
    SyntheticEvaluationArtifacts,
    build_synthetic_evaluation_summary,
    resolve_synthetic_evaluation_artifacts,
    write_synthetic_evaluation_artifacts,
)


def test_reporting_public_interface() -> None:
    assert is_dataclass(
        SyntheticEvaluationArtifacts
    )

    assert [
        field.name
        for field in fields(
            SyntheticEvaluationArtifacts
        )
    ] == [
        "summary_json",
        "predictions_csv",
        "metrics_csv",
        "scenario_metrics_csv",
        "confusion_matrix_figure",
        "reconstruction_errors_figure",
        "scree_plot_figure",
        "scenario_rates_figure",
    ]

    assert callable(
        build_synthetic_evaluation_summary
    )

    assert callable(
        resolve_synthetic_evaluation_artifacts
    )

    assert callable(
        write_synthetic_evaluation_artifacts
    )

def _reporting_inputs() -> tuple[
    PCAFitResult,
    AnomalyThresholdResult,
    BinaryEvaluationResult,
    pd.DataFrame,
]:
    fit_result = PCAFitResult(
        model=ManualPCA(n_components=2),
        n_components=2,
        explained_variance_target=0.95,
        achieved_explained_variance=0.96,
        full_explained_variance=np.array(
            [7.0, 2.6, 0.4],
            dtype=np.float64,
        ),
        full_explained_variance_ratio=np.array(
            [0.70, 0.26, 0.04],
            dtype=np.float64,
        ),
        full_cumulative_explained_variance=np.array(
            [0.70, 0.96, 1.00],
            dtype=np.float64,
        ),
    )

    threshold_result = AnomalyThresholdResult(
        threshold=0.50,
        quantile=0.99,
        quantile_method="linear",
        calibration_count=100,
    )

    binary_result = BinaryEvaluationResult(
        total=10,
        normal_support=4,
        anomaly_support=6,
        predicted_normal=4,
        predicted_anomaly=6,
        true_negatives=4,
        false_positives=0,
        false_negatives=0,
        true_positives=6,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        accuracy=1.0,
        false_positive_rate=0.0,
        false_negative_rate=0.0,
        confusion_matrix=((4, 0), (0, 6)),
    )

    scenario_result = pd.DataFrame(
        {
            "scenario": [
                "normal",
                "brute_force",
                "dos",
                "exfiltration",
                "port_scan",
            ],
            "true_label": [0, 1, 1, 1, 1],
            "observations": [4, 2, 2, 1, 1],
            "predicted_normal": [4, 0, 0, 0, 0],
            "predicted_anomaly": [0, 2, 2, 1, 1],
            "predicted_anomaly_rate": [
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "mean_reconstruction_error": [
                0.02,
                2.0,
                10.0,
                4.0,
                6.0,
            ],
            "median_reconstruction_error": [
                0.02,
                2.0,
                10.0,
                4.0,
                6.0,
            ],
            "maximum_reconstruction_error": [
                0.04,
                3.0,
                12.0,
                4.0,
                6.0,
            ],
        }
    )

    return (
        fit_result,
        threshold_result,
        binary_result,
        scenario_result,
    )


def test_resolves_expected_artifact_paths(
    tmp_path: Path,
) -> None:
    artifacts = (
        resolve_synthetic_evaluation_artifacts(
            tmp_path
        )
    )

    assert artifacts.summary_json == (
        tmp_path
        / "results"
        / "synthetic_evaluation.json"
    )

    assert artifacts.predictions_csv == (
        tmp_path
        / "results"
        / "synthetic_predictions.csv"
    )

    assert artifacts.metrics_csv == (
        tmp_path
        / "reports"
        / "tables"
        / "synthetic_metrics.csv"
    )

    assert artifacts.scenario_metrics_csv == (
        tmp_path
        / "reports"
        / "tables"
        / "synthetic_scenario_metrics.csv"
    )

    assert artifacts.confusion_matrix_figure == (
        tmp_path
        / "reports"
        / "figures"
        / "synthetic_confusion_matrix.png"
    )

    assert (
        artifacts.reconstruction_errors_figure
        == (
            tmp_path
            / "reports"
            / "figures"
            / "synthetic_reconstruction_errors.png"
        )
    )

    assert artifacts.scree_plot_figure == (
        tmp_path
        / "reports"
        / "figures"
        / "synthetic_scree_plot.png"
    )

    assert artifacts.scenario_rates_figure == (
        tmp_path
        / "reports"
        / "figures"
        / "synthetic_scenario_rates.png"
    )

    assert not any(tmp_path.rglob("*"))


def test_builds_json_serializable_summary() -> None:
    (
        fit_result,
        threshold_result,
        binary_result,
        scenario_result,
    ) = _reporting_inputs()

    summary = build_synthetic_evaluation_summary(
        fit_result,
        threshold_result,
        binary_result,
        scenario_result,
    )

    assert summary["status"] == "passed"

    assert summary["data"] == {
        "test_observations": 10,
        "normal_observations": 4,
        "anomaly_observations": 6,
    }

    assert summary["pca"][
        "selected_components"
    ] == 2

    assert summary["pca"][
        "explained_variance_target"
    ] == pytest.approx(0.95)

    assert summary["pca"][
        "achieved_explained_variance"
    ] == pytest.approx(0.96)

    assert summary["pca"][
        "full_explained_variance_ratios"
    ] == pytest.approx(
        [0.70, 0.26, 0.04]
    )

    assert summary["pca"][
        "full_cumulative_explained_variance"
    ] == pytest.approx(
        [0.70, 0.96, 1.00]
    )

    assert summary["threshold"] == {
        "value": 0.50,
        "quantile": 0.99,
        "quantile_method": "linear",
        "calibration_count": 100,
        "comparison": "strictly_greater_than",
    }

    assert summary["metrics"][
        "confusion_matrix"
    ] == [
        [4, 0],
        [0, 6],
    ]

    assert summary["metrics"]["precision"] == 1.0
    assert summary["metrics"]["recall"] == 1.0
    assert summary["metrics"]["f1"] == 1.0

    assert len(summary["scenarios"]) == 5

    assert summary["scenarios"][0][
        "scenario"
    ] == "normal"

    serialized = json.dumps(
        summary,
        sort_keys=True,
    )

    assert serialized

def _reporting_evaluation_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "true_anomaly": [
                0, 0, 0, 0,
                1, 1,
                1, 1,
                1,
                1,
            ],
            "predicted_anomaly": [
                0, 0, 0, 0,
                1, 1,
                1, 1,
                1,
                1,
            ],
            "scenario": [
                "normal",
                "normal",
                "normal",
                "normal",
                "brute_force",
                "brute_force",
                "dos",
                "dos",
                "exfiltration",
                "port_scan",
            ],
            "reconstruction_error": [
                0.00,
                0.02,
                0.02,
                0.04,
                1.00,
                3.00,
                8.00,
                12.00,
                4.00,
                6.00,
            ],
        },
        index=pd.Index(
            [
                f"flow-{index:02d}"
                for index in range(10)
            ],
            name="flow_id",
        ),
    )


def test_writes_complete_synthetic_artifacts(
    tmp_path: Path,
) -> None:
    (
        fit_result,
        threshold_result,
        binary_result,
        scenario_result,
    ) = _reporting_inputs()

    evaluation_data = (
        _reporting_evaluation_data()
    )

    original_evaluation = evaluation_data.copy(
        deep=True
    )
    original_scenarios = scenario_result.copy(
        deep=True
    )

    artifacts = (
        write_synthetic_evaluation_artifacts(
            evaluation_data,
            fit_result,
            threshold_result,
            binary_result,
            scenario_result,
            output_root=tmp_path,
            dpi=72,
        )
    )

    expected = (
        resolve_synthetic_evaluation_artifacts(
            tmp_path
        )
    )

    assert artifacts == expected

    artifact_paths = [
        getattr(artifacts, field.name)
        for field in fields(
            SyntheticEvaluationArtifacts
        )
    ]

    assert len(artifact_paths) == 8

    for artifact_path in artifact_paths:
        assert artifact_path.is_file()
        assert artifact_path.stat().st_size > 0

    summary = json.loads(
        artifacts.summary_json.read_text(
            encoding="utf-8"
        )
    )

    assert summary["status"] == "passed"
    assert summary["metrics"][
        "confusion_matrix"
    ] == [
        [4, 0],
        [0, 6],
    ]

    predictions = pd.read_csv(
        artifacts.predictions_csv
    )

    assert list(predictions.columns) == [
        "flow_id",
        "true_anomaly",
        "predicted_anomaly",
        "scenario",
        "reconstruction_error",
    ]

    assert predictions.shape == (10, 5)

    metrics = pd.read_csv(
        artifacts.metrics_csv
    )

    assert metrics.shape == (1, 16)

    assert metrics.loc[
        0,
        "true_negatives",
    ] == 4

    assert metrics.loc[
        0,
        "true_positives",
    ] == 6

    scenarios = pd.read_csv(
        artifacts.scenario_metrics_csv
    )

    pd.testing.assert_frame_equal(
        scenarios,
        scenario_result,
        check_dtype=False,
    )

    png_signature = (
        b"\x89PNG\r\n\x1a\n"
    )

    figure_paths = [
        artifacts.confusion_matrix_figure,
        artifacts.reconstruction_errors_figure,
        artifacts.scree_plot_figure,
        artifacts.scenario_rates_figure,
    ]

    for figure_path in figure_paths:
        assert (
            figure_path.read_bytes()[:8]
            == png_signature
        )
        assert figure_path.stat().st_size > 1000

    text_artifacts = [
        artifacts.summary_json,
        artifacts.predictions_csv,
        artifacts.metrics_csv,
        artifacts.scenario_metrics_csv,
    ]

    for text_artifact in text_artifacts:
        assert text_artifact.read_bytes().endswith(
            b"\n"
        )

    pd.testing.assert_frame_equal(
        evaluation_data,
        original_evaluation,
    )

    pd.testing.assert_frame_equal(
        scenario_result,
        original_scenarios,
    )
