"""Tests for official UNSW-NB15 evaluation reporting."""

from __future__ import annotations

from dataclasses import fields
from inspect import signature
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from cyber_pca.evaluation import (
    evaluate_binary_predictions,
)
from cyber_pca.unsw_evaluation import (
    UNSW_ATTACK_CATEGORY_ORDER,
    align_unsw_evaluation_data,
    evaluate_unsw_attack_categories,
)
from cyber_pca.unsw_experiment import (
    run_unsw_detection,
)
from cyber_pca.unsw_preprocessing import (
    UNSWPreprocessor,
    UNSWStandardizedDataSplits,
)
from cyber_pca.unsw_reporting import (
    UNSWEvaluationArtifacts,
    build_unsw_evaluation_summary,
    resolve_unsw_evaluation_artifacts,
    write_unsw_evaluation_artifacts,
)


def test_unsw_reporting_public_contract(
) -> None:
    assert tuple(
        field.name
        for field in fields(
            UNSWEvaluationArtifacts
        )
    ) == (
        "summary_json",
        "predictions_csv",
        "metrics_csv",
        "attack_category_metrics_csv",
        "confusion_matrix_figure",
        "reconstruction_errors_figure",
        "scree_plot_figure",
        "attack_category_rates_figure",
    )

    assert tuple(
        signature(
            resolve_unsw_evaluation_artifacts
        ).parameters
    ) == (
        "output_root",
    )

    assert tuple(
        signature(
            build_unsw_evaluation_summary
        ).parameters
    ) == (
        "detection_result",
        "binary_result",
        "attack_category_result",
    )

    assert tuple(
        signature(
            write_unsw_evaluation_artifacts
        ).parameters
    ) == (
        "evaluation_data",
        "detection_result",
        "binary_result",
        "attack_category_result",
        "output_root",
        "dpi",
    )


def test_resolves_unsw_artifact_paths(
    tmp_path: Path,
) -> None:
    artifacts = (
        resolve_unsw_evaluation_artifacts(
            tmp_path
        )
    )

    assert artifacts.summary_json == (
        tmp_path
        / "results"
        / "unsw_nb15_evaluation.json"
    )
    assert artifacts.predictions_csv == (
        tmp_path
        / "results"
        / "unsw_nb15_predictions.csv"
    )
    assert artifacts.metrics_csv == (
        tmp_path
        / "reports"
        / "tables"
        / "unsw_nb15_metrics.csv"
    )
    assert (
        artifacts.attack_category_metrics_csv
        == tmp_path
        / "reports"
        / "tables"
        / (
            "unsw_nb15_"
            "attack_category_metrics.csv"
        )
    )
    assert (
        artifacts.confusion_matrix_figure
        == tmp_path
        / "reports"
        / "figures"
        / "unsw_nb15_confusion_matrix.png"
    )
    assert (
        artifacts.reconstruction_errors_figure
        == tmp_path
        / "reports"
        / "figures"
        / (
            "unsw_nb15_"
            "reconstruction_errors.png"
        )
    )
    assert (
        artifacts.scree_plot_figure
        == tmp_path
        / "reports"
        / "figures"
        / "unsw_nb15_scree_plot.png"
    )
    assert (
        artifacts.attack_category_rates_figure
        == tmp_path
        / "reports"
        / "figures"
        / (
            "unsw_nb15_"
            "attack_category_rates.png"
        )
    )

def _reporting_fixture(
) -> tuple[
    pd.DataFrame,
    object,
    object,
    pd.DataFrame,
]:
    columns = (
        "duration",
        "packet_rate",
        "proto_tcp",
    )

    raw_fit = np.asarray(
        [
            [-3.0, -2.8, 0.0],
            [-2.0, -2.1, 1.0],
            [-1.0, -0.8, 0.0],
            [-0.4, -0.2, 1.0],
            [0.3, 0.5, 0.0],
            [1.1, 0.9, 1.0],
            [2.0, 2.2, 0.0],
            [3.0, 2.7, 1.0],
        ],
        dtype=np.float64,
    )

    raw_calibration = np.asarray(
        [
            [-1.5, -1.2, 0.0],
            [-0.2, 0.1, 1.0],
            [0.8, 1.0, 0.0],
            [2.4, 2.0, 1.0],
        ],
        dtype=np.float64,
    )

    raw_test_values = np.asarray(
        [
            [-1.2, -1.0, 0.0],
            [-0.8, -0.7, 1.0],
            [-0.3, 0.0, 0.0],
            [0.1, 0.4, 1.0],
            [0.5, 0.8, 0.0],
            [1.0, 1.2, 1.0],
            [1.5, 1.7, 0.0],
            [2.1, 2.4, 1.0],
            [3.5, -2.5, 0.0],
            [4.5, -3.5, 1.0],
        ],
        dtype=np.float64,
    )

    scaler = StandardScaler()
    scaler.fit(raw_fit)

    normal_fit = pd.DataFrame(
        scaler.transform(raw_fit),
        columns=columns,
        index=pd.Index(
            [
                f"unsw_training:{value}"
                for value in range(1, 9)
            ],
            name="flow_id",
        ),
    )

    normal_calibration = pd.DataFrame(
        scaler.transform(
            raw_calibration
        ),
        columns=columns,
        index=pd.Index(
            [
                f"unsw_training:{value}"
                for value in range(9, 13)
            ],
            name="flow_id",
        ),
    )

    test = pd.DataFrame(
        scaler.transform(
            raw_test_values
        ),
        columns=columns,
        index=pd.Index(
            [
                f"unsw_testing:{value}"
                for value in range(1, 11)
            ],
            name="flow_id",
        ),
    )

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )

    encoder.fit(
        pd.DataFrame(
            {
                "proto": [
                    "tcp",
                    "udp",
                ]
            }
        )
    )

    standardized = (
        UNSWStandardizedDataSplits(
            normal_fit=normal_fit,
            normal_calibration=(
                normal_calibration
            ),
            test=test,
            preprocessor=UNSWPreprocessor(
                encoder=encoder,
                scaler=scaler,
                feature_names=columns,
            ),
        )
    )

    detection = run_unsw_detection(
        standardized,
        explained_variance_target=0.80,
        threshold_quantile=0.75,
    )

    raw_test = pd.DataFrame(
        {
            "id": np.arange(
                1,
                11,
                dtype=np.int64,
            ),
            "label": np.asarray(
                [0] + [1] * 9,
                dtype=np.int8,
            ),
            "attack_cat": (
                UNSW_ATTACK_CATEGORY_ORDER
            ),
        }
    )

    evaluation_data = (
        align_unsw_evaluation_data(
            raw_test,
            detection
            .reconstruction_errors.test,
            detection.test_predictions,
        )
    )

    binary_result = (
        evaluate_binary_predictions(
            evaluation_data
        )
    )

    category_result = (
        evaluate_unsw_attack_categories(
            evaluation_data
        )
    )

    return (
        evaluation_data,
        detection,
        binary_result,
        category_result,
    )


def test_builds_serializable_unsw_summary(
) -> None:
    (
        evaluation_data,
        detection,
        binary_result,
        category_result,
    ) = _reporting_fixture()

    summary = (
        build_unsw_evaluation_summary(
            detection,
            binary_result,
            category_result,
        )
    )

    serialized = json.dumps(
        summary,
        allow_nan=False,
        sort_keys=True,
    )

    assert serialized
    assert summary["status"] == "passed"
    assert summary["phase"] == 8
    assert summary["dataset"] == (
        "UNSW-NB15"
    )
    assert summary["data"][
        "test_observations"
    ] == evaluation_data.shape[0]
    assert summary["pca"][
        "selected_components"
    ] == detection.fit_result.n_components
    assert summary["threshold"][
        "value"
    ] == (
        detection.threshold_result.threshold
    )
    assert summary["metrics"][
        "confusion_matrix"
    ] == [
        list(row)
        for row in (
            binary_result.confusion_matrix
        )
    ]
    assert len(
        summary["attack_categories"]
    ) == 10
    assert summary["protocol"][
        "post_evaluation_tuning"
    ] is False
    assert (
        "untuned"
        in summary["limitations"].casefold()
    )


def test_writes_complete_unsw_artifacts(
    tmp_path: Path,
) -> None:
    (
        evaluation_data,
        detection,
        binary_result,
        category_result,
    ) = _reporting_fixture()

    artifacts = (
        write_unsw_evaluation_artifacts(
            evaluation_data,
            detection,
            binary_result,
            category_result,
            output_root=tmp_path,
            dpi=72,
        )
    )

    expected_relative_paths = (
        "results/unsw_nb15_evaluation.json",
        "results/unsw_nb15_predictions.csv",
        (
            "reports/tables/"
            "unsw_nb15_metrics.csv"
        ),
        (
            "reports/tables/"
            "unsw_nb15_"
            "attack_category_metrics.csv"
        ),
        (
            "reports/figures/"
            "unsw_nb15_confusion_matrix.png"
        ),
        (
            "reports/figures/"
            "unsw_nb15_"
            "reconstruction_errors.png"
        ),
        (
            "reports/figures/"
            "unsw_nb15_scree_plot.png"
        ),
        (
            "reports/figures/"
            "unsw_nb15_"
            "attack_category_rates.png"
        ),
    )

    actual_relative_paths = tuple(
        sorted(
            path.relative_to(
                tmp_path
            ).as_posix()
            for path in tmp_path.rglob("*")
            if path.is_file()
        )
    )

    assert actual_relative_paths == tuple(
        sorted(expected_relative_paths)
    )

    for artifact_path in (
        artifacts.summary_json,
        artifacts.predictions_csv,
        artifacts.metrics_csv,
        artifacts.attack_category_metrics_csv,
    ):
        assert artifact_path.read_bytes().endswith(
            b"\n"
        )

    summary = json.loads(
        artifacts.summary_json.read_text(
            encoding="utf-8"
        )
    )

    assert summary["phase"] == 8
    assert summary["status"] == "passed"

    predictions = pd.read_csv(
        artifacts.predictions_csv
    )

    assert predictions.shape[0] == (
        evaluation_data.shape[0]
    )
    assert tuple(predictions.columns) == (
        "flow_id",
        "true_anomaly",
        "predicted_anomaly",
        "scenario",
        "reconstruction_error",
    )

    metrics = pd.read_csv(
        artifacts.metrics_csv
    )

    assert metrics.shape[0] == 1
    assert metrics.loc[
        0,
        "total",
    ] == binary_result.total

    category_metrics = pd.read_csv(
        artifacts
        .attack_category_metrics_csv
    )

    assert category_metrics.shape[0] == 10
    assert category_metrics[
        "attack_category"
    ].tolist() == list(
        UNSW_ATTACK_CATEGORY_ORDER
    )

    for figure_path in (
        artifacts.confusion_matrix_figure,
        artifacts.reconstruction_errors_figure,
        artifacts.scree_plot_figure,
        artifacts.attack_category_rates_figure,
    ):
        with Image.open(
            figure_path
        ) as image:
            image.verify()
            assert image.format == "PNG"
