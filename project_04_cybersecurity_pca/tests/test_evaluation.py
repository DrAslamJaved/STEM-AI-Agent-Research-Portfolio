"""Tests for synthetic anomaly-evaluation logic."""

from dataclasses import fields, is_dataclass

import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from cyber_pca.evaluation import (
    BinaryEvaluationResult,
    align_evaluation_data,
    evaluate_binary_predictions,
    evaluate_scenarios,
)


def test_binary_evaluation_result_contract() -> None:
    assert is_dataclass(BinaryEvaluationResult)

    assert tuple(
        field.name
        for field in fields(BinaryEvaluationResult)
    ) == (
        "total",
        "normal_support",
        "anomaly_support",
        "predicted_normal",
        "predicted_anomaly",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "confusion_matrix",
    )


def test_evaluation_functions_are_callable() -> None:
    assert callable(align_evaluation_data)
    assert callable(evaluate_binary_predictions)
    assert callable(evaluate_scenarios)


def _alignment_inputs() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    raw_test = pd.DataFrame(
        {
            "flow_id": [
                "flow-c",
                "flow-a",
                "flow-b",
            ],
            "is_anomaly": [1, 0, 1],
            "scenario": [
                "dos",
                "normal",
                "port_scan",
            ],
            "bytes_in": [
                900.0,
                100.0,
                700.0,
            ],
        }
    )

    evaluation_index = pd.Index(
        [
            "flow-b",
            "flow-a",
            "flow-c",
        ],
        name="flow_id",
    )

    reconstruction_errors = pd.Series(
        [0.80, 0.10, 1.20],
        index=evaluation_index,
        name="reconstruction_error",
        dtype=np.float64,
    )

    predictions = pd.Series(
        [1, 0, 1],
        index=evaluation_index,
        name="is_anomaly",
        dtype=np.int8,
    )

    return (
        raw_test,
        reconstruction_errors,
        predictions,
    )


def test_align_evaluation_data_contract() -> None:
    (
        raw_test,
        reconstruction_errors,
        predictions,
    ) = _alignment_inputs()

    aligned = align_evaluation_data(
        raw_test,
        reconstruction_errors,
        predictions,
    )

    expected_index = pd.Index(
        [
            "flow-b",
            "flow-a",
            "flow-c",
        ],
        name="flow_id",
    )

    assert aligned.index.equals(expected_index)

    assert tuple(aligned.columns) == (
        "true_anomaly",
        "predicted_anomaly",
        "scenario",
        "reconstruction_error",
    )

    assert aligned["true_anomaly"].tolist() == [
        1,
        0,
        1,
    ]
    assert aligned["predicted_anomaly"].tolist() == [
        1,
        0,
        1,
    ]
    assert aligned["scenario"].tolist() == [
        "port_scan",
        "normal",
        "dos",
    ]

    np.testing.assert_allclose(
        aligned["reconstruction_error"],
        [0.80, 0.10, 1.20],
        rtol=0.0,
        atol=0.0,
    )

    assert (
        aligned["true_anomaly"].dtype
        == np.dtype(np.int8)
    )
    assert (
        aligned["predicted_anomaly"].dtype
        == np.dtype(np.int8)
    )
    assert (
        aligned["reconstruction_error"].dtype
        == np.dtype(np.float64)
    )


def test_alignment_uses_flow_ids_not_row_order() -> None:
    (
        raw_test,
        reconstruction_errors,
        predictions,
    ) = _alignment_inputs()

    first = align_evaluation_data(
        raw_test,
        reconstruction_errors,
        predictions,
    )

    shuffled_raw = raw_test.sample(
        frac=1.0,
        random_state=91,
    ).reset_index(drop=True)

    second = align_evaluation_data(
        shuffled_raw,
        reconstruction_errors,
        predictions,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_alignment_does_not_mutate_inputs() -> None:
    (
        raw_test,
        reconstruction_errors,
        predictions,
    ) = _alignment_inputs()

    original_raw = raw_test.copy(deep=True)
    original_errors = (
        reconstruction_errors.copy(deep=True)
    )
    original_predictions = predictions.copy(
        deep=True
    )

    aligned = align_evaluation_data(
        raw_test,
        reconstruction_errors,
        predictions,
    )

    pd.testing.assert_frame_equal(
        raw_test,
        original_raw,
    )
    pd.testing.assert_series_equal(
        reconstruction_errors,
        original_errors,
    )
    pd.testing.assert_series_equal(
        predictions,
        original_predictions,
    )

    aligned.loc[
        "flow-b",
        "scenario",
    ] = "modified"

    pd.testing.assert_frame_equal(
        raw_test,
        original_raw,
    )

def _metric_evaluation_data() -> pd.DataFrame:
    index = pd.Index(
        [
            f"metric-flow-{position}"
            for position in range(10)
        ],
        name="flow_id",
    )

    return pd.DataFrame(
        {
            "true_anomaly": np.array(
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                dtype=np.int8,
            ),
            "predicted_anomaly": np.array(
                [0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                dtype=np.int8,
            ),
            "scenario": [
                "normal",
                "normal",
                "normal",
                "normal",
                "normal",
                "normal",
                "brute_force",
                "brute_force",
                "brute_force",
                "brute_force",
            ],
            "reconstruction_error": np.linspace(
                0.01,
                1.00,
                10,
                dtype=np.float64,
            ),
        },
        index=index,
    )


def test_binary_evaluation_counts_and_formulas() -> None:
    evaluation_data = _metric_evaluation_data()

    result = evaluate_binary_predictions(
        evaluation_data
    )

    assert result.total == 10
    assert result.normal_support == 6
    assert result.anomaly_support == 4
    assert result.predicted_normal == 5
    assert result.predicted_anomaly == 5

    assert result.true_negatives == 4
    assert result.false_positives == 2
    assert result.false_negatives == 1
    assert result.true_positives == 3

    assert result.confusion_matrix == (
        (4, 2),
        (1, 3),
    )

    assert result.precision == pytest.approx(
        3.0 / 5.0
    )
    assert result.recall == pytest.approx(
        3.0 / 4.0
    )
    assert result.f1 == pytest.approx(
        2.0 / 3.0
    )
    assert result.accuracy == pytest.approx(
        7.0 / 10.0
    )
    assert (
        result.false_positive_rate
        == pytest.approx(2.0 / 6.0)
    )
    assert (
        result.false_negative_rate
        == pytest.approx(1.0 / 4.0)
    )


def test_binary_evaluation_agrees_with_sklearn() -> None:
    evaluation_data = _metric_evaluation_data()

    result = evaluate_binary_predictions(
        evaluation_data
    )

    y_true = evaluation_data[
        "true_anomaly"
    ].to_numpy()

    y_pred = evaluation_data[
        "predicted_anomaly"
    ].to_numpy()

    expected_matrix = sklearn_confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    assert result.confusion_matrix == tuple(
        tuple(int(value) for value in row)
        for row in expected_matrix
    )

    assert result.precision == pytest.approx(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )
    assert result.recall == pytest.approx(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )
    assert result.f1 == pytest.approx(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )
    assert result.accuracy == pytest.approx(
        accuracy_score(y_true, y_pred)
    )


def test_binary_evaluation_zero_division_policy() -> None:
    index = pd.Index(
        ["normal-1", "normal-2", "normal-3"],
        name="flow_id",
    )

    evaluation_data = pd.DataFrame(
        {
            "true_anomaly": np.zeros(
                3,
                dtype=np.int8,
            ),
            "predicted_anomaly": np.zeros(
                3,
                dtype=np.int8,
            ),
            "scenario": ["normal"] * 3,
            "reconstruction_error": np.array(
                [0.01, 0.02, 0.03],
                dtype=np.float64,
            ),
        },
        index=index,
    )

    result = evaluate_binary_predictions(
        evaluation_data,
        zero_division=0,
    )

    assert result.confusion_matrix == (
        (3, 0),
        (0, 0),
    )
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.false_positive_rate == 0.0
    assert result.false_negative_rate == 0.0
    assert result.accuracy == 1.0


def test_binary_evaluation_is_order_invariant_and_nonmutating() -> None:
    evaluation_data = _metric_evaluation_data()
    original = evaluation_data.copy(deep=True)

    baseline = evaluate_binary_predictions(
        evaluation_data
    )

    shuffled = evaluation_data.sample(
        frac=1.0,
        random_state=107,
    )

    repeated = evaluate_binary_predictions(
        shuffled
    )

    assert repeated == baseline

    pd.testing.assert_frame_equal(
        evaluation_data,
        original,
    )

def _scenario_evaluation_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "true_anomaly": [
                0, 0, 0, 0,
                1, 1, 1,
                1, 1,
                1, 1,
                1,
            ],
            "predicted_anomaly": [
                0, 0, 0, 1,
                1, 1, 0,
                1, 1,
                1, 0,
                1,
            ],
            "scenario": [
                "normal",
                "normal",
                "normal",
                "normal",
                "brute_force",
                "brute_force",
                "brute_force",
                "dos",
                "dos",
                "exfiltration",
                "exfiltration",
                "port_scan",
            ],
            "reconstruction_error": [
                0.01, 0.02, 0.03, 0.20,
                1.00, 2.00, 0.10,
                10.00, 12.00,
                5.00, 0.15,
                3.00,
            ],
        },
        index=pd.Index(
            [f"flow-{index:02d}" for index in range(12)],
            name="flow_id",
        ),
    )


def test_scenario_evaluation_contract() -> None:
    evaluation_data = _scenario_evaluation_fixture()

    result = evaluate_scenarios(evaluation_data)

    assert list(result.columns) == [
        "scenario",
        "true_label",
        "observations",
        "predicted_normal",
        "predicted_anomaly",
        "predicted_anomaly_rate",
        "mean_reconstruction_error",
        "median_reconstruction_error",
        "maximum_reconstruction_error",
    ]

    assert result["scenario"].tolist() == [
        "normal",
        "brute_force",
        "dos",
        "exfiltration",
        "port_scan",
    ]

    assert result["true_label"].tolist() == [
        0, 1, 1, 1, 1,
    ]

    assert int(result["observations"].sum()) == 12


def test_scenario_evaluation_counts_and_statistics() -> None:
    result = evaluate_scenarios(
        _scenario_evaluation_fixture()
    ).set_index("scenario")

    normal = result.loc["normal"]
    assert normal["observations"] == 4
    assert normal["predicted_normal"] == 3
    assert normal["predicted_anomaly"] == 1
    assert normal["predicted_anomaly_rate"] == pytest.approx(0.25)
    assert normal["mean_reconstruction_error"] == pytest.approx(0.065)
    assert normal["median_reconstruction_error"] == pytest.approx(0.025)
    assert normal["maximum_reconstruction_error"] == pytest.approx(0.20)

    brute_force = result.loc["brute_force"]
    assert brute_force["observations"] == 3
    assert brute_force["predicted_normal"] == 1
    assert brute_force["predicted_anomaly"] == 2
    assert brute_force["predicted_anomaly_rate"] == pytest.approx(2.0 / 3.0)
    assert brute_force["mean_reconstruction_error"] == pytest.approx(
        3.1 / 3.0
    )
    assert brute_force["median_reconstruction_error"] == pytest.approx(1.0)
    assert brute_force["maximum_reconstruction_error"] == pytest.approx(2.0)

    dos = result.loc["dos"]
    assert dos["predicted_anomaly_rate"] == pytest.approx(1.0)
    assert dos["mean_reconstruction_error"] == pytest.approx(11.0)

    exfiltration = result.loc["exfiltration"]
    assert exfiltration["predicted_anomaly_rate"] == pytest.approx(0.5)
    assert exfiltration["median_reconstruction_error"] == pytest.approx(
        2.575
    )

    port_scan = result.loc["port_scan"]
    assert port_scan["predicted_anomaly_rate"] == pytest.approx(1.0)
    assert port_scan["maximum_reconstruction_error"] == pytest.approx(3.0)


def test_scenario_evaluation_is_order_invariant_and_nonmutating() -> None:
    evaluation_data = _scenario_evaluation_fixture()
    original = evaluation_data.copy(deep=True)

    baseline = evaluate_scenarios(evaluation_data)

    shuffled = evaluation_data.sample(
        frac=1.0,
        random_state=42,
    )

    reordered = evaluate_scenarios(shuffled)

    pd.testing.assert_frame_equal(
        baseline,
        reordered,
    )

    pd.testing.assert_frame_equal(
        evaluation_data,
        original,
    )
