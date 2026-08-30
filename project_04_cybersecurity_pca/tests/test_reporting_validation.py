"""Validation tests for Phase 6 reporting boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cyber_pca.detector import (
    calibrate_anomaly_threshold,
    compute_reconstruction_errors,
    predict_anomalies,
)
from cyber_pca.evaluation import (
    align_evaluation_data,
    evaluate_binary_predictions,
    evaluate_scenarios,
)
from cyber_pca.pca_workflow import fit_normal_pca
from cyber_pca.preprocessing import (
    split_normal_calibration_test,
    standardize_splits,
)
from cyber_pca.reporting import (
    _to_builtin,
    build_synthetic_evaluation_summary,
    resolve_synthetic_evaluation_artifacts,
    write_synthetic_evaluation_artifacts,
)
from cyber_pca.synthetic_data import (
    generate_synthetic_network_data,
)


@pytest.fixture(scope="module")
def reporting_inputs() -> SimpleNamespace:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset,
        random_seed=42,
    )

    standardized_splits = standardize_splits(
        raw_splits
    )

    fit_result = fit_normal_pca(
        standardized_splits
    )

    reconstruction_errors = (
        compute_reconstruction_errors(
            standardized_splits,
            fit_result,
        )
    )

    threshold_result = (
        calibrate_anomaly_threshold(
            reconstruction_errors
        )
    )

    predictions = predict_anomalies(
        reconstruction_errors.test,
        threshold_result,
    )

    evaluation_data = align_evaluation_data(
        raw_splits.test,
        reconstruction_errors.test,
        predictions,
    )

    binary_result = evaluate_binary_predictions(
        evaluation_data
    )

    scenario_result = evaluate_scenarios(
        evaluation_data
    )

    return SimpleNamespace(
        evaluation_data=evaluation_data,
        fit_result=fit_result,
        threshold_result=threshold_result,
        binary_result=binary_result,
        scenario_result=scenario_result,
    )


def _summary_arguments(
    inputs: SimpleNamespace,
) -> dict[str, object]:
    return {
        "fit_result": inputs.fit_result,
        "threshold_result": inputs.threshold_result,
        "binary_result": inputs.binary_result,
        "scenario_result": inputs.scenario_result,
    }


def _write_artifacts(
    inputs: SimpleNamespace,
    evaluation_data: object,
    output_root: Path,
    *,
    dpi: object = 150,
) -> None:
    write_synthetic_evaluation_artifacts(
        evaluation_data,
        inputs.fit_result,
        inputs.threshold_result,
        inputs.binary_result,
        inputs.scenario_result,
        output_root=output_root,
        dpi=dpi,
    )


def test_to_builtin_converts_numpy_scalar() -> None:
    converted = _to_builtin(
        np.float64(1.25)
    )

    assert converted == pytest.approx(1.25)
    assert isinstance(converted, float)


@pytest.mark.parametrize(
    "invalid_output_root",
    [
        None,
        3.14,
    ],
)
def test_artifact_resolution_rejects_invalid_root(
    invalid_output_root: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="output_root",
    ):
        resolve_synthetic_evaluation_artifacts(
            invalid_output_root
        )


@pytest.mark.parametrize(
    (
        "argument_name",
        "invalid_value",
    ),
    [
        ("fit_result", object()),
        ("threshold_result", object()),
        ("binary_result", object()),
        ("scenario_result", object()),
    ],
)
def test_summary_rejects_invalid_argument_types(
    reporting_inputs: SimpleNamespace,
    argument_name: str,
    invalid_value: object,
) -> None:
    arguments = _summary_arguments(
        reporting_inputs
    )
    arguments[argument_name] = invalid_value

    with pytest.raises(TypeError):
        build_synthetic_evaluation_summary(
            **arguments
        )


def test_summary_rejects_empty_scenario_table(
    reporting_inputs: SimpleNamespace,
) -> None:
    arguments = _summary_arguments(
        reporting_inputs
    )
    arguments["scenario_result"] = (
        reporting_inputs.scenario_result.iloc[
            0:0
        ].copy()
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_synthetic_evaluation_summary(
            **arguments
        )


def test_summary_rejects_incorrect_scenario_columns(
    reporting_inputs: SimpleNamespace,
) -> None:
    arguments = _summary_arguments(
        reporting_inputs
    )

    invalid_scenarios = (
        reporting_inputs.scenario_result.rename(
            columns={
                "scenario": "unexpected_scenario"
            }
        )
    )

    arguments["scenario_result"] = (
        invalid_scenarios
    )

    with pytest.raises(
        ValueError,
        match="columns must exactly match",
    ):
        build_synthetic_evaluation_summary(
            **arguments
        )


def test_summary_rejects_scenario_count_mismatch(
    reporting_inputs: SimpleNamespace,
) -> None:
    arguments = _summary_arguments(
        reporting_inputs
    )

    invalid_scenarios = (
        reporting_inputs.scenario_result.copy(
            deep=True
        )
    )

    invalid_scenarios.loc[
        invalid_scenarios.index[0],
        "observations",
    ] += 1

    arguments["scenario_result"] = (
        invalid_scenarios
    )

    with pytest.raises(
        ValueError,
        match="observations do not match",
    ):
        build_synthetic_evaluation_summary(
            **arguments
        )


def test_writer_rejects_nondatframe_evaluation(
    reporting_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        _write_artifacts(
            reporting_inputs,
            [],
            tmp_path / "artifacts",
        )


def test_writer_rejects_empty_evaluation(
    reporting_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    empty_evaluation = (
        reporting_inputs.evaluation_data.iloc[
            0:0
        ].copy()
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        _write_artifacts(
            reporting_inputs,
            empty_evaluation,
            tmp_path / "artifacts",
        )


def test_writer_rejects_incorrect_columns(
    reporting_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    invalid_evaluation = (
        reporting_inputs.evaluation_data.rename(
            columns={
                "true_anomaly": "true_label"
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="columns must exactly match",
    ):
        _write_artifacts(
            reporting_inputs,
            invalid_evaluation,
            tmp_path / "artifacts",
        )


def test_writer_rejects_unnamed_index(
    reporting_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    invalid_evaluation = (
        reporting_inputs.evaluation_data.copy(
            deep=True
        )
    )
    invalid_evaluation.index.name = None

    with pytest.raises(
        ValueError,
        match="index must be named",
    ):
        _write_artifacts(
            reporting_inputs,
            invalid_evaluation,
            tmp_path / "artifacts",
        )


@pytest.mark.parametrize(
    "invalid_dpi",
    [
        True,
        150.0,
    ],
)
def test_writer_rejects_noninteger_dpi(
    reporting_inputs: SimpleNamespace,
    tmp_path: Path,
    invalid_dpi: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="dpi must be an integer",
    ):
        _write_artifacts(
            reporting_inputs,
            reporting_inputs.evaluation_data,
            tmp_path / "artifacts",
            dpi=invalid_dpi,
        )


@pytest.mark.parametrize(
    "invalid_dpi",
    [
        0,
        -1,
    ],
)
def test_writer_rejects_nonpositive_dpi(
    reporting_inputs: SimpleNamespace,
    tmp_path: Path,
    invalid_dpi: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="dpi must be positive",
    ):
        _write_artifacts(
            reporting_inputs,
            reporting_inputs.evaluation_data,
            tmp_path / "artifacts",
            dpi=invalid_dpi,
        )
