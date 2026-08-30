"""Tests for label-blind UNSW-NB15 detection orchestration."""

from __future__ import annotations

from dataclasses import fields
from inspect import signature

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from cyber_pca.detector import (
    AnomalyThresholdResult,
    ReconstructionErrorSplits,
)
from cyber_pca.pca_workflow import PCAFitResult
from cyber_pca.unsw_experiment import (
    UNSWDetectionResult,
    compute_unsw_reconstruction_errors,
    fit_unsw_normal_pca,
    run_unsw_detection,
)
from cyber_pca.unsw_preprocessing import (
    UNSWPreprocessor,
    UNSWStandardizedDataSplits,
)


def test_unsw_detection_public_contract() -> None:
    assert tuple(
        field.name
        for field in fields(
            UNSWDetectionResult
        )
    ) == (
        "fit_result",
        "reconstruction_errors",
        "threshold_result",
        "test_predictions",
    )

    fit_parameters = signature(
        fit_unsw_normal_pca
    ).parameters

    assert tuple(fit_parameters) == (
        "splits",
        "explained_variance_target",
        "eigenvalue_tolerance",
    )
    assert (
        fit_parameters[
            "explained_variance_target"
        ].default
        == 0.95
    )
    assert (
        fit_parameters[
            "eigenvalue_tolerance"
        ].default
        == 1e-12
    )

    error_parameters = signature(
        compute_unsw_reconstruction_errors
    ).parameters

    assert tuple(error_parameters) == (
        "splits",
        "fit_result",
    )

    detection_parameters = signature(
        run_unsw_detection
    ).parameters

    assert tuple(detection_parameters) == (
        "splits",
        "explained_variance_target",
        "eigenvalue_tolerance",
        "threshold_quantile",
        "quantile_method",
    )

    assert (
        detection_parameters[
            "explained_variance_target"
        ].default
        == 0.95
    )
    assert (
        detection_parameters[
            "eigenvalue_tolerance"
        ].default
        == 1e-12
    )
    assert (
        detection_parameters[
            "threshold_quantile"
        ].default
        == 0.99
    )
    assert (
        detection_parameters[
            "quantile_method"
        ].default
        == "linear"
    )

def _standardized_unsw_fixture(
) -> UNSWStandardizedDataSplits:
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

    raw_test = np.asarray(
        [
            [-1.2, -1.0, 0.0],
            [0.2, 0.4, 1.0],
            [2.6, 2.4, 0.0],
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
        scaler.transform(raw_test),
        columns=columns,
        index=pd.Index(
            [
                f"unsw_testing:{value}"
                for value in range(1, 5)
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

    preprocessor = UNSWPreprocessor(
        encoder=encoder,
        scaler=scaler,
        feature_names=columns,
    )

    return UNSWStandardizedDataSplits(
        normal_fit=normal_fit,
        normal_calibration=(
            normal_calibration
        ),
        test=test,
        preprocessor=preprocessor,
    )


def test_runs_label_blind_unsw_detection(
) -> None:
    splits = _standardized_unsw_fixture()

    result = run_unsw_detection(
        splits,
        explained_variance_target=0.80,
        threshold_quantile=0.75,
    )

    assert isinstance(
        result.fit_result,
        PCAFitResult,
    )
    assert isinstance(
        result.reconstruction_errors,
        ReconstructionErrorSplits,
    )
    assert isinstance(
        result.threshold_result,
        AnomalyThresholdResult,
    )
    assert isinstance(
        result.test_predictions,
        pd.Series,
    )
    assert (
        result.test_predictions.name
        == "predicted_anomaly"
    )

    assert (
        result.threshold_result.calibration_count
        == splits.normal_calibration.shape[0]
    )
    assert result.test_predictions.index.equals(
        splits.test.index
    )
    assert set(
        result.test_predictions.tolist()
    ).issubset({0, 1})

    assert (
        result.reconstruction_errors.normal_fit
        .index.equals(
            splits.normal_fit.index
        )
    )
    assert (
        result.reconstruction_errors
        .normal_calibration.index.equals(
            splits.normal_calibration.index
        )
    )
    assert (
        result.reconstruction_errors.test
        .index.equals(
            splits.test.index
        )
    )


def test_calibration_and_test_do_not_fit_pca(
) -> None:
    splits = _standardized_unsw_fixture()

    baseline = run_unsw_detection(
        splits,
        explained_variance_target=0.80,
        threshold_quantile=0.75,
    )

    changed_calibration = (
        splits.normal_calibration.copy(
            deep=True
        )
    )
    changed_calibration.iloc[:, :] += 25.0

    changed_test = splits.test.copy(
        deep=True
    )
    changed_test.iloc[:, :] -= 30.0

    changed_splits = (
        UNSWStandardizedDataSplits(
            normal_fit=splits.normal_fit,
            normal_calibration=(
                changed_calibration
            ),
            test=changed_test,
            preprocessor=splits.preprocessor,
        )
    )

    changed = run_unsw_detection(
        changed_splits,
        explained_variance_target=0.80,
        threshold_quantile=0.75,
    )

    assert (
        changed.fit_result.n_components
        == baseline.fit_result.n_components
    )
    assert (
        changed.fit_result
        .achieved_explained_variance
        == baseline.fit_result
        .achieved_explained_variance
    )

    np.testing.assert_allclose(
        changed.fit_result
        .full_explained_variance,
        baseline.fit_result
        .full_explained_variance,
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_allclose(
        changed.fit_result.model.components_,
        baseline.fit_result.model.components_,
        rtol=0.0,
        atol=0.0,
    )

    test_only_splits = (
        UNSWStandardizedDataSplits(
            normal_fit=splits.normal_fit,
            normal_calibration=(
                splits.normal_calibration
            ),
            test=changed_test,
            preprocessor=splits.preprocessor,
        )
    )

    test_only_result = run_unsw_detection(
        test_only_splits,
        explained_variance_target=0.80,
        threshold_quantile=0.75,
    )

    assert (
        test_only_result.threshold_result
        == baseline.threshold_result
    )
