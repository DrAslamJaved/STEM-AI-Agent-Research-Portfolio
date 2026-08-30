"""Defensive validation tests for the PCA fitting workflow."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cyber_pca.pca_manual import ManualPCA
from cyber_pca.pca_workflow import (
    fit_normal_pca,
    select_n_components,
    transform_pca_splits,
)
from cyber_pca.preprocessing import (
    StandardizedDataSplits,
    split_normal_calibration_test,
    standardize_splits,
)
from cyber_pca.synthetic_data import (
    generate_synthetic_network_data,
)


@pytest.fixture(scope="module")
def valid_standardized_splits() -> StandardizedDataSplits:
    """Create one valid, reusable standardized dataset."""
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=401,
    )

    raw_splits = split_normal_calibration_test(
        dataset,
        random_seed=402,
    )

    return standardize_splits(raw_splits)


def test_component_selection_rejects_zero_total_variance() -> None:
    ratios = np.zeros(4, dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="positive sum",
    ):
        select_n_components(
            ratios,
            explained_variance_target=0.95,
        )


def test_pca_fit_rejects_incorrect_split_type() -> None:
    with pytest.raises(
        TypeError,
        match="StandardizedDataSplits",
    ):
        fit_normal_pca(object())  # type: ignore[arg-type]


def test_pca_fit_rejects_non_dataframe_partition(
    valid_standardized_splits: StandardizedDataSplits,
) -> None:
    invalid_splits = replace(
        valid_standardized_splits,
        normal_calibration=(
            valid_standardized_splits
            .normal_calibration
            .to_numpy()
        ),
    )

    with pytest.raises(
        TypeError,
        match="normal_calibration must be a pandas DataFrame",
    ):
        fit_normal_pca(invalid_splits)


def test_pca_fit_rejects_empty_partition(
    valid_standardized_splits: StandardizedDataSplits,
) -> None:
    empty_calibration = (
        valid_standardized_splits
        .normal_calibration
        .iloc[0:0]
        .copy()
    )

    invalid_splits = replace(
        valid_standardized_splits,
        normal_calibration=empty_calibration,
    )

    with pytest.raises(
        ValueError,
        match="normal_calibration must not be empty",
    ):
        fit_normal_pca(invalid_splits)


def test_pca_fit_rejects_incorrect_feature_columns(
    valid_standardized_splits: StandardizedDataSplits,
) -> None:
    invalid_fit = (
        valid_standardized_splits
        .normal_fit
        .copy()
    )

    invalid_fit = invalid_fit.rename(
        columns={
            invalid_fit.columns[0]: "unexpected_feature",
        }
    )

    invalid_splits = replace(
        valid_standardized_splits,
        normal_fit=invalid_fit,
    )

    with pytest.raises(
        ValueError,
        match="columns must exactly match",
    ):
        fit_normal_pca(invalid_splits)


def test_pca_fit_rejects_unnamed_identifier_index(
    valid_standardized_splits: StandardizedDataSplits,
) -> None:
    invalid_fit = (
        valid_standardized_splits
        .normal_fit
        .copy()
    )
    invalid_fit.index.name = None

    invalid_splits = replace(
        valid_standardized_splits,
        normal_fit=invalid_fit,
    )

    with pytest.raises(
        ValueError,
        match="index must be named flow_id",
    ):
        fit_normal_pca(invalid_splits)


def test_pca_fit_rejects_missing_flow_identifier(
    valid_standardized_splits: StandardizedDataSplits,
) -> None:
    invalid_calibration = (
        valid_standardized_splits
        .normal_calibration
        .copy()
    )

    identifiers = invalid_calibration.index.tolist()
    identifiers[0] = np.nan

    invalid_calibration.index = pd.Index(
        identifiers,
        name="flow_id",
    )

    invalid_splits = replace(
        valid_standardized_splits,
        normal_calibration=invalid_calibration,
    )

    with pytest.raises(
        ValueError,
        match="missing flow IDs",
    ):
        fit_normal_pca(invalid_splits)


def test_pca_fit_rejects_duplicate_flow_identifiers(
    valid_standardized_splits: StandardizedDataSplits,
) -> None:
    invalid_test = (
        valid_standardized_splits
        .test
        .copy()
    )

    identifiers = invalid_test.index.tolist()
    identifiers[1] = identifiers[0]

    invalid_test.index = pd.Index(
        identifiers,
        name="flow_id",
    )

    invalid_splits = replace(
        valid_standardized_splits,
        test=invalid_test,
    )

    with pytest.raises(
        ValueError,
        match="duplicate flow IDs",
    ):
        fit_normal_pca(invalid_splits)


def test_pca_fit_rejects_nonfinite_values(
    valid_standardized_splits: StandardizedDataSplits,
) -> None:
    invalid_calibration = (
        valid_standardized_splits
        .normal_calibration
        .copy()
    )
    invalid_calibration.iloc[0, 0] = np.nan

    invalid_splits = replace(
        valid_standardized_splits,
        normal_calibration=invalid_calibration,
    )

    with pytest.raises(
        ValueError,
        match="nonfinite values",
    ):
        fit_normal_pca(invalid_splits)


def test_pca_fit_requires_two_fitting_observations(
    valid_standardized_splits: StandardizedDataSplits,
) -> None:
    one_row_fit = (
        valid_standardized_splits
        .normal_fit
        .iloc[:1]
        .copy()
    )

    invalid_splits = replace(
        valid_standardized_splits,
        normal_fit=one_row_fit,
    )

    with pytest.raises(
        ValueError,
        match="at least two observations",
    ):
        fit_normal_pca(invalid_splits)


@pytest.mark.parametrize(
    (
        "source_partition",
        "target_partition",
        "expected_message",
    ),
    [
        (
            "normal_fit",
            "normal_calibration",
            "Fitting and calibration flow IDs overlap",
        ),
        (
            "normal_fit",
            "test",
            "Fitting and test flow IDs overlap",
        ),
        (
            "normal_calibration",
            "test",
            "Calibration and test flow IDs overlap",
        ),
    ],
)
def test_pca_fit_rejects_identifier_overlap(
    valid_standardized_splits: StandardizedDataSplits,
    source_partition: str,
    target_partition: str,
    expected_message: str,
) -> None:
    source_frame = getattr(
        valid_standardized_splits,
        source_partition,
    )

    invalid_target = getattr(
        valid_standardized_splits,
        target_partition,
    ).copy()

    target_identifiers = invalid_target.index.tolist()
    target_identifiers[0] = source_frame.index[0]

    invalid_target.index = pd.Index(
        target_identifiers,
        name="flow_id",
    )

    invalid_splits = replace(
        valid_standardized_splits,
        **{target_partition: invalid_target},
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        fit_normal_pca(invalid_splits)


def test_transform_rejects_incorrect_fit_result_type(
    valid_standardized_splits: StandardizedDataSplits,
) -> None:
    with pytest.raises(
        TypeError,
        match="fit_result must be a PCAFitResult",
    ):
        transform_pca_splits(
            valid_standardized_splits,
            object(),  # type: ignore[arg-type]
        )


def test_transform_rejects_unfitted_model(
    valid_standardized_splits: StandardizedDataSplits,
) -> None:
    valid_result = fit_normal_pca(
        valid_standardized_splits
    )

    invalid_result = replace(
        valid_result,
        model=ManualPCA(
            n_components=valid_result.n_components
        ),
    )

    with pytest.raises(
        ValueError,
        match="unfitted PCA model",
    ):
        transform_pca_splits(
            valid_standardized_splits,
            invalid_result,
        )


def test_transform_rejects_wrong_model_feature_count(
    valid_standardized_splits: StandardizedDataSplits,
) -> None:
    valid_result = fit_normal_pca(
        valid_standardized_splits
    )

    wrong_feature_model = ManualPCA(
        n_components=1
    )

    wrong_feature_model.fit(
        np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            dtype=np.float64,
        )
    )

    invalid_result = replace(
        valid_result,
        model=wrong_feature_model,
    )

    with pytest.raises(
        ValueError,
        match="feature count does not match",
    ):
        transform_pca_splits(
            valid_standardized_splits,
            invalid_result,
        )
