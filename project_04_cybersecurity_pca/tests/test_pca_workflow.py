"""Tests for normal-only PCA fitting and scoring."""

from __future__ import annotations

import numpy as np
import pytest
from cyber_pca.pca_workflow import select_n_components

from cyber_pca import (
    StandardizedDataSplits,
    generate_synthetic_network_data,
    split_normal_calibration_test,
    standardize_splits,
)
from cyber_pca.pca_workflow import (
    PCAFitResult,
    PCAScoreSplits,
    fit_normal_pca,
    select_n_components,
    transform_pca_splits,
)


@pytest.mark.parametrize(
    ("ratios", "target", "expected"),
    [
        ([0.50, 0.30, 0.15, 0.05], 0.50, 1),
        ([0.50, 0.30, 0.15, 0.05], 0.80, 2),
        ([0.50, 0.30, 0.15, 0.05], 0.81, 3),
        ([0.50, 0.30, 0.15, 0.05], 1.00, 4),
    ],
)
def test_selects_minimum_component_count(
    ratios: list[float],
    target: float,
    expected: int,
) -> None:
    assert select_n_components(
        ratios,
        explained_variance_target=target,
    ) == expected


def test_normal_pca_satisfies_variance_target() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=600,
        n_attack_per_type=100,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    standardized = standardize_splits(raw_splits)

    result = fit_normal_pca(
        standardized,
        explained_variance_target=0.95,
    )

    assert isinstance(result, PCAFitResult)
    assert 1 <= result.n_components <= 10

    assert (
        result.achieved_explained_variance
        >= 0.95
    )

    if result.n_components > 1:
        previous_cumulative = (
            result.full_cumulative_explained_variance[
                result.n_components - 2
            ]
        )

        assert previous_cumulative < 0.95

    assert result.model.n_components_ == (
        result.n_components
    )

    assert result.full_explained_variance.shape == (
        10,
    )

    assert (
        result.full_explained_variance_ratio.shape
        == (10,)
    )

    assert (
        result.full_cumulative_explained_variance.shape
        == (10,)
    )


def test_calibration_and_test_values_do_not_change_pca_fit() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=600,
        n_attack_per_type=100,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    standardized = standardize_splits(raw_splits)

    baseline = fit_normal_pca(standardized)

    altered = StandardizedDataSplits(
        normal_fit=standardized.normal_fit.copy(),
        normal_calibration=(
            standardized.normal_calibration
            + 1.0e9
        ),
        test=standardized.test + 1.0e12,
        scaler=standardized.scaler,
    )

    altered_result = fit_normal_pca(altered)

    np.testing.assert_array_equal(
        baseline.full_explained_variance,
        altered_result.full_explained_variance,
    )

    baseline_projection = (
        baseline.model.components_.T
        @ baseline.model.components_
    )

    altered_projection = (
        altered_result.model.components_.T
        @ altered_result.model.components_
    )

    np.testing.assert_allclose(
        baseline_projection,
        altered_projection,
        atol=1.0e-14,
    )


def test_pca_scores_preserve_shapes_and_identifiers() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    standardized = standardize_splits(raw_splits)

    fit_result = fit_normal_pca(
        standardized,
        explained_variance_target=0.95,
    )

    scores = transform_pca_splits(
        standardized,
        fit_result,
    )

    assert isinstance(scores, PCAScoreSplits)

    component_count = fit_result.n_components

    assert scores.normal_fit.shape == (
        72,
        component_count,
    )

    assert scores.normal_calibration.shape == (
        24,
        component_count,
    )

    assert scores.test.shape == (
        84,
        component_count,
    )

    expected_columns = tuple(
        f"PC{index}"
        for index in range(
            1,
            component_count + 1,
        )
    )

    assert tuple(
        scores.normal_fit.columns
    ) == expected_columns

    assert tuple(scores.test.columns) == (
        expected_columns
    )

    np.testing.assert_array_equal(
        scores.normal_fit.index.to_numpy(),
        standardized.normal_fit.index.to_numpy(),
    )

    np.testing.assert_array_equal(
        scores.test.index.to_numpy(),
        standardized.test.index.to_numpy(),
    )


def test_repeated_pca_fits_are_deterministic() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=300,
        n_attack_per_type=50,
        random_seed=42,
    )

    standardized = standardize_splits(
        split_normal_calibration_test(dataset)
    )

    first = fit_normal_pca(standardized)
    second = fit_normal_pca(standardized)

    assert first.n_components == second.n_components

    np.testing.assert_array_equal(
        first.full_explained_variance,
        second.full_explained_variance,
    )

    np.testing.assert_array_equal(
        first.model.components_,
        second.model.components_,
    )

def test_component_selection_uses_minimum_valid_count() -> None:
    variance_ratios = np.array(
        [0.50, 0.30, 0.15, 0.05],
        dtype=np.float64,
    )

    assert select_n_components(
        variance_ratios,
        explained_variance_target=0.49,
    ) == 1

    assert select_n_components(
        variance_ratios,
        explained_variance_target=0.79,
    ) == 2

    assert select_n_components(
        variance_ratios,
        explained_variance_target=0.81,
    ) == 3

    assert select_n_components(
        variance_ratios,
        explained_variance_target=1.00,
    ) == 4


@pytest.mark.parametrize(
    "invalid_target",
    [
        0.0,
        -0.01,
        1.0001,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_component_selection_rejects_invalid_numeric_targets(
    invalid_target: float,
) -> None:
    variance_ratios = np.array(
        [0.70, 0.20, 0.10],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        select_n_components(
            variance_ratios,
            explained_variance_target=invalid_target,
        )


@pytest.mark.parametrize(
    "invalid_target",
    [
        True,
        False,
        "0.95",
        None,
    ],
)
def test_component_selection_rejects_non_numeric_targets(
    invalid_target: object,
) -> None:
    variance_ratios = np.array(
        [0.70, 0.20, 0.10],
        dtype=np.float64,
    )

    with pytest.raises(TypeError):
        select_n_components(
            variance_ratios,
            explained_variance_target=invalid_target,
        )


@pytest.mark.parametrize(
    "invalid_ratios",
    [
        np.array([], dtype=np.float64),
        np.array([[0.70, 0.30]], dtype=np.float64),
        np.array([0.80, 0.30, -0.10], dtype=np.float64),
        np.array([0.70, np.nan, 0.30], dtype=np.float64),
        np.array([0.70, np.inf, 0.30], dtype=np.float64),
        np.array([0.40, 0.40], dtype=np.float64),
    ],
)
def test_component_selection_rejects_invalid_variance_ratios(
    invalid_ratios: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        select_n_components(
            invalid_ratios,
            explained_variance_target=0.95,
        )


def test_component_selection_does_not_mutate_variance_ratios() -> None:
    variance_ratios = np.array(
        [0.60, 0.25, 0.10, 0.05],
        dtype=np.float64,
    )
    original = variance_ratios.copy()

    selected = select_n_components(
        variance_ratios,
        explained_variance_target=0.95,
    )

    assert selected == 3
    np.testing.assert_array_equal(
        variance_ratios,
        original,
    )
