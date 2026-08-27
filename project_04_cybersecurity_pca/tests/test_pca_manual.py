"""Tests for the transparent covariance-based PCA implementation."""

from __future__ import annotations

import numpy as np
import pytest

from cyber_pca.pca_manual import ManualPCA

from sklearn.decomposition import PCA


@pytest.fixture
def sample_matrix() -> np.ndarray:
    """Return a deterministic matrix with nonzero feature variance."""

    return np.asarray(
        [
            [2.0, 0.5, 1.1, 3.2],
            [2.4, 0.7, 0.9, 3.8],
            [3.1, 1.2, 1.5, 4.1],
            [3.8, 1.0, 1.8, 4.9],
            [4.2, 1.8, 2.2, 5.1],
            [5.0, 2.1, 2.0, 6.3],
        ],
        dtype=np.float64,
    )


def test_fit_rejects_one_dimensional_input() -> None:
    model = ManualPCA()

    with pytest.raises(ValueError, match="two-dimensional"):
        model.fit(np.asarray([1.0, 2.0, 3.0]))


def test_fit_requires_at_least_two_observations() -> None:
    model = ManualPCA()

    with pytest.raises(ValueError, match="at least 2 observations"):
        model.fit(np.asarray([[1.0, 2.0, 3.0]]))


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_fit_rejects_nonfinite_values(invalid_value: float) -> None:
    matrix = np.asarray(
        [
            [1.0, 2.0],
            [3.0, invalid_value],
        ]
    )

    with pytest.raises(ValueError, match="finite"):
        ManualPCA().fit(matrix)


def test_fit_rejects_nonnumeric_values() -> None:
    matrix = [
        ["normal", "tcp"],
        ["attack", "udp"],
    ]

    with pytest.raises(ValueError, match="numeric"):
        ManualPCA().fit(matrix)


@pytest.mark.parametrize("invalid_components", [1.5, "2", True])
def test_constructor_rejects_noninteger_components(
    invalid_components: object,
) -> None:
    with pytest.raises(TypeError, match="integer or None"):
        ManualPCA(n_components=invalid_components)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_components", [0, -1, 5])
def test_fit_rejects_component_count_outside_feature_range(
    sample_matrix: np.ndarray,
    invalid_components: int,
) -> None:
    model = ManualPCA(n_components=invalid_components)

    with pytest.raises(ValueError, match="between 1 and 4"):
        model.fit(sample_matrix)


def test_fit_calculates_feature_means(sample_matrix: np.ndarray) -> None:
    model = ManualPCA(n_components=2).fit(sample_matrix)

    expected_means = np.mean(sample_matrix, axis=0)

    np.testing.assert_allclose(model.mean_, expected_means, atol=1e-12)


def test_fit_calculates_sample_covariance(sample_matrix: np.ndarray) -> None:
    model = ManualPCA(n_components=2).fit(sample_matrix)

    centered = sample_matrix - np.mean(sample_matrix, axis=0)
    expected_covariance = centered.T @ centered / (sample_matrix.shape[0] - 1)

    np.testing.assert_allclose(
        model.covariance_,
        expected_covariance,
        atol=1e-12,
    )


def test_covariance_is_symmetric(sample_matrix: np.ndarray) -> None:
    model = ManualPCA(n_components=2).fit(sample_matrix)

    np.testing.assert_allclose(
        model.covariance_,
        model.covariance_.T,
        atol=1e-12,
    )


def test_covariance_uses_float64(sample_matrix: np.ndarray) -> None:
    integer_matrix = sample_matrix.astype(np.int64)

    model = ManualPCA(n_components=2).fit(integer_matrix)

    assert model.covariance_.dtype == np.float64

def test_eigenvalues_are_sorted_descending(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA().fit(sample_matrix)

    assert np.all(np.diff(model.explained_variance_) <= 0.0)


def test_eigenpairs_satisfy_covariance_equation(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA().fit(sample_matrix)

    eigenvectors = model.components_.T

    np.testing.assert_allclose(
        model.covariance_ @ eigenvectors,
        eigenvectors * model.explained_variance_[np.newaxis, :],
        atol=1e-10,
    )


def test_component_vectors_are_orthonormal(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA().fit(sample_matrix)

    np.testing.assert_allclose(
        model.components_ @ model.components_.T,
        np.eye(sample_matrix.shape[1]),
        atol=1e-10,
    )


def test_eigenvalues_are_nonnegative(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA().fit(sample_matrix)

    assert np.all(model.explained_variance_ >= 0.0)
    assert np.isrealobj(model.explained_variance_)
    assert np.isrealobj(model.components_)


def test_full_explained_variance_ratios_sum_to_one(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA().fit(sample_matrix)

    assert model.explained_variance_ratio_.sum() == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_cumulative_explained_variance_is_monotonic(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA().fit(sample_matrix)

    expected = np.cumsum(model.explained_variance_ratio_)

    np.testing.assert_allclose(
        model.cumulative_explained_variance_,
        expected,
        atol=1e-12,
    )

    assert np.all(
        np.diff(model.cumulative_explained_variance_) >= 0.0
    )

    assert model.cumulative_explained_variance_[-1] == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_retained_component_shapes(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA(n_components=2).fit(sample_matrix)

    assert model.n_components_ == 2
    assert model.components_.shape == (2, sample_matrix.shape[1])
    assert model.explained_variance_.shape == (2,)
    assert model.explained_variance_ratio_.shape == (2,)
    assert model.cumulative_explained_variance_.shape == (2,)

def test_transform_matches_matrix_projection(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA(n_components=2).fit(sample_matrix)

    scores = model.transform(sample_matrix)

    expected_scores = (
        sample_matrix - model.mean_
    ) @ model.components_.T

    assert scores.shape == (sample_matrix.shape[0], 2)

    np.testing.assert_allclose(
        scores,
        expected_scores,
        atol=1e-12,
    )


def test_inverse_transform_matches_matrix_formula(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA(n_components=2).fit(sample_matrix)
    scores = model.transform(sample_matrix)

    reconstructed = model.inverse_transform(scores)
    expected = scores @ model.components_ + model.mean_

    np.testing.assert_allclose(
        reconstructed,
        expected,
        atol=1e-12,
    )


def test_full_component_reconstruction_is_exact(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA().fit(sample_matrix)

    reconstructed = model.reconstruct(sample_matrix)

    np.testing.assert_allclose(
        reconstructed,
        sample_matrix,
        atol=1e-10,
    )


def test_reconstruction_error_is_observation_mse(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA(n_components=2).fit(sample_matrix)

    reconstructed = model.reconstruct(sample_matrix)
    errors = model.reconstruction_error(sample_matrix)

    expected_errors = np.mean(
        (sample_matrix - reconstructed) ** 2,
        axis=1,
    )

    assert errors.shape == (sample_matrix.shape[0],)
    assert np.all(errors >= 0.0)

    np.testing.assert_allclose(
        errors,
        expected_errors,
        atol=1e-12,
    )


def test_aggregate_reconstruction_error_does_not_increase(
    sample_matrix: np.ndarray,
) -> None:
    mean_errors = []

    for component_count in range(
        1,
        sample_matrix.shape[1] + 1,
    ):
        model = ManualPCA(
            n_components=component_count
        ).fit(sample_matrix)

        mean_errors.append(
            float(
                np.mean(
                    model.reconstruction_error(sample_matrix)
                )
            )
        )

    assert np.all(np.diff(mean_errors) <= 1e-14)


def test_fit_transform_matches_separate_operations(
    sample_matrix: np.ndarray,
) -> None:
    combined_model = ManualPCA(n_components=2)
    combined_scores = combined_model.fit_transform(sample_matrix)

    separate_model = ManualPCA(n_components=2).fit(sample_matrix)
    separate_scores = separate_model.transform(sample_matrix)

    np.testing.assert_allclose(
        combined_scores,
        separate_scores,
        atol=1e-12,
    )


def test_transform_requires_fitted_model(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA(n_components=2)

    with pytest.raises(RuntimeError, match="fitted"):
        model.transform(sample_matrix)


def test_transform_rejects_wrong_feature_count(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA(n_components=2).fit(sample_matrix)

    with pytest.raises(ValueError, match="expected 4"):
        model.transform(sample_matrix[:, :3])


def test_inverse_transform_rejects_wrong_component_count(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA(n_components=2).fit(sample_matrix)
    invalid_scores = np.ones((3, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="expected 2"):
        model.inverse_transform(invalid_scores)


def test_manual_pca_matches_sklearn_principal_subspace(
    sample_matrix: np.ndarray,
) -> None:
    component_count = 2

    manual = ManualPCA(
        n_components=component_count
    ).fit(sample_matrix)

    reference = PCA(
        n_components=component_count,
        svd_solver="full",
    ).fit(sample_matrix)

    manual_projection = (
        manual.components_.T @ manual.components_
    )

    reference_projection = (
        reference.components_.T @ reference.components_
    )

    np.testing.assert_allclose(
        manual_projection,
        reference_projection,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        manual.explained_variance_,
        reference.explained_variance_,
        atol=1e-10,
    )


def test_near_collinear_input_remains_numerically_stable() -> None:
    base = np.linspace(-2.0, 2.0, 30)

    matrix = np.column_stack(
        [
            base,
            2.0 * base + 1e-11 * np.sin(base),
            np.cos(base),
        ]
    )

    model = ManualPCA().fit(matrix)
    eigenvectors = model.components_.T

    eigenpair_residual = np.max(
        np.abs(
            model.covariance_ @ eigenvectors
            - eigenvectors
            * model.explained_variance_[np.newaxis, :]
        )
    )

    assert model.covariance_.dtype == np.float64
    assert np.all(np.isfinite(model.explained_variance_))
    assert np.all(model.explained_variance_ >= 0.0)
    assert np.isrealobj(model.components_)
    assert eigenpair_residual < 1e-9


def test_repeated_fits_are_deterministic(
    sample_matrix: np.ndarray,
) -> None:
    first = ManualPCA(n_components=3).fit(sample_matrix)
    second = ManualPCA(n_components=3).fit(
        sample_matrix.copy()
    )

    np.testing.assert_array_equal(
        first.components_,
        second.components_,
    )

    np.testing.assert_array_equal(
        first.explained_variance_,
        second.explained_variance_,
    )


@pytest.mark.parametrize(
    "invalid_tolerance",
    [0.0, -1.0, np.inf, np.nan],
)
def test_constructor_rejects_invalid_eigenvalue_tolerance(
    invalid_tolerance: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        ManualPCA(
            eigenvalue_tolerance=invalid_tolerance
        )


def test_zero_variance_input_is_rejected() -> None:
    constant_matrix = np.ones(
        (5, 3),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="positive total variance",
    ):
        ManualPCA().fit(constant_matrix)


def test_transform_accepts_single_observation(
    sample_matrix: np.ndarray,
) -> None:
    model = ManualPCA(n_components=2).fit(sample_matrix)

    scores = model.transform(sample_matrix[:1])

    assert scores.shape == (1, 2)
    assert np.all(np.isfinite(scores))