"""Tests for independent PCA mathematical validators."""

from __future__ import annotations

import numpy as np
import pytest

from cyber_pca.validation import (
    run_math_validation,
    validate_covariance_symmetry,
    validate_eigendecomposition,
    validate_explained_variance,
    validate_full_reconstruction,
    validate_nonnegative_eigenvalues,
    validate_orthonormality,
)


def test_run_math_validation_returns_passing_report() -> None:
    matrix = np.asarray(
        [
            [1.0, 2.0, 0.0],
            [2.0, 2.5, 1.0],
            [4.0, 3.0, 1.5],
            [5.0, 5.0, 2.0],
        ],
        dtype=np.float64,
    )

    report = run_math_validation(matrix)

    assert report["status"] == "passed"
    assert all(report["checks"].values())

    assert report["dimensions"] == {
        "observations": 4,
        "features": 3,
    }

    assert report["numerics"]["dtype"] == "float64"

    assert (
        report["numerics"][
            "maximum_absolute_reconstruction_error"
        ]
        < 1e-10
    )

    assert report["numerics"][
        "explained_variance_ratio_sum"
    ] == pytest.approx(1.0, abs=1e-12)


def test_covariance_symmetry_validator() -> None:
    symmetric = np.asarray(
        [
            [2.0, 0.5],
            [0.5, 1.0],
        ]
    )

    nonsymmetric = np.asarray(
        [
            [2.0, 0.5],
            [0.0, 1.0],
        ]
    )

    nonsquare = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    assert validate_covariance_symmetry(symmetric)
    assert not validate_covariance_symmetry(nonsymmetric)
    assert not validate_covariance_symmetry(nonsquare)

    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        validate_covariance_symmetry([1.0, 2.0])


def test_eigendecomposition_validator_detects_inconsistency() -> None:
    covariance = np.diag([3.0, 1.0])
    components = np.eye(2)

    assert validate_eigendecomposition(
        covariance,
        [3.0, 1.0],
        components,
    )

    assert not validate_eigendecomposition(
        covariance,
        [2.0, 1.0],
        components,
    )

    assert not validate_eigendecomposition(
        covariance,
        [3.0],
        components,
    )

    assert not validate_eigendecomposition(
        np.eye(3),
        [3.0, 1.0],
        components,
    )


def test_orthonormality_validator_detects_invalid_basis() -> None:
    assert validate_orthonormality(np.eye(3))

    invalid_basis = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    assert not validate_orthonormality(invalid_basis)


def test_nonnegative_eigenvalue_validator() -> None:
    assert validate_nonnegative_eigenvalues(
        [2.0, -1e-13],
        tolerance=1e-12,
    )

    assert not validate_nonnegative_eigenvalues(
        [2.0, -1e-3],
        tolerance=1e-12,
    )

    assert not validate_nonnegative_eigenvalues([])
    assert not validate_nonnegative_eigenvalues([[1.0]])
    assert not validate_nonnegative_eigenvalues([1.0, np.nan])


def test_explained_variance_validator() -> None:
    assert validate_explained_variance(
        [0.6, 0.4],
        [0.6, 1.0],
    )

    assert not validate_explained_variance(
        [0.6, 0.3],
        [0.6, 0.9],
    )

    assert not validate_explained_variance(
        [0.5, -0.5, 1.0],
        [0.5, 0.0, 1.0],
    )

    assert not validate_explained_variance(
        [0.6, 0.4],
        [0.6],
    )

    assert not validate_explained_variance([], [])


def test_full_reconstruction_validator() -> None:
    original = np.eye(2)

    assert validate_full_reconstruction(
        original,
        original.copy(),
    )

    assert not validate_full_reconstruction(
        original,
        np.zeros((2, 2)),
    )

    assert not validate_full_reconstruction(
        original,
        np.zeros((2, 3)),
    )
