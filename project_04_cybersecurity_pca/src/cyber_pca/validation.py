"""Independent mathematical validation for manual PCA."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from cyber_pca.pca_manual import FloatArray, ManualPCA


def _as_float_matrix(
    value: ArrayLike,
    name: str,
) -> FloatArray:
    """Convert a value to a finite two-dimensional float64 matrix."""

    try:
        matrix = np.asarray(
            value,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite two-dimensional matrix"
        ) from exc

    if (
        matrix.ndim != 2
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError(
            f"{name} must be a finite two-dimensional matrix"
        )

    return matrix


def validate_covariance_symmetry(
    covariance: ArrayLike,
    *,
    atol: float = 1.0e-12,
) -> bool:
    """Return whether a covariance matrix is square and symmetric."""

    matrix = _as_float_matrix(
        covariance,
        "covariance",
    )

    return bool(
        matrix.shape[0] == matrix.shape[1]
        and np.allclose(
            matrix,
            matrix.T,
            atol=atol,
            rtol=0.0,
        )
    )


def validate_eigendecomposition(
    covariance: ArrayLike,
    eigenvalues: ArrayLike,
    components: ArrayLike,
    *,
    atol: float = 1.0e-10,
) -> bool:
    """Check the equation C v = lambda v for component rows."""

    matrix = _as_float_matrix(
        covariance,
        "covariance",
    )

    component_rows = _as_float_matrix(
        components,
        "components",
    )

    vectors = component_rows.T

    try:
        values = np.asarray(
            eigenvalues,
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return False

    if (
        values.ndim != 1
        or not np.all(np.isfinite(values))
        or values.size != vectors.shape[1]
    ):
        return False

    if (
        matrix.shape[0] != matrix.shape[1]
        or matrix.shape[1] != vectors.shape[0]
    ):
        return False

    left_side = matrix @ vectors
    right_side = (
        vectors * values[np.newaxis, :]
    )

    return bool(
        np.allclose(
            left_side,
            right_side,
            atol=atol,
            rtol=1.0e-10,
        )
    )


def validate_orthonormality(
    components: ArrayLike,
    *,
    atol: float = 1.0e-10,
) -> bool:
    """Check that component rows form an orthonormal set."""

    vectors = _as_float_matrix(
        components,
        "components",
    )

    if vectors.shape[0] < 1:
        return False

    identity = np.eye(
        vectors.shape[0],
        dtype=np.float64,
    )

    return bool(
        np.allclose(
            vectors @ vectors.T,
            identity,
            atol=atol,
            rtol=0.0,
        )
    )


def validate_nonnegative_eigenvalues(
    eigenvalues: ArrayLike,
    *,
    tolerance: float = 1.0e-12,
) -> bool:
    """Allow negative eigenvalues only within numerical tolerance."""

    try:
        values = np.asarray(
            eigenvalues,
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return False

    return bool(
        values.ndim == 1
        and values.size > 0
        and np.all(np.isfinite(values))
        and np.all(values >= -tolerance)
    )


def validate_explained_variance(
    ratios: ArrayLike,
    cumulative: ArrayLike,
    *,
    atol: float = 1.0e-12,
) -> bool:
    """Validate full variance ratios and cumulative monotonicity."""

    try:
        ratio_values = np.asarray(
            ratios,
            dtype=np.float64,
        )

        cumulative_values = np.asarray(
            cumulative,
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return False

    if (
        ratio_values.ndim != 1
        or cumulative_values.ndim != 1
        or ratio_values.shape != cumulative_values.shape
        or ratio_values.size == 0
        or not np.all(np.isfinite(ratio_values))
        or not np.all(np.isfinite(cumulative_values))
    ):
        return False

    return bool(
        np.all(ratio_values >= -atol)
        and np.isclose(
            ratio_values.sum(),
            1.0,
            atol=atol,
        )
        and np.all(
            np.diff(cumulative_values) >= -atol
        )
        and np.allclose(
            cumulative_values,
            np.cumsum(ratio_values),
            atol=atol,
            rtol=0.0,
        )
    )


def validate_full_reconstruction(
    original: ArrayLike,
    reconstructed: ArrayLike,
    *,
    atol: float = 1.0e-10,
) -> bool:
    """Check full-component reconstruction against original input."""

    source = _as_float_matrix(
        original,
        "original",
    )

    recovered = _as_float_matrix(
        reconstructed,
        "reconstructed",
    )

    return bool(
        source.shape == recovered.shape
        and np.allclose(
            source,
            recovered,
            atol=atol,
            rtol=1.0e-10,
        )
    )


def run_math_validation(
    X: ArrayLike,
) -> dict[str, Any]:
    """Fit full PCA and return a JSON-serializable validation report."""

    matrix = ManualPCA._validate_matrix(
        X,
        min_samples=2,
    )

    model = ManualPCA().fit(matrix)
    reconstructed = model.reconstruct(matrix)

    full_cumulative_variance = np.cumsum(
        model.all_explained_variance_ratio_
    )

    checks = {
        "covariance_symmetric": (
            validate_covariance_symmetry(
                model.covariance_
            )
        ),
        "eigenvalues_descending": bool(
            np.all(
                np.diff(
                    model.all_explained_variance_
                )
                <= model.numerical_tolerance_
            )
        ),
        "eigenpair_consistency": (
            validate_eigendecomposition(
                model.covariance_,
                model.all_explained_variance_,
                model.all_components_,
            )
        ),
        "eigenvectors_orthonormal": (
            validate_orthonormality(
                model.all_components_
            )
        ),
        "eigenvalues_nonnegative": (
            validate_nonnegative_eigenvalues(
                model.all_explained_variance_,
                tolerance=model.numerical_tolerance_,
            )
        ),
        "explained_variance_valid": (
            validate_explained_variance(
                model.all_explained_variance_ratio_,
                full_cumulative_variance,
            )
        ),
        "full_reconstruction_exact": (
            validate_full_reconstruction(
                matrix,
                reconstructed,
            )
        ),
    }

    status = (
        "passed"
        if all(checks.values())
        else "failed"
    )

    return {
        "status": status,
        "checks": checks,
        "dimensions": {
            "observations": int(matrix.shape[0]),
            "features": int(matrix.shape[1]),
        },
        "numerics": {
            "dtype": str(model.covariance_.dtype),
            "eigenvalue_tolerance": float(
                model.numerical_tolerance_
            ),
            "maximum_absolute_reconstruction_error": float(
                np.max(
                    np.abs(
                        matrix - reconstructed
                    )
                )
            ),
            "explained_variance_ratio_sum": float(
                np.sum(
                    model.all_explained_variance_ratio_
                )
            ),
        },
        "eigenvalues": (
            model.all_explained_variance_.tolist()
        ),
        "explained_variance_ratios": (
            model.all_explained_variance_ratio_.tolist()
        ),
    }
