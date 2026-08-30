"""Transparent covariance-based principal component analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class ManualPCA:
    """Principal component analysis using covariance eigendecomposition.

    Parameters
    ----------
    n_components:
        Number of principal components to retain. ``None`` retains all
        feature directions.

    eigenvalue_tolerance:
        Relative tolerance for distinguishing harmless floating-point
        round-off from a materially negative covariance eigenvalue.
    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        eigenvalue_tolerance: float = 1.0e-12,
    ) -> None:
        """Initialize PCA configuration without fitting observations."""

        if n_components is not None and (
            isinstance(n_components, bool)
            or not isinstance(n_components, int)
        ):
            raise TypeError("n_components must be an integer or None")

        if (
            not np.isfinite(eigenvalue_tolerance)
            or eigenvalue_tolerance <= 0.0
        ):
            raise ValueError(
                "eigenvalue_tolerance must be finite and positive"
            )

        self.n_components = n_components
        self.eigenvalue_tolerance = float(eigenvalue_tolerance)

    @staticmethod
    def _validate_matrix(
        X: ArrayLike,
        *,
        min_samples: int = 2,
    ) -> FloatArray:
        """Convert input to a finite two-dimensional float64 matrix."""

        try:
            matrix = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "X must contain real numeric values"
            ) from exc

        if matrix.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix")

        if matrix.shape[0] < min_samples:
            raise ValueError(
                f"X must contain at least {min_samples} observations"
            )

        if matrix.shape[1] < 1:
            raise ValueError("X must contain at least one feature")

        if not np.all(np.isfinite(matrix)):
            raise ValueError("X must contain only finite values")

        return matrix

    @staticmethod
    def _orient_eigenvectors(
        eigenvectors: FloatArray,
    ) -> FloatArray:
        """Choose a deterministic sign for each eigenvector.

        An eigenvector and its negative represent the same direction.
        For reproducible output, the loading with the largest absolute
        magnitude is oriented to be positive.
        """

        oriented = eigenvectors.copy()

        for column_index in range(oriented.shape[1]):
            eigenvector = oriented[:, column_index]
            pivot_index = int(np.argmax(np.abs(eigenvector)))

            if eigenvector[pivot_index] < 0.0:
                oriented[:, column_index] *= -1.0

        return oriented

    def fit(self, X: ArrayLike) -> ManualPCA:
        """Fit PCA using the sample covariance eigendecomposition."""

        matrix = self._validate_matrix(X)
        n_samples, n_features = matrix.shape

        if self.n_components is not None and not (
            1 <= self.n_components <= n_features
        ):
            raise ValueError(
                "n_components must be between "
                f"1 and {n_features}, inclusive"
            )

        self.n_samples_seen_ = n_samples
        self.n_features_in_ = n_features
        self.n_components_ = (
            n_features
            if self.n_components is None
            else self.n_components
        )

        self.mean_ = np.mean(
            matrix,
            axis=0,
            dtype=np.float64,
        )

        centered = matrix - self.mean_

        self.covariance_ = (
            centered.T @ centered
        ) / np.float64(n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_)

        descending_order = np.argsort(eigenvalues)[::-1]

        eigenvalues = np.asarray(
            eigenvalues[descending_order],
            dtype=np.float64,
        )

        eigenvectors = np.asarray(
            eigenvectors[:, descending_order],
            dtype=np.float64,
        )

        spectral_scale = max(
            1.0,
            float(np.max(np.abs(eigenvalues))),
        )

        self.numerical_tolerance_ = (
            self.eigenvalue_tolerance
            * spectral_scale
            * max(1, n_features)
        )

        minimum_eigenvalue = float(np.min(eigenvalues))

        if minimum_eigenvalue < -self.numerical_tolerance_:
            raise np.linalg.LinAlgError(
                "covariance matrix has a materially negative eigenvalue"
            )

        eigenvalues = np.where(
            eigenvalues < 0.0,
            0.0,
            eigenvalues,
        )

        eigenvectors = self._orient_eigenvectors(eigenvectors)

        total_variance = float(np.sum(eigenvalues))

        if total_variance <= self.numerical_tolerance_:
            raise ValueError("X must contain positive total variance")

        full_variance_ratios = eigenvalues / total_variance

        self.all_explained_variance_ = eigenvalues
        self.all_explained_variance_ratio_ = full_variance_ratios
        self.all_components_ = eigenvectors.T

        self.explained_variance_ = eigenvalues[
            : self.n_components_
        ]

        self.explained_variance_ratio_ = full_variance_ratios[
            : self.n_components_
        ]

        self.cumulative_explained_variance_ = np.cumsum(
            self.explained_variance_ratio_
        )

        self.components_ = self.all_components_[
            : self.n_components_
        ]

        return self


    def _require_fitted(self) -> None:
        """Raise an informative error when PCA has not been fitted."""

        if not hasattr(self, "components_"):
            raise RuntimeError(
                "ManualPCA must be fitted before this operation"
            )

    def _validate_transform_input(
        self,
        X: ArrayLike,
    ) -> FloatArray:
        """Validate observations supplied after model fitting."""

        self._require_fitted()

        matrix = self._validate_matrix(
            X,
            min_samples=1,
        )

        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features; "
                f"expected {self.n_features_in_}"
            )

        return matrix

    def transform(self, X: ArrayLike) -> FloatArray:
        """Project observations onto retained principal components."""

        matrix = self._validate_transform_input(X)
        centered = matrix - self.mean_

        return centered @ self.components_.T

    def inverse_transform(
        self,
        scores: ArrayLike,
    ) -> FloatArray:
        """Map principal-component scores back to feature space."""

        self._require_fitted()

        score_matrix = self._validate_matrix(
            scores,
            min_samples=1,
        )

        if score_matrix.shape[1] != self.n_components_:
            raise ValueError(
                f"scores have {score_matrix.shape[1]} components; "
                f"expected {self.n_components_}"
            )

        return score_matrix @ self.components_ + self.mean_

    def reconstruct(self, X: ArrayLike) -> FloatArray:
        """Reconstruct observations from retained components."""

        scores = self.transform(X)

        return self.inverse_transform(scores)

    def reconstruction_error(
        self,
        X: ArrayLike,
    ) -> FloatArray:
        """Calculate mean squared reconstruction error per observation."""

        matrix = self._validate_transform_input(X)
        centered = matrix - self.mean_
        scores = centered @ self.components_.T
        reconstructed = scores @ self.components_ + self.mean_

        return np.mean(
            (matrix - reconstructed) ** 2,
            axis=1,
            dtype=np.float64,
        )

    def fit_transform(self, X: ArrayLike) -> FloatArray:
        """Fit PCA and return retained principal-component scores."""

        self.fit(X)

        return self.transform(X)
