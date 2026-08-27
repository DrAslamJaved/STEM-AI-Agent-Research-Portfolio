"""Cybersecurity anomaly detection using PCA and eigenvalue analysis."""

from cyber_pca.pca_manual import ManualPCA
from cyber_pca.validation import run_math_validation


__version__ = "0.1.0"

__all__ = [
    "ManualPCA",
    "run_math_validation",
    "__version__",
]
