"""Cybersecurity anomaly detection using PCA and eigenvalue analysis."""

from cyber_pca.pca_manual import ManualPCA
from cyber_pca.validation import run_math_validation

from cyber_pca.synthetic_data import (
    ATTACK_TYPES,
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    generate_synthetic_network_data,
)

from cyber_pca.preprocessing import (
    RawDataSplits,
    StandardizedDataSplits,
    split_normal_calibration_test,
    standardize_splits,
)

from cyber_pca.detector import (
    AnomalyThresholdResult,
    ReconstructionErrorSplits,
    calibrate_anomaly_threshold,
    compute_reconstruction_errors,
    predict_anomalies,
)

from .pca_workflow import (
    PCAFitResult,
    PCAScoreSplits,
    fit_normal_pca,
    select_n_components,
    transform_pca_splits,
)

from .detector import (
    AnomalyThresholdResult,
    ReconstructionErrorSplits,
    calibrate_anomaly_threshold,
    compute_reconstruction_errors,
    predict_anomalies,
)

from .evaluation import (
    BinaryEvaluationResult,
    align_evaluation_data,
    evaluate_binary_predictions,
    evaluate_scenarios,
)

from .reporting import (
    SyntheticEvaluationArtifacts,
    build_synthetic_evaluation_summary,
    resolve_synthetic_evaluation_artifacts,
    write_synthetic_evaluation_artifacts,
)

__version__ = "0.1.0"

__all__ = [
    "ManualPCA",
    "run_math_validation",
    "__version__",
    "ATTACK_TYPES",
    "FEATURE_COLUMNS",
    "OUTPUT_COLUMNS",
    "generate_synthetic_network_data",
    "RawDataSplits",
    "StandardizedDataSplits",
    "split_normal_calibration_test",
    "standardize_splits",
    "AnomalyThresholdResult",
    "ReconstructionErrorSplits",
    "calibrate_anomaly_threshold",
    "compute_reconstruction_errors",
    "predict_anomalies",
    "BinaryEvaluationResult",
    "align_evaluation_data",
    "evaluate_binary_predictions",
    "evaluate_scenarios",
    "SyntheticEvaluationArtifacts",
    "build_synthetic_evaluation_summary",
    "resolve_synthetic_evaluation_artifacts",
    "write_synthetic_evaluation_artifacts",
]
