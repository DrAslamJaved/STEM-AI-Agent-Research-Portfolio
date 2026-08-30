"""Cybersecurity anomaly detection using PCA and eigenvalue analysis."""

from cyber_pca.detector import (
    AnomalyThresholdResult,
    ReconstructionErrorSplits,
    calibrate_anomaly_threshold,
    compute_reconstruction_errors,
    predict_anomalies,
)
from cyber_pca.evaluation import (
    BinaryEvaluationResult,
    align_evaluation_data,
    evaluate_binary_predictions,
    evaluate_scenarios,
)
from cyber_pca.pca_manual import ManualPCA
from cyber_pca.pca_workflow import (
    PCAFitResult,
    PCAScoreSplits,
    fit_normal_pca,
    select_n_components,
    transform_pca_splits,
)
from cyber_pca.preprocessing import (
    RawDataSplits,
    StandardizedDataSplits,
    split_normal_calibration_test,
    standardize_splits,
)
from cyber_pca.reporting import (
    SyntheticEvaluationArtifacts,
    build_synthetic_evaluation_summary,
    resolve_synthetic_evaluation_artifacts,
    write_synthetic_evaluation_artifacts,
)
from cyber_pca.synthetic_data import (
    ATTACK_TYPES,
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    generate_synthetic_network_data,
)
from cyber_pca.unsw_data import (
    UNSWNB15Data,
    UNSWNB15Paths,
    UNSW_CATEGORICAL_COLUMNS,
    UNSW_CURATED_COLUMNS,
    build_unsw_nb15_manifest,
    load_unsw_nb15,
    resolve_unsw_nb15_paths,
    validate_unsw_nb15,
    write_unsw_nb15_manifest,
)
from cyber_pca.unsw_preprocessing import (
    UNSWPreprocessor,
    UNSWRawDataSplits,
    UNSWStandardizedDataSplits,
    build_unsw_preprocessing_evidence,
    split_unsw_normal_calibration_test,
    standardize_unsw_splits,
    write_unsw_preprocessing_evidence,
)
from cyber_pca.validation import (
    run_math_validation,
)


__version__ = "0.1.0"


__all__ = [
    "ATTACK_TYPES",
    "AnomalyThresholdResult",
    "BinaryEvaluationResult",
    "FEATURE_COLUMNS",
    "ManualPCA",
    "OUTPUT_COLUMNS",
    "PCAFitResult",
    "PCAScoreSplits",
    "RawDataSplits",
    "ReconstructionErrorSplits",
    "StandardizedDataSplits",
    "SyntheticEvaluationArtifacts",
    "UNSWNB15Data",
    "UNSWNB15Paths",
    "UNSWPreprocessor",
    "UNSWRawDataSplits",
    "UNSWStandardizedDataSplits",
    "UNSW_CATEGORICAL_COLUMNS",
    "UNSW_CURATED_COLUMNS",
    "__version__",
    "align_evaluation_data",
    "build_synthetic_evaluation_summary",
    "build_unsw_nb15_manifest",
    "build_unsw_preprocessing_evidence",
    "calibrate_anomaly_threshold",
    "compute_reconstruction_errors",
    "evaluate_binary_predictions",
    "evaluate_scenarios",
    "fit_normal_pca",
    "generate_synthetic_network_data",
    "load_unsw_nb15",
    "predict_anomalies",
    "resolve_synthetic_evaluation_artifacts",
    "resolve_unsw_nb15_paths",
    "run_math_validation",
    "select_n_components",
    "split_normal_calibration_test",
    "split_unsw_normal_calibration_test",
    "standardize_splits",
    "standardize_unsw_splits",
    "transform_pca_splits",
    "validate_unsw_nb15",
    "write_synthetic_evaluation_artifacts",
    "write_unsw_nb15_manifest",
    "write_unsw_preprocessing_evidence",
]

from cyber_pca.unsw_evaluation import (
    align_unsw_evaluation_data,
    evaluate_unsw_attack_categories,
)
from cyber_pca.unsw_experiment import (
    UNSWDetectionResult,
    compute_unsw_reconstruction_errors,
    fit_unsw_normal_pca,
    run_unsw_detection,
)
from cyber_pca.unsw_reporting import (
    UNSWEvaluationArtifacts,
    build_unsw_evaluation_summary,
    resolve_unsw_evaluation_artifacts,
    write_unsw_evaluation_artifacts,
)
