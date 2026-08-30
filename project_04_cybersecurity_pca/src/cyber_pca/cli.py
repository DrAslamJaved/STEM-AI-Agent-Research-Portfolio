"""Command-line interface for Project 4 workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from cyber_pca.detector import (
    calibrate_anomaly_threshold,
    compute_reconstruction_errors,
    predict_anomalies,
)
from cyber_pca.evaluation import (
    align_evaluation_data,
    evaluate_binary_predictions,
    evaluate_scenarios,
)
from cyber_pca.pca_workflow import fit_normal_pca
from cyber_pca.preprocessing import (
    split_normal_calibration_test,
    standardize_splits,
)
from cyber_pca.reporting import (
    resolve_synthetic_evaluation_artifacts,
    write_synthetic_evaluation_artifacts,
)
from cyber_pca.synthetic_data import (
    generate_synthetic_network_data,
)

from cyber_pca.validation import run_math_validation


from cyber_pca.unsw_data import (
    load_unsw_nb15,
)
from cyber_pca.unsw_evaluation import (
    align_unsw_evaluation_data,
    evaluate_unsw_attack_categories,
)
from cyber_pca.unsw_experiment import (
    run_unsw_detection,
)
from cyber_pca.unsw_preprocessing import (
    split_unsw_normal_calibration_test,
    standardize_unsw_splits,
)
from cyber_pca.unsw_reporting import (
    resolve_unsw_evaluation_artifacts,
    write_unsw_evaluation_artifacts,
)


DEFAULT_VALIDATION_OUTPUT = Path(
    "reports/validation/math_validation.json"
)


def deterministic_validation_matrix() -> np.ndarray:
    """Return a deterministic matrix for mathematical validation."""

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


def build_parser() -> argparse.ArgumentParser:
    """Build the Project 4 argument parser."""

    parser = argparse.ArgumentParser(
        prog="cyber-pca",
        description=(
            "Validated PCA mathematics for agentic "
            "cybersecurity anomaly detection."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    validate_parser = subparsers.add_parser(
        "validate-math",
        help=(
            "execute the Phase 1 mathematical "
            "validation contract"
        ),
    )

    validate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "print ordered operations without "
            "writing validation artifacts"
        ),
    )

    validate_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VALIDATION_OUTPUT,
        help=(
            "JSON output path "
            f"(default: {DEFAULT_VALIDATION_OUTPUT})"
        ),
    )

    evaluation_parser = subparsers.add_parser(
        "evaluate-synthetic",
        help=(
            "execute the frozen synthetic PCA "
            "evaluation workflow"
        ),
    )

    evaluation_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "print ordered operations without "
            "writing evaluation artifacts"
        ),
    )

    evaluation_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help=(
            "root directory for results and "
            "reports (default: current directory)"
        ),
    )

    evaluation_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help=(
            "PNG resolution in dots per inch "
            "(default: 150)"
        ),
    )

    unsw_parser = subparsers.add_parser(
        "evaluate-unsw",
        help=(
            "evaluate the frozen detector on the "
            "official UNSW-NB15 test partition"
        ),
    )

    unsw_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "print ordered operations without "
            "reading data or writing artifacts"
        ),
    )

    unsw_parser.add_argument(
        "--raw-directory",
        type=Path,
        default=Path("data/raw"),
        help=(
            "directory containing the official "
            "UNSW-NB15 CSV files "
            "(default: data/raw)"
        ),
    )

    unsw_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help=(
            "root directory for results and "
            "reports (default: current directory)"
        ),
    )

    unsw_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help=(
            "PNG resolution in dots per inch "
            "(default: 150)"
        ),
    )

    return parser


def _print_dry_run(output: Path) -> None:
    """Print the validation plan without writing files."""

    operations = (
        "load the deterministic float64 validation matrix",
        "center observations using feature means",
        (
            "compute C = X_centered.T @ "
            "X_centered / (n - 1)"
        ),
        (
            "solve the symmetric eigenproblem "
            "with numpy.linalg.eigh"
        ),
        (
            "sort eigenpairs in descending "
            "eigenvalue order"
        ),
        (
            "validate symmetry, eigenpairs, "
            "orthonormality, and explained variance"
        ),
        "validate full-component reconstruction",
        f"write the validation report to {output}",
    )

    print("DRY RUN — no files will be written")

    for index, operation in enumerate(
        operations,
        start=1,
    ):
        print(f"{index}. {operation}")


def _execute_validation(output: Path) -> int:
    """Execute mathematical validation and write JSON evidence."""

    report = run_math_validation(
        deterministic_validation_matrix()
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "Mathematical validation: "
        f"{report['status'].upper()}"
    )

    for check_name, passed in report["checks"].items():
        result = "PASS" if passed else "FAIL"
        print(f"- {check_name}: {result}")

    print(f"Validation report: {output}")

    return (
        0
        if report["status"] == "passed"
        else 1
    )

def _print_synthetic_evaluation_dry_run(
    output_root: Path,
    dpi: int,
) -> None:
    """Print the synthetic evaluation plan."""

    artifacts = (
        resolve_synthetic_evaluation_artifacts(
            output_root
        )
    )

    operations = (
        "generate deterministic synthetic traffic",
        (
            "create normal fitting, normal "
            "calibration, and hidden-label test splits"
        ),
        (
            "fit StandardScaler using normal "
            "fitting traffic only"
        ),
        (
            "fit PCA using normal fitting "
            "traffic only"
        ),
        (
            "select the minimum component count "
            "meeting the 0.95 variance target"
        ),
        (
            "compute standardized-space "
            "reconstruction errors"
        ),
        (
            "calibrate the 0.99 quantile threshold "
            "using normal calibration traffic only"
        ),
        (
            "predict test anomalies before "
            "accessing test labels"
        ),
        (
            "align hidden labels by flow_id and "
            "calculate binary and scenario metrics"
        ),
        (
            "write deterministic JSON, CSV, and "
            f"PNG artifacts at {dpi} DPI"
        ),
    )

    print("DRY RUN - no files will be written")

    for index, operation in enumerate(
        operations,
        start=1,
    ):
        print(f"{index}. {operation}")

    print("Planned artifacts:")

    planned_paths = (
        artifacts.summary_json,
        artifacts.predictions_csv,
        artifacts.metrics_csv,
        artifacts.scenario_metrics_csv,
        artifacts.confusion_matrix_figure,
        artifacts.reconstruction_errors_figure,
        artifacts.scree_plot_figure,
        artifacts.scenario_rates_figure,
    )

    for path in planned_paths:
        print(f"- {path}")


def _execute_synthetic_evaluation(
    output_root: Path,
    dpi: int,
) -> int:
    """Execute the frozen synthetic evaluation."""

    dataset = generate_synthetic_network_data()

    raw_splits = split_normal_calibration_test(
        dataset
    )

    standardized_splits = standardize_splits(
        raw_splits
    )

    fit_result = fit_normal_pca(
        standardized_splits
    )

    error_splits = compute_reconstruction_errors(
        standardized_splits,
        fit_result,
    )

    threshold_result = (
        calibrate_anomaly_threshold(
            error_splits
        )
    )

    # Predictions are frozen before test labels
    # enter evaluation.
    predictions = predict_anomalies(
        error_splits.test,
        threshold_result,
    )

    evaluation_data = align_evaluation_data(
        raw_splits.test,
        error_splits.test,
        predictions,
    )

    binary_result = (
        evaluate_binary_predictions(
            evaluation_data
        )
    )

    scenario_result = evaluate_scenarios(
        evaluation_data
    )

    artifacts = (
        write_synthetic_evaluation_artifacts(
            evaluation_data,
            fit_result,
            threshold_result,
            binary_result,
            scenario_result,
            output_root=output_root,
            dpi=dpi,
        )
    )

    print("Synthetic evaluation: PASSED")
    print(
        "- selected components: "
        f"{fit_result.n_components}"
    )
    print(
        "- achieved explained variance: "
        f"{fit_result.achieved_explained_variance:.17g}"
    )
    print(
        "- frozen threshold: "
        f"{threshold_result.threshold:.17g}"
    )
    print(
        "- confusion matrix: "
        f"{binary_result.confusion_matrix}"
    )
    print(
        "- precision: "
        f"{binary_result.precision:.17g}"
    )
    print(
        "- recall: "
        f"{binary_result.recall:.17g}"
    )
    print(
        "- f1: "
        f"{binary_result.f1:.17g}"
    )
    print(
        "- false positive rate: "
        f"{binary_result.false_positive_rate:.17g}"
    )
    print(
        "- false negative rate: "
        f"{binary_result.false_negative_rate:.17g}"
    )
    print(
        "Evaluation report: "
        f"{artifacts.summary_json}"
    )

    return 0

def _print_unsw_evaluation_dry_run(
    raw_directory: Path,
    output_root: Path,
    dpi: int,
) -> None:
    """Print the official UNSW-NB15 evaluation plan."""

    artifacts = (
        resolve_unsw_evaluation_artifacts(
            output_root
        )
    )

    operations = (
        (
            "load and validate the official "
            f"UNSW-NB15 files from {raw_directory}"
        ),
        (
            "create deterministic normal fitting, "
            "normal calibration, and official test "
            "partitions"
        ),
        (
            "fit the encoder and standardizer using "
            "normal fitting data only"
        ),
        (
            "fit PCA and select components using "
            "normal fitting data only"
        ),
        (
            "calibrate the anomaly threshold using "
            "normal calibration data only"
        ),
        (
            "freeze official test predictions "
            "before accessing test labels or "
            "attack categories"
        ),
        (
            "align predictions and hidden labels "
            "by source_partition and id"
        ),
        (
            "calculate binary and attack-category "
            "metrics without post-evaluation tuning"
        ),
        (
            "write deterministic JSON, CSV, and "
            f"PNG artifacts at {dpi} DPI"
        ),
    )

    print("DRY RUN - no files will be written")
    print(f"Raw directory: {raw_directory}")

    for index, operation in enumerate(
        operations,
        start=1,
    ):
        print(f"{index}. {operation}")

    print("Planned artifacts:")

    for field_name in (
        artifacts.__dataclass_fields__
    ):
        artifact_path = getattr(
            artifacts,
            field_name,
        )
        print(f"- {artifact_path}")


def _execute_unsw_evaluation(
    raw_directory: Path,
    output_root: Path,
    dpi: int,
) -> int:
    """Execute the frozen official UNSW-NB15 evaluation."""

    dataset = load_unsw_nb15(
        raw_directory
    )

    raw_splits = (
        split_unsw_normal_calibration_test(
            dataset
        )
    )

    standardized_splits = (
        standardize_unsw_splits(
            raw_splits
        )
    )

    detection_result = run_unsw_detection(
        standardized_splits
    )

    # Predictions are frozen before official test
    # labels or attack categories enter evaluation.
    evaluation_data = (
        align_unsw_evaluation_data(
            raw_splits.test,
            (
                detection_result
                .reconstruction_errors
                .test
            ),
            detection_result.test_predictions,
        )
    )

    binary_result = (
        evaluate_binary_predictions(
            evaluation_data
        )
    )

    attack_category_result = (
        evaluate_unsw_attack_categories(
            evaluation_data
        )
    )

    artifacts = (
        write_unsw_evaluation_artifacts(
            evaluation_data,
            detection_result,
            binary_result,
            attack_category_result,
            output_root=output_root,
            dpi=dpi,
        )
    )

    print(
        "Official UNSW-NB15 evaluation: PASSED"
    )
    print(
        "- selected components: "
        f"{detection_result.fit_result.n_components}"
    )
    print(
        "- achieved explained variance: "
        f"{detection_result.fit_result.achieved_explained_variance:.17g}"
    )
    print(
        "- frozen threshold: "
        f"{detection_result.threshold_result.threshold:.17g}"
    )
    print(
        "- confusion matrix: "
        f"{binary_result.confusion_matrix}"
    )
    print(
        "- precision: "
        f"{binary_result.precision:.17g}"
    )
    print(
        "- recall: "
        f"{binary_result.recall:.17g}"
    )
    print(
        "- f1: "
        f"{binary_result.f1:.17g}"
    )
    print(
        "- false positive rate: "
        f"{binary_result.false_positive_rate:.17g}"
    )
    print(
        "- false negative rate: "
        f"{binary_result.false_negative_rate:.17g}"
    )
    print(
        "Evaluation report: "
        f"{artifacts.summary_json}"
    )

    return 0


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command is None:
        parser.print_help()
        return 0

    if arguments.command == "validate-math":
        if arguments.dry_run:
            _print_dry_run(arguments.output)
            return 0

        return _execute_validation(
            arguments.output
        )

    if arguments.command == "evaluate-synthetic":
        if arguments.dry_run:
            _print_synthetic_evaluation_dry_run(
                arguments.output_root,
                arguments.dpi,
            )
            return 0

        return _execute_synthetic_evaluation(
            arguments.output_root,
            arguments.dpi,
        )

    if arguments.command == "evaluate-unsw":
        if arguments.dry_run:
            _print_unsw_evaluation_dry_run(
                arguments.raw_directory,
                arguments.output_root,
                arguments.dpi,
            )
            return 0

        return _execute_unsw_evaluation(
            arguments.raw_directory,
            arguments.output_root,
            arguments.dpi,
        )

    parser.error(
        f"unsupported command: {arguments.command}"
    )

    return 2
