"""Command-line interface for Project 4 workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from cyber_pca.validation import run_math_validation


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

    parser.error(
        f"unsupported command: {arguments.command}"
    )

    return 2
