"""Unified command-line interface for the time-series agent."""

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from time_series_agent.exceptions import (
    CliExecutionError,
)


ScriptRunner = Callable[..., subprocess.CompletedProcess]


WORKFLOWS: dict[str, tuple[str, ...]] = {
    "audit-raw": (
        "inspect_raw_data.py",
    ),
    "validate": (
        "run_validation.py",
    ),
    "preprocess": (
        "run_preprocessing.py",
    ),
    "explore": (
        "run_exploration.py",
    ),
    "forecast": (
        "run_gradient_boosting_preview.py",
    ),
    "evaluate": (
        "run_holdout_evaluation.py",
        "run_expanding_validation.py",
        "run_holt_winters_validation.py",
        "run_machine_learning_validation.py",
    ),
    "anomalies": (
        "run_residual_collection.py",
        "run_anomaly_detection.py",
        "run_anomaly_episode_analysis.py",
        "run_anomaly_reporting.py",
    ),
    "recommend": (
        "run_model_recommendation.py",
    ),
    "run-all": (
        "inspect_raw_data.py",
        "run_validation.py",
        "run_preprocessing.py",
        "run_exploration.py",
        "run_feature_engineering.py",
        "run_baseline_preview.py",
        "run_holdout_evaluation.py",
        "run_expanding_validation.py",
        "run_holt_winters_preview.py",
        "run_holt_winters_validation.py",
        "run_gradient_boosting_preview.py",
        "run_machine_learning_validation.py",
        "run_residual_collection.py",
        "run_anomaly_detection.py",
        "run_anomaly_episode_analysis.py",
        "run_anomaly_reporting.py",
        "run_model_recommendation.py",
    ),
}


WORKFLOW_DESCRIPTIONS = {
    "audit-raw": (
        "Inspect the immutable raw dataset."
    ),
    "validate": (
        "Run structured time-series validation."
    ),
    "preprocess": (
        "Create leakage-safe processed data."
    ),
    "explore": (
        "Generate exploratory metrics and figures."
    ),
    "forecast": (
        "Generate the preferred next-24-hour forecast."
    ),
    "evaluate": (
        "Evaluate baseline, classical, and machine-learning models."
    ),
    "anomalies": (
        "Collect residuals, detect anomalies, group episodes, "
        "and create reports."
    ),
    "recommend": (
        "Generate the preferred-model and fallback-model decision."
    ),
    "run-all": (
        "Run the complete reproducible project pipeline."
    ),
}


def get_project_root() -> Path:
    """Return the repository directory containing scripts."""
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="time-series-agent",
        description=(
            "Reproducible hourly forecasting and "
            "residual-anomaly agent."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="workflow",
        required=True,
        title="workflows",
    )

    for workflow_name in WORKFLOWS:
        workflow_parser = subparsers.add_parser(
            workflow_name,
            help=WORKFLOW_DESCRIPTIONS[
                workflow_name
            ],
            description=WORKFLOW_DESCRIPTIONS[
                workflow_name
            ],
        )

        workflow_parser.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "Print the execution plan without "
                "running any scripts."
            ),
        )

    return parser


def _resolve_workflow_scripts(
    workflow: str,
    project_root: Path,
) -> tuple[Path, ...]:
    """Resolve and validate every script in one workflow."""
    if workflow not in WORKFLOWS:
        raise CliExecutionError(
            f"Unknown workflow: '{workflow}'."
        )

    script_directory = project_root / "scripts"

    script_paths = tuple(
        script_directory / script_name
        for script_name in WORKFLOWS[workflow]
    )

    missing_scripts = [
        str(path)
        for path in script_paths
        if not path.is_file()
    ]

    if missing_scripts:
        raise CliExecutionError(
            "Workflow contains missing scripts: "
            + ", ".join(missing_scripts)
        )

    return script_paths


def _print_execution_plan(
    workflow: str,
    script_paths: tuple[Path, ...],
    dry_run: bool,
) -> None:
    """Print a numbered, readable execution plan."""
    print(f"Workflow: {workflow}")
    print(
        f"Steps: {len(script_paths)}"
    )
    print()

    for step_number, script_path in enumerate(
        script_paths,
        start=1,
    ):
        print(
            f"{step_number}. "
            f"scripts/{script_path.name}"
        )

    if dry_run:
        print()
        print(
            "Dry run complete: no scripts were executed."
        )


def run_workflow(
    workflow: str,
    dry_run: bool = False,
    project_root: str | Path | None = None,
    runner: ScriptRunner = subprocess.run,
) -> tuple[Path, ...]:
    """Run or preview one configured workflow."""
    root = (
        Path(project_root)
        if project_root is not None
        else get_project_root()
    )

    script_paths = _resolve_workflow_scripts(
        workflow=workflow,
        project_root=root,
    )

    _print_execution_plan(
        workflow=workflow,
        script_paths=script_paths,
        dry_run=dry_run,
    )

    if dry_run:
        return tuple()

    completed_paths: list[Path] = []

    for step_number, script_path in enumerate(
        script_paths,
        start=1,
    ):
        print()
        print(
            f"[{step_number}/{len(script_paths)}] "
            f"Running {script_path.name}",
            flush=True,
        )

        try:
            runner(
                [
                    sys.executable,
                    str(script_path),
                ],
                cwd=root,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            raise CliExecutionError(
                f"Workflow '{workflow}' failed while "
                f"running '{script_path.name}' with "
                f"exit code {error.returncode}."
            ) from error
        except OSError as error:
            raise CliExecutionError(
                f"Workflow '{workflow}' could not start "
                f"'{script_path.name}': {error}"
            ) from error

        completed_paths.append(script_path)

    print()
    print(
        f"Workflow '{workflow}' completed successfully."
    )

    return tuple(completed_paths)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        run_workflow(
            workflow=arguments.workflow,
            dry_run=arguments.dry_run,
        )
    except CliExecutionError as error:
        parser.exit(
            status=1,
            message=f"ERROR: {error}\n",
        )

    return 0