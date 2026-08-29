"""Black-box tests for synthetic evaluation CLI."""

import json
from pathlib import Path
import subprocess
import sys
import pytest


def _run_module(
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "cyber_pca",
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_module_help_lists_synthetic_evaluation() -> None:
    result = _run_module("--help")

    assert result.returncode == 0
    assert "evaluate-synthetic" in result.stdout


def test_synthetic_evaluation_dry_run_writes_nothing(
    tmp_path: Path,
) -> None:
    result = _run_module(
        "evaluate-synthetic",
        "--dry-run",
        "--output-root",
        str(tmp_path),
    )

    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
    assert "no files will be written" in result.stdout
    assert not any(tmp_path.rglob("*"))


def test_synthetic_evaluation_command_writes_artifacts(
    tmp_path: Path,
) -> None:
    result = _run_module(
        "evaluate-synthetic",
        "--output-root",
        str(tmp_path),
    )

    assert result.returncode == 0, result.stderr

    assert (
        "Synthetic evaluation: PASSED"
        in result.stdout
    )

    expected_paths = [
        (
            tmp_path
            / "results"
            / "synthetic_evaluation.json"
        ),
        (
            tmp_path
            / "results"
            / "synthetic_predictions.csv"
        ),
        (
            tmp_path
            / "reports"
            / "tables"
            / "synthetic_metrics.csv"
        ),
        (
            tmp_path
            / "reports"
            / "tables"
            / "synthetic_scenario_metrics.csv"
        ),
        (
            tmp_path
            / "reports"
            / "figures"
            / "synthetic_confusion_matrix.png"
        ),
        (
            tmp_path
            / "reports"
            / "figures"
            / "synthetic_reconstruction_errors.png"
        ),
        (
            tmp_path
            / "reports"
            / "figures"
            / "synthetic_scree_plot.png"
        ),
        (
            tmp_path
            / "reports"
            / "figures"
            / "synthetic_scenario_rates.png"
        ),
    ]

    for expected_path in expected_paths:
        assert expected_path.is_file()
        assert expected_path.stat().st_size > 0

    summary_path = expected_paths[0]

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert summary["status"] == "passed"

    assert summary["metrics"][
        "confusion_matrix"
    ] == [
        [797, 3],
        [0, 1000],
    ]

    assert summary["metrics"][
        "precision"
    ] == pytest.approx(
        0.9970089730807578
    )

    assert summary["metrics"][
        "recall"
    ] == pytest.approx(1.0)

    assert summary["metrics"][
        "f1"
    ] == pytest.approx(
        0.9985022466300548
    )
