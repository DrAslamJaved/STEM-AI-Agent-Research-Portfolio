"""In-process coverage tests for the Phase 6 CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyber_pca.cli import main


EXPECTED_ARTIFACTS = (
    "results/synthetic_evaluation.json",
    "results/synthetic_predictions.csv",
    "reports/tables/synthetic_metrics.csv",
    "reports/tables/synthetic_scenario_metrics.csv",
    "reports/figures/synthetic_confusion_matrix.png",
    "reports/figures/synthetic_reconstruction_errors.png",
    "reports/figures/synthetic_scree_plot.png",
    "reports/figures/synthetic_scenario_rates.png",
)


def test_inprocess_synthetic_dry_run_writes_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "dry-run"

    exit_code = main(
        [
            "evaluate-synthetic",
            "--dry-run",
            "--output-root",
            str(output_root),
            "--dpi",
            "150",
        ]
    )

    console_output = capsys.readouterr().out

    assert exit_code == 0
    assert "DRY RUN" in console_output
    assert "no files will be written" in console_output
    assert "predict test anomalies" in console_output
    assert not output_root.exists()


def test_inprocess_synthetic_evaluation_writes_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "evaluation"

    exit_code = main(
        [
            "evaluate-synthetic",
            "--output-root",
            str(output_root),
            "--dpi",
            "150",
        ]
    )

    console_output = capsys.readouterr().out

    assert exit_code == 0
    assert "Synthetic evaluation: PASSED" in console_output
    assert "selected components: 5" in console_output
    assert (
        "confusion matrix: "
        "((797, 3), (0, 1000))"
        in console_output
    )
    assert "precision:" in console_output
    assert "recall: 1" in console_output
    assert "f1:" in console_output

    generated_files = sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    )

    assert generated_files == sorted(
        EXPECTED_ARTIFACTS
    )

    summary_path = (
        output_root
        / "results"
        / "synthetic_evaluation.json"
    )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    assert isinstance(summary, dict)
    assert summary
