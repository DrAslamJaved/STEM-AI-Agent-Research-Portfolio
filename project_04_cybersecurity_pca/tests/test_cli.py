"""Tests for the Project 4 command-line interface."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from cyber_pca.cli import (
    build_parser,
    deterministic_validation_matrix,
    main,
)


def test_parser_has_expected_program_name() -> None:
    parser = build_parser()

    assert parser.prog == "cyber-pca"


def test_deterministic_validation_matrix_shape() -> None:
    matrix = deterministic_validation_matrix()

    assert matrix.shape == (6, 4)
    assert matrix.dtype.name == "float64"


def test_no_command_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([])

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "validate-math" in output


def test_help_exits_cleanly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exception:
        main(["--help"])

    output = capsys.readouterr().out

    assert exception.value.code == 0
    assert "cybersecurity anomaly detection" in output


def test_dry_run_does_not_write_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "not-written.json"

    exit_code = main(
        [
            "validate-math",
            "--dry-run",
            "--output",
            str(output_path),
        ]
    )

    console_output = capsys.readouterr().out

    assert exit_code == 0
    assert "DRY RUN" in console_output
    assert "numpy.linalg.eigh" in console_output
    assert not output_path.exists()


def test_validate_math_writes_passing_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = (
        tmp_path
        / "nested"
        / "math_validation.json"
    )

    exit_code = main(
        [
            "validate-math",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()

    report = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert report["status"] == "passed"
    assert all(report["checks"].values())

    console_output = capsys.readouterr().out

    assert "PASSED" in console_output
    assert str(output_path) in console_output


def test_module_entrypoint_displays_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["cyber-pca", "--help"],
    )

    with pytest.raises(SystemExit) as exception:
        runpy.run_module(
            "cyber_pca",
            run_name="__main__",
        )

    output = capsys.readouterr().out

    assert exception.value.code == 0
    assert "validate-math" in output
