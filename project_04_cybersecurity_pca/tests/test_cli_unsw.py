"""Tests for the official UNSW-NB15 command-line workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

import cyber_pca.cli as cli


def test_help_lists_unsw_evaluation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(
        SystemExit
    ) as exception:
        cli.main(["--help"])

    output = capsys.readouterr().out

    assert exception.value.code == 0
    assert "evaluate-unsw" in output
    assert (
        "official UNSW-NB15"
        in output
    )


def test_unsw_dry_run_writes_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_directory = (
        tmp_path
        / "raw-does-not-exist"
    )

    output_root = (
        tmp_path
        / "must-not-be-created"
    )

    exit_code = cli.main(
        [
            "evaluate-unsw",
            "--dry-run",
            "--raw-directory",
            str(raw_directory),
            "--output-root",
            str(output_root),
            "--dpi",
            "72",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "DRY RUN" in output
    assert "no files will be written" in output
    assert "normal fitting data only" in output
    assert (
        "normal calibration data only"
        in output
    )
    assert (
        "before accessing test labels"
        in output
    )
    assert str(raw_directory) in output
    assert (
        "unsw_nb15_evaluation.json"
        in output
    )
    assert (
        "unsw_nb15_predictions.csv"
        in output
    )
    assert not raw_directory.exists()
    assert not output_root.exists()


def test_unsw_command_dispatches_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_directory = (
        tmp_path
        / "official-raw"
    )

    output_root = (
        tmp_path
        / "evaluation-output"
    )

    observed: dict[
        str,
        object,
    ] = {}

    def fake_execute(
        supplied_raw_directory: Path,
        supplied_output_root: Path,
        supplied_dpi: int,
    ) -> int:
        observed["raw_directory"] = (
            supplied_raw_directory
        )
        observed["output_root"] = (
            supplied_output_root
        )
        observed["dpi"] = supplied_dpi

        return 0

    monkeypatch.setattr(
        cli,
        "_execute_unsw_evaluation",
        fake_execute,
        raising=False,
    )

    exit_code = cli.main(
        [
            "evaluate-unsw",
            "--raw-directory",
            str(raw_directory),
            "--output-root",
            str(output_root),
            "--dpi",
            "96",
        ]
    )

    assert exit_code == 0
    assert observed == {
        "raw_directory": raw_directory,
        "output_root": output_root,
        "dpi": 96,
    }
