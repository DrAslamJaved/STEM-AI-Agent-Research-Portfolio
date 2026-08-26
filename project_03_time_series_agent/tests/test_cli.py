"""Tests for the unified command-line interface."""

import subprocess
import sys
from pathlib import Path

import pytest

from time_series_agent.cli import (
    WORKFLOWS,
    build_parser,
    main,
    run_workflow,
)
from time_series_agent.exceptions import (
    CliExecutionError,
)


def create_workflow_scripts(
    root: Path,
    workflow: str,
) -> None:
    """Create empty scripts required by one test workflow."""
    script_directory = root / "scripts"
    script_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for script_name in WORKFLOWS[workflow]:
        (
            script_directory / script_name
        ).write_text(
            "# test script\n",
            encoding="utf-8",
        )


def test_parser_accepts_every_workflow() -> None:
    """Every configured workflow should be parseable."""
    parser = build_parser()

    for workflow in WORKFLOWS:
        arguments = parser.parse_args(
            [workflow]
        )

        assert arguments.workflow == workflow
        assert not arguments.dry_run


def test_parser_accepts_dry_run() -> None:
    """Dry-run flag should be accepted after a workflow."""
    arguments = build_parser().parse_args(
        ["run-all", "--dry-run"]
    )

    assert arguments.workflow == "run-all"
    assert arguments.dry_run


def test_dry_run_does_not_execute_scripts(
    tmp_path,
    capsys,
) -> None:
    """Dry run should print the plan without executing it."""
    create_workflow_scripts(
        tmp_path,
        "anomalies",
    )

    calls = []

    def fake_runner(*args, **kwargs):
        calls.append(
            (args, kwargs)
        )

    completed = run_workflow(
        workflow="anomalies",
        dry_run=True,
        project_root=tmp_path,
        runner=fake_runner,
    )

    output = capsys.readouterr().out

    assert completed == ()
    assert calls == []
    assert "Dry run complete" in output
    assert "run_anomaly_detection.py" in output


def test_workflow_executes_in_defined_order(
    tmp_path,
) -> None:
    """Scripts should run in their declared order."""
    create_workflow_scripts(
        tmp_path,
        "anomalies",
    )

    calls = []

    def fake_runner(
        command,
        cwd,
        check,
    ):
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "check": check,
            }
        )

        return subprocess.CompletedProcess(
            command,
            returncode=0,
        )

    completed = run_workflow(
        workflow="anomalies",
        project_root=tmp_path,
        runner=fake_runner,
    )

    assert [
        path.name
        for path in completed
    ] == list(WORKFLOWS["anomalies"])

    assert [
        Path(call["command"][1]).name
        for call in calls
    ] == list(WORKFLOWS["anomalies"])

    assert all(
        call["command"][0] == sys.executable
        for call in calls
    )
    assert all(
        call["cwd"] == tmp_path
        for call in calls
    )
    assert all(
        call["check"]
        for call in calls
    )


def test_missing_script_is_rejected(
    tmp_path,
) -> None:
    """A workflow cannot silently skip missing scripts."""
    with pytest.raises(
        CliExecutionError,
        match="missing scripts",
    ):
        run_workflow(
            workflow="validate",
            project_root=tmp_path,
        )


def test_unknown_workflow_is_rejected(
    tmp_path,
) -> None:
    """Unknown workflows should fail clearly."""
    with pytest.raises(
        CliExecutionError,
        match="Unknown workflow",
    ):
        run_workflow(
            workflow="unknown",
            project_root=tmp_path,
        )


def test_script_failure_stops_workflow(
    tmp_path,
) -> None:
    """A failed script should stop the workflow."""
    create_workflow_scripts(
        tmp_path,
        "anomalies",
    )

    calls = []

    def failing_runner(
        command,
        cwd,
        check,
    ):
        calls.append(command)

        if len(calls) == 2:
            raise subprocess.CalledProcessError(
                returncode=7,
                cmd=command,
            )

        return subprocess.CompletedProcess(
            command,
            returncode=0,
        )

    with pytest.raises(
        CliExecutionError,
        match="exit code 7",
    ):
        run_workflow(
            workflow="anomalies",
            project_root=tmp_path,
            runner=failing_runner,
        )

    assert len(calls) == 2


def test_run_all_contains_every_project_stage() -> None:
    """The complete workflow should contain all stages."""
    run_all = WORKFLOWS["run-all"]

    expected_scripts = {
        "run_validation.py",
        "run_preprocessing.py",
        "run_exploration.py",
        "run_feature_engineering.py",
        "run_machine_learning_validation.py",
        "run_residual_collection.py",
        "run_anomaly_detection.py",
        "run_anomaly_episode_analysis.py",
        "run_anomaly_reporting.py",
        "run_model_recommendation.py",
    }

    assert expected_scripts.issubset(
        set(run_all)
    )

    assert len(run_all) == len(set(run_all))


def test_main_supports_real_dry_run(
    capsys,
) -> None:
    """The installed entry function should support dry run."""
    exit_code = main(
        ["validate", "--dry-run"]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "run_validation.py" in output
    assert "no scripts were executed" in output


def test_python_module_help_is_available() -> None:
    """Package module should expose command help."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "time_series_agent",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "run-all" in result.stdout
    assert "anomalies" in result.stdout