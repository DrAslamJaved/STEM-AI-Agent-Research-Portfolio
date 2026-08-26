"""Tests for the Project 03 GitHub Actions workflow."""

from pathlib import Path
from typing import Any

import yaml


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_DIRECTORY.parent

WORKFLOW_PATH = (
    REPOSITORY_ROOT
    / ".github"
    / "workflows"
    / "project_03_ci.yml"
)


def load_workflow() -> dict[str, Any]:
    """Load workflow without YAML 1.1 boolean conversion."""
    content = WORKFLOW_PATH.read_text(encoding="utf-8")

    workflow = yaml.load(
        content,
        Loader=yaml.BaseLoader,
    )

    assert isinstance(workflow, dict)
    return workflow


def test_ci_workflow_exists() -> None:
    """The repository should contain the CI workflow."""
    assert WORKFLOW_PATH.is_file()


def test_ci_has_required_triggers() -> None:
    """CI should support pushes, pull requests, and manual runs."""
    workflow = load_workflow()

    triggers = workflow["on"]

    assert "push" in triggers
    assert "pull_request" in triggers
    assert "workflow_dispatch" in triggers


def test_ci_uses_supported_python_versions() -> None:
    """The test matrix should cover supported Python versions."""
    workflow = load_workflow()

    versions = workflow["jobs"]["test"]["strategy"][
        "matrix"
    ]["python-version"]

    assert versions == ["3.11", "3.12"]


def test_ci_uses_project_working_directory() -> None:
    """Commands should run inside Project 03."""
    workflow = load_workflow()

    working_directory = workflow["jobs"]["test"][
        "defaults"
    ]["run"]["working-directory"]

    assert (
        working_directory
        == "project_03_time_series_agent"
    )


def test_ci_uses_read_only_repository_permission() -> None:
    """The workflow should use least-privilege permissions."""
    workflow = load_workflow()

    assert workflow["permissions"]["contents"] == "read"


def test_ci_uses_current_official_actions() -> None:
    """CI should use the selected official action versions."""
    workflow = load_workflow()

    steps = workflow["jobs"]["test"]["steps"]
    used_actions = {
        step["uses"]
        for step in steps
        if "uses" in step
    }

    assert "actions/checkout@v7" in used_actions
    assert "actions/setup-python@v7" in used_actions
    assert "actions/upload-artifact@v7" in used_actions


def test_ci_runs_required_validation_commands() -> None:
    """CI should compile, test, and inspect the CLI."""
    workflow = load_workflow()

    steps = workflow["jobs"]["test"]["steps"]

    commands = "\n".join(
        step.get("run", "")
        for step in steps
    )
    assert "--cov-fail-under=90" in commands
    assert "compileall src tests scripts" in commands
    assert "python -m pytest" in commands
    assert "--cov=src/time_series_agent" in commands
    assert "-W error::DeprecationWarning" in commands
    assert "python -m time_series_agent --help" in commands
    assert (
        "python -m time_series_agent "
        "run-all --dry-run"
        in commands
    )