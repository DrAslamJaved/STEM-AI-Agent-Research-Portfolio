"""Phase 9 tests for the standalone clean-environment runner and figure scripts.

Both scripts live under ``scripts/`` rather than the installed package, so
they are loaded directly from their file path with :mod:`importlib`. Importing
them must not create a virtual environment, run pytest, or write any file --
that only happens inside each script's ``main()`` / ``__main__`` guard.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module(relative_path: str, module_name: str) -> ModuleType:
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner_module() -> ModuleType:
    return _load_module("scripts/run_phase09_reproducibility.py", "_phase09_runner_under_test")


@pytest.fixture(scope="module")
def figure_module() -> ModuleType:
    return _load_module("scripts/generate_phase09_figure.py", "_phase09_figure_under_test")


def test_runner_derives_project_root_from_its_own_file_location(runner_module: ModuleType) -> None:
    expected_root = Path(runner_module.__file__).resolve().parent.parent
    assert runner_module.PROJECT_ROOT == expected_root


def test_runner_manages_only_the_phase09_artifacts_directory(runner_module: ModuleType) -> None:
    assert runner_module.MANAGED_ROOT == (
        runner_module.PROJECT_ROOT / "artifacts" / "phase09_reproducibility"
    ).resolve()
    assert runner_module.VENV_DIR.is_relative_to(runner_module.MANAGED_ROOT)
    assert runner_module.PYTEST_BASETEMP.is_relative_to(runner_module.MANAGED_ROOT)
    assert runner_module.JUNIT_XML_PATH.is_relative_to(runner_module.MANAGED_ROOT)
    assert runner_module.COVERAGE_XML_PATH.is_relative_to(runner_module.MANAGED_ROOT)
    assert runner_module.COVERAGE_JSON_PATH.is_relative_to(runner_module.MANAGED_ROOT)
    assert runner_module.CLI_MANIFEST_DIR.is_relative_to(runner_module.MANAGED_ROOT)


def test_safe_clean_removes_only_the_exact_managed_directory(
    tmp_path: Path, runner_module: ModuleType
) -> None:
    managed_root = tmp_path / "artifacts" / "phase09_reproducibility"
    (managed_root / "venv").mkdir(parents=True)
    (managed_root / "venv" / "marker.txt").write_text("x", encoding="utf-8")

    runner_module._safe_clean(managed_root, managed_root=managed_root)

    assert not managed_root.exists()


def test_safe_clean_refuses_to_remove_an_unmanaged_directory(
    tmp_path: Path, runner_module: ModuleType
) -> None:
    managed_root = tmp_path / "artifacts" / "phase09_reproducibility"
    managed_root.mkdir(parents=True)
    unrelated = tmp_path / "some_other_important_directory"
    unrelated.mkdir()
    (unrelated / "keep_me.txt").write_text("do not delete", encoding="utf-8")

    with pytest.raises(runner_module.Phase09RunnerError, match="Refusing to remove"):
        runner_module._safe_clean(unrelated, managed_root=managed_root)

    assert (unrelated / "keep_me.txt").is_file()


def test_safe_clean_refuses_a_parent_of_the_managed_directory(
    tmp_path: Path, runner_module: ModuleType
) -> None:
    managed_root = tmp_path / "artifacts" / "phase09_reproducibility"
    managed_root.mkdir(parents=True)
    project_artifacts_dir = tmp_path / "artifacts"

    with pytest.raises(runner_module.Phase09RunnerError, match="Refusing to remove"):
        runner_module._safe_clean(project_artifacts_dir, managed_root=managed_root)

    assert managed_root.exists()


def test_safe_clean_is_a_no_op_when_the_managed_directory_does_not_exist(
    tmp_path: Path, runner_module: ModuleType
) -> None:
    managed_root = tmp_path / "artifacts" / "phase09_reproducibility"

    runner_module._safe_clean(managed_root, managed_root=managed_root)

    assert not managed_root.exists()


def test_venv_python_path_matches_the_current_platform(runner_module: ModuleType) -> None:
    venv_dir = Path("/tmp/example-venv")
    python_path = runner_module._venv_python(venv_dir)
    if sys.platform == "win32":
        assert python_path == venv_dir / "Scripts" / "python.exe"
    else:
        assert python_path == venv_dir / "bin" / "python"


def test_runner_requires_the_lock_file(tmp_path: Path, runner_module: ModuleType, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "LOCK_FILE_PATH", tmp_path / "does_not_exist.lock")

    with pytest.raises(runner_module.Phase09RunnerError, match="Missing lock file"):
        runner_module._install_locked_dependencies(Path(sys.executable))


def _sample_controlled_experiments_report() -> dict[str, object]:
    return {
        "direct_rag": {
            "audit_metrics": {"coverage": 1.0, "claim_count": 300},
            "official_scifact": {
                "abstract_level": {"f1": 0.2},
                "sentence_level": {"f1": 0.1},
            },
        },
        "audited_agent": {
            "audit_metrics": {"coverage": 0.5},
            "official_scifact": {
                "abstract_level": {"f1": 0.05},
                "sentence_level": {"f1": 0.04},
            },
        },
        "comparison_to_direct_rag": {
            "official_scifact": {"abstract_level_f1": -0.15, "sentence_level_f1": -0.06},
        },
        "official_bootstrap_confidence_intervals": {
            "metrics": {
                "abstract_level_f1": {"lower": -0.2, "upper": -0.1},
                "sentence_level_f1": {"lower": -0.09, "upper": -0.03},
            }
        },
    }


def test_extract_figure_data_reads_the_expected_fields(figure_module: ModuleType) -> None:
    report = _sample_controlled_experiments_report()

    data = figure_module.extract_figure_data(report)

    assert data["coverage"] == {"direct_rag": 1.0, "audited_agent": 0.5}
    assert data["abstract_level_f1"]["delta"] == -0.15
    assert data["claim_count"] == 300


def test_build_svg_is_deterministic_and_contains_the_disclaimer(figure_module: ModuleType) -> None:
    report = _sample_controlled_experiments_report()
    data = figure_module.extract_figure_data(report)

    first = figure_module.build_svg(data)
    second = figure_module.build_svg(data)

    assert first == second
    assert figure_module.DISCLAIMER in first
    assert "not an independent test" in first
    assert first.strip().startswith("<svg")
    assert first.strip().endswith("</svg>")
    assert "Direct RAG" in first
    assert "Audited agent" in first


def test_build_svg_never_implies_blanket_superiority(figure_module: ModuleType) -> None:
    report = _sample_controlled_experiments_report()
    data = figure_module.extract_figure_data(report)

    svg_text = figure_module.build_svg(data)

    assert "not a blanket superiority claim" in svg_text


def test_requirements_lock_pins_exact_third_party_versions_only() -> None:
    lock_text = (PROJECT_ROOT / "requirements-dev.lock").read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in lock_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert lines, "requirements-dev.lock must declare at least one dependency."
    for line in lines:
        assert "==" in line, f"{line!r} must pin an exact version with '=='."
        assert not line.startswith("-e"), "requirements-dev.lock must not contain an editable install line."
        assert "scientific-evidence-agent" not in line.lower()
        assert ":" not in line.split("==")[0], f"{line!r} must not reference a local path."


def test_committed_figure_matches_the_frozen_phase08_result(figure_module: ModuleType) -> None:
    """The committed SVG must faithfully reflect the frozen, hash-verified result."""
    report = json.loads((PROJECT_ROOT / "results" / "controlled_experiments_dev.json").read_text(encoding="utf-8"))
    data = figure_module.extract_figure_data(report)
    expected_svg = figure_module.build_svg(data)

    committed_svg = (PROJECT_ROOT / "reports" / "figures" / "phase_09_evaluation_tradeoff.svg").read_text(
        encoding="utf-8"
    )

    assert committed_svg == expected_svg
