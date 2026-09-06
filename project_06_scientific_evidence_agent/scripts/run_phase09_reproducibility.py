"""One-command, cross-platform clean-environment reproducibility gate.

Phase 9 must prove that a brand-new, isolated Python environment -- built
solely from ``requirements-dev.lock`` and an editable install of this project
-- reproduces the test suite and the Phase 9 evidence-integrity gate. This
script is the single entry point for that proof.

Usage (PowerShell), from the project root:

    & .\\.venv\\Scripts\\python.exe .\\scripts\\run_phase09_reproducibility.py --clean

Usage (POSIX shells):

    ./.venv/bin/python ./scripts/run_phase09_reproducibility.py --clean

The script itself uses only the standard library, so it can be invoked with
any Python 3.12 interpreter; it does not need this project's own
dependencies to be pre-installed in the interpreter that runs it. It:

1. derives the project root from its own file location, never the caller's
   current working directory;
2. creates an isolated virtual environment only under
   ``artifacts/phase09_reproducibility/`` (optionally removing that exact,
   managed directory first when ``--clean`` is supplied -- and refuses to
   remove anything else);
3. installs the exact locked third-party dependencies from
   ``requirements-dev.lock``, then installs this project editable with
   ``--no-deps`` so dependency resolution is never re-run;
4. runs ``pip check``, the full pytest suite (with JUnit XML and branch
   coverage XML/JSON/terminal reports), ``compileall``, and the Phase 9
   ``evidence_agent reproducibility`` CLI check;
5. writes every transient output under ``artifacts/phase09_reproducibility/``;
6. returns a non-zero exit status on the first failing step.

No step downloads the raw SciFact release, trains or recalibrates a model,
or writes to any Git-tracked ``results/``, ``reports/``, or ``agent_trace/``
path.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANAGED_ROOT = (PROJECT_ROOT / "artifacts" / "phase09_reproducibility").resolve()
VENV_DIR = MANAGED_ROOT / "venv"
PYTEST_BASETEMP = MANAGED_ROOT / "pytest_tmp"
JUNIT_XML_PATH = MANAGED_ROOT / "pytest_junit.xml"
COVERAGE_XML_PATH = MANAGED_ROOT / "coverage.xml"
COVERAGE_JSON_PATH = MANAGED_ROOT / "coverage.json"
CLI_MANIFEST_DIR = MANAGED_ROOT / "manifest"
LOCK_FILE_PATH = PROJECT_ROOT / "requirements-dev.lock"
REPRODUCIBILITY_CONFIG_PATH = PROJECT_ROOT / "configs" / "reproducibility.yaml"


class Phase09RunnerError(RuntimeError):
    """Raised when a clean-environment gate step cannot run at all."""


def _venv_python(venv_dir: Path) -> Path:
    """Return the interpreter path for a venv, without requiring it to exist."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _safe_clean(target: Path, *, managed_root: Path = MANAGED_ROOT) -> None:
    """Remove ``target`` only if it is exactly the one managed Phase 9 directory.

    This is the guard that makes ``--clean`` safe: it never walks up or down
    to a different directory, and it never removes anything outside the
    Phase 9 artifacts tree even if called with an unexpected path.
    """
    resolved_target = target.resolve()
    resolved_managed_root = managed_root.resolve()
    if resolved_target != resolved_managed_root:
        raise Phase09RunnerError(
            f"Refusing to remove {resolved_target}: it is not the managed "
            f"Phase 9 directory {resolved_managed_root}."
        )
    if resolved_target.exists():
        shutil.rmtree(resolved_target)


def _run(command: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise Phase09RunnerError(
            f"Command failed with exit code {result.returncode}: {' '.join(str(part) for part in command)}"
        )


def _create_isolated_environment() -> Path:
    print(f"Creating isolated virtual environment at {VENV_DIR}", flush=True)
    venv.EnvBuilder(with_pip=True, clear=True).create(VENV_DIR)
    python = _venv_python(VENV_DIR)
    if not python.is_file():
        raise Phase09RunnerError(f"Isolated interpreter was not created at {python}.")
    return python


def _install_locked_dependencies(python: Path) -> None:
    if not LOCK_FILE_PATH.is_file():
        raise Phase09RunnerError(f"Missing lock file: {LOCK_FILE_PATH}")
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    _run([str(python), "-m", "pip", "install", "--no-deps", "-r", str(LOCK_FILE_PATH)])
    _run([str(python), "-m", "pip", "install", "--no-deps", "-e", str(PROJECT_ROOT)])
    _run([str(python), "-m", "pip", "check"])


def _run_pytest(python: Path) -> None:
    _run(
        [
            str(python),
            "-m",
            "pytest",
            "--basetemp",
            str(PYTEST_BASETEMP),
            f"--junitxml={JUNIT_XML_PATH}",
            "--cov=evidence_agent",
            "--cov-branch",
            f"--cov-report=xml:{COVERAGE_XML_PATH}",
            f"--cov-report=json:{COVERAGE_JSON_PATH}",
            "--cov-report=term-missing",
            "-q",
        ]
    )


def _run_compileall(python: Path) -> None:
    _run([str(python), "-m", "compileall", "-q", "src", "tests"])


def _run_reproducibility_check(python: Path) -> None:
    _run(
        [
            str(python),
            "-m",
            "evidence_agent",
            "reproducibility",
            "--config",
            str(REPRODUCIBILITY_CONFIG_PATH),
            "--output-dir",
            str(CLI_MANIFEST_DIR),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the managed artifacts/phase09_reproducibility/ directory before running.",
    )
    args = parser.parse_args(argv)

    try:
        if args.clean:
            _safe_clean(MANAGED_ROOT)
        MANAGED_ROOT.mkdir(parents=True, exist_ok=True)

        python = _create_isolated_environment()
        _install_locked_dependencies(python)
        _run_pytest(python)
        _run_compileall(python)
        _run_reproducibility_check(python)
    except Phase09RunnerError as error:
        print(f"Phase 9 reproducibility gate FAILED: {error}", file=sys.stderr)
        return 1

    print("Phase 9 reproducibility gate PASSED.")
    print(f"JUnit XML:      {JUNIT_XML_PATH}")
    print(f"Coverage XML:   {COVERAGE_XML_PATH}")
    print(f"Coverage JSON:  {COVERAGE_JSON_PATH}")
    print(f"Manifest:       {CLI_MANIFEST_DIR}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
