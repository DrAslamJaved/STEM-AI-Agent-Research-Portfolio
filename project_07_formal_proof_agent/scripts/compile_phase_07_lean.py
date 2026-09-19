"""Compile the Phase 07 fuzzy-similarity Lean source with the frozen toolchain.

The command writes evidence even when Lean returns a nonzero exit code, then
returns that code.  A successful record is the only Phase 07 artifact that may
be described as formal proof validity.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFAULT_PROJECT_ROOT / "src"))

from formal_math.fuzzy_formalization import (  # noqa: E402
    PINNED_LEAN_TOOLCHAIN,
    PINNED_MATHLIB_REVISION,
    PINNED_MINIF2F_COMMIT,
    build_lean_compile_command,
    sha256_file,
    source_is_ascii,
    source_path,
    static_policy_violations,
)


def utc_now() -> str:
    """Return a portable UTC timestamp with an explicit Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def pinned_commit(lean_cwd: Path) -> str:
    """Read the checked-out miniF2F commit without mutating the checkout."""
    result = subprocess.run(
        ("git", "-C", str(lean_cwd), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        diagnostic = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"unable to read miniF2F commit: {diagnostic}")
    return result.stdout.strip()


def write_evidence(path: Path, payload: dict[str, Any]) -> None:
    """Persist compact compiler evidence without storing a machine-specific CWD."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--lean-cwd",
        type=Path,
        required=True,
        help="The Phase 02 configured miniF2F v1 checkout containing leanpkg.toml.",
    )
    parser.add_argument(
        "--elan",
        type=Path,
        default=Path.home() / ".elan" / "bin" / "elan.exe",
        help="Path to elan.exe; defaults to the standard Windows installation path.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    lean_cwd = args.lean_cwd.resolve()
    elan = args.elan.resolve()
    output = args.output or project_root / "results" / "phase_07_lean_compile.json"
    source = source_path(project_root)

    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"Phase 07 Lean source is missing: {source}")
    source_text = source.read_text(encoding="utf-8")
    violations = static_policy_violations(source_text)
    if violations:
        raise ValueError(f"Phase 07 Lean source violates static policy: {', '.join(violations)}")
    if not source_is_ascii(source_text):
        raise ValueError("Phase 07 Lean source must be ASCII for the pinned Windows Lean 3 compiler")
    if not (lean_cwd / "leanpkg.toml").is_file():
        raise FileNotFoundError(f"Pinned miniF2F working directory is invalid: {lean_cwd}")
    if not elan.is_file():
        raise FileNotFoundError(f"elan executable is missing: {elan}")

    actual_commit = pinned_commit(lean_cwd)
    if actual_commit != PINNED_MINIF2F_COMMIT:
        raise RuntimeError(
            "miniF2F commit mismatch: "
            f"expected {PINNED_MINIF2F_COMMIT}, found {actual_commit}. Do not substitute another revision."
        )

    command = build_lean_compile_command(elan, source)
    started_at = utc_now()
    try:
        result = subprocess.run(
            command,
            cwd=lean_cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
        )
        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
        status = "compiled" if exit_code == 0 else "failed_compile"
    except subprocess.TimeoutExpired as error:
        exit_code = None
        stdout = error.stdout or ""
        stderr = (error.stderr or "") + f"\nTimed out after {args.timeout_seconds} seconds."
        status = "compiler_timeout"

    payload: dict[str, Any] = {
        "schema_version": 1,
        "phase": "07",
        "formal_validity_decision": "independent_lean_compilation",
        "benchmark_commit": PINNED_MINIF2F_COMMIT,
        "lean_toolchain": PINNED_LEAN_TOOLCHAIN,
        "mathlib_revision": PINNED_MATHLIB_REVISION,
        "source_path": "formalizations/lean3/FuzzySimilarity.lean",
        "source_encoding": "ASCII",
        "source_sha256": sha256_file(source),
        "compiler_command": list(command[1:-1]) + ["formalizations/lean3/FuzzySimilarity.lean"],
        "compiler_cwd_kind": "configured_pinned_minif2f_v1_checkout",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "compiler_exit_code": exit_code,
        "compiler_stdout": stdout,
        "compiler_stderr": stderr,
        "status": status,
        "compilation_passed": status == "compiled",
    }
    write_evidence(output, payload)
    print(f"Wrote Phase 07 Lean compiler evidence: {output}")
    print(f"Phase 07 Lean status: {status}")
    return 0 if status == "compiled" else (1 if exit_code is None else exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
