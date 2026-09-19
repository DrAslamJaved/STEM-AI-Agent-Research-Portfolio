#!/usr/bin/env python3
"""Validate the compact, committed evidence for Project 07 Phase 02."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "benchmark": "openai/miniF2F",
    "frozen_branch": "v1",
    "benchmark_commit": "f0dcc8b59e630fba00ba9569ca6714700e0a8801",
    "lean_toolchain": "leanprover-community/lean:3.42.1",
    "mathlib_revision": "cb2b02fff213ed6f65bebd64446baac64137dcda",
    "configure_status": "passed",
    "build_status": "passed",
    "build_exit_code": 0,
}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_manifest(manifest: dict) -> list[str]:
    failures: list[str] = []
    for key, expected in EXPECTED.items():
        if manifest.get(key) != expected:
            failures.append(
                f"{key!r}: expected {expected!r}, found {manifest.get(key)!r}"
            )
    if not manifest.get("lean_version", "").startswith("Lean (version 3.42.1,"):
        failures.append("lean_version must identify Lean 3.42.1")
    if not manifest.get("validated_at_utc"):
        failures.append("validated_at_utc is required")
    return failures


def validate_provenance_text(path: Path) -> list[str]:
    if not path.is_file():
        return [f"Missing required provenance manifest: {path}"]
    text = path.read_text(encoding="utf-8")
    required = (
        EXPECTED["benchmark_commit"],
        EXPECTED["frozen_branch"],
        EXPECTED["lean_toolchain"],
        EXPECTED["mathlib_revision"],
    )
    return [f"Provenance manifest is missing {item!r}" for item in required if item not in text]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()

    manifest_path = project_root / "reports" / "phase_02" / "toolchain_validation.json"
    provenance_path = project_root / "benchmarks" / "minif2f" / "provenance.json"
    failures = validate_manifest(read_json(manifest_path))
    failures.extend(validate_provenance_text(provenance_path))

    if failures:
        print("FAIL: Phase 02 toolchain evidence is invalid")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("PASS: Phase 02 toolchain evidence is locked and valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
