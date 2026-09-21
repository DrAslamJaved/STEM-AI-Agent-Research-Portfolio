"""Run the deterministic Phase 08 release-gate smoke fixture.

This script intentionally creates only synthetic test-split records. It checks
aggregation, provenance, paired comparisons, and the conservative release gate;
it does not produce a miniF2F or model-performance result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_release import (  # noqa: E402
    FinalAttempt,
    build_release_bundle,
    sha256_file,
    write_attempts_jsonl,
    write_release_bundle,
    write_release_report,
)


def synthetic_attempts() -> list[FinalAttempt]:
    """Return a fixed, pre-declared fixture for all three study arms."""

    entries = [
        ("synthetic_nat_refl", "proof", "llm_only", 1, 0, "compiled"),
        ("synthetic_nat_add_zero", "proof", "llm_only", 1, 0, "compiled"),
        ("synthetic_fuzzy_refutation", "refutation", "llm_only", 1, 0, "compiled"),
        ("synthetic_nat_refl", "proof", "llm_sympy", 1, 0, "compiled"),
        ("synthetic_nat_add_zero", "proof", "llm_sympy", 1, 0, "compiled"),
        ("synthetic_fuzzy_refutation", "refutation", "llm_sympy", 1, 0, "compiled"),
        ("synthetic_nat_refl", "proof", "llm_lean_repair", 1, 0, "compiled"),
        ("synthetic_nat_add_zero", "proof", "llm_lean_repair", 1, 0, "failed_compile"),
        ("synthetic_nat_add_zero", "proof", "llm_lean_repair", 1, 1, "compiled"),
        ("synthetic_fuzzy_refutation", "refutation", "llm_lean_repair", 1, 0, "compiled"),
    ]
    return [
        FinalAttempt(
            candidate_id=f"phase08-{arm}-{theorem_id}-{sample_index}-{iteration}",
            theorem_id=theorem_id,
            arm=arm,
            split="test",
            sample_index=sample_index,
            iteration=iteration,
            status=status,
            kind=kind,
        )
        for theorem_id, kind, arm, sample_index, iteration, status in entries
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    project_root = args.project_root
    result_root = project_root / "results"
    task_manifest = project_root / "data" / "processed" / "phase_08_synthetic_test_tasks.jsonl"
    attempts_path = result_root / "phase_08_synthetic_attempts.jsonl"
    manifest_path = result_root / "phase_08_synthetic_final_manifest.json"
    bundle_path = result_root / "phase_08_synthetic_release.json"
    report_path = project_root / "reports" / "phase_08" / "phase_08_synthetic_release.md"

    attempts = synthetic_attempts()
    write_attempts_jsonl(attempts_path, attempts)
    manifest = {
        "schema_version": 1,
        "phase": "08",
        "run_id": "phase_08_synthetic_release_v1",
        "evaluation_split": "test",
        "test_split_locked": True,
        "formal_validity_decision": "independent_lean_compilation",
        "samples_per_theorem": 1,
        "max_repair_iterations": 2,
        "model_identifier": "scripted-offline-release-fixture",
        "seed": 20260920,
        "synthetic_demo": True,
        "arms": {
            "llm_only": {
                "lean_feedback": "forbidden",
                "sympy_feedback": "forbidden",
                "max_repair_iterations": 0,
            },
            "llm_sympy": {
                "lean_feedback": "forbidden",
                "sympy_feedback": "allowed",
                "max_repair_iterations": 0,
            },
            "llm_lean_repair": {
                "lean_feedback": "allowed",
                "sympy_feedback": "not_required",
                "max_repair_iterations": 2,
            },
        },
        "provenance": {
            "benchmark_commit": "f0dcc8b59e630fba00ba9569ca6714700e0a8801",
            "lean_toolchain": "leanprover-community/lean:3.42.1",
            "mathlib_revision": "cb2b02fff213ed6f65bebd64446baac64137dcda",
            "task_manifest_sha256": sha256_file(task_manifest),
            "candidate_records_sha256": sha256_file(attempts_path),
        },
        "human_review": {
            "approved": False,
            "reviewer": "not_applicable_synthetic_fixture",
        },
    }
    manifest_path.write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    bundle = build_release_bundle(manifest, attempts)
    write_release_bundle(bundle_path, bundle)
    write_release_report(report_path, bundle)

    print(f"Wrote synthetic final manifest: {manifest_path}")
    print(f"Wrote synthetic attempt records: {attempts_path}")
    print(f"Wrote synthetic release evidence: {bundle_path}")
    print("This is a release-gate smoke fixture, not a miniF2F or model-performance result.")


if __name__ == "__main__":
    main()
