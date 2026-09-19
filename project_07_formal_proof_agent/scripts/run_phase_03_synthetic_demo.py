"""Run a clearly labelled Phase 03 harness smoke test.

This does not invoke an LLM or claim benchmark performance.  It proves the
candidate capture, shortcut-rejection, and independent compiler interfaces.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_harness import (  # noqa: E402
    Candidate,
    CompilerOutcome,
    Task,
    evaluate_batch,
    write_evaluation_bundle,
)


def fake_compiler(source: str) -> CompilerOutcome:
    """Deterministic stand-in used only for this synthetic smoke run."""
    if "by rfl" in source:
        return CompilerOutcome(passed=True, stdout="synthetic compiler accepted rfl", exit_code=0)
    return CompilerOutcome(passed=False, stderr="synthetic compiler rejected candidate", exit_code=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project 07 root; defaults to the script's parent directory.",
    )
    args = parser.parse_args()

    tasks = [
        Task(
            theorem_id="synthetic_nat_refl",
            split="valid",
            declaration="example : (1 : Nat) = 1 :=",
            source="synthetic_phase_03_demo",
        )
    ]
    candidates = [
        Candidate(
            candidate_id="synthetic_llm_only_valid",
            theorem_id="synthetic_nat_refl",
            arm="llm_only",
            lean_code="by rfl",
            prompt_id="phase_03_synthetic",
        ),
        Candidate(
            candidate_id="synthetic_sympy_shortcut",
            theorem_id="synthetic_nat_refl",
            arm="llm_sympy",
            lean_code="by sorry",
            prompt_id="phase_03_synthetic",
        ),
        Candidate(
            candidate_id="synthetic_lean_repair_failure",
            theorem_id="synthetic_nat_refl",
            arm="llm_lean_repair",
            lean_code="by exact 0",
            prompt_id="phase_03_synthetic",
            iteration=1,
        ),
    ]
    results = evaluate_batch(tasks, candidates, fake_compiler, mode="development")
    output = args.project_root / "results" / "phase_03_synthetic_demo.json"
    write_evaluation_bundle(
        output,
        tasks=tasks,
        candidates=candidates,
        results=results,
        mode="development",
        synthetic_demo=True,
    )
    print(f"Wrote synthetic Phase 03 evidence: {output}")
    print("This is a harness smoke test, not a miniF2F or model-performance result.")


if __name__ == "__main__":
    main()
