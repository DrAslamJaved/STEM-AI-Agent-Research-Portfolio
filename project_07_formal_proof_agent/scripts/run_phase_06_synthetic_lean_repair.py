"""Run a deterministic Phase 06 Lean generate-verify-repair smoke test.

The provider and compiler are offline fixtures. They exercise feedback ordering
and evidence capture only; they do not invoke an external LLM or compile
miniF2F.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_harness import CompilerOutcome, load_tasks_jsonl  # noqa: E402
from formal_math.lean_repair import (  # noqa: E402
    InitialGenerationRequest,
    LeanCompletion,
    LeanRepairRequest,
    LeanRepairRunConfig,
    generate_verify_repair,
    write_lean_repair_attempt_records,
    write_lean_repair_evidence,
)


class ScriptedLeanRepairProvider:
    """Fixed completions chosen before the synthetic compiler runs."""

    def complete_initial(self, request: InitialGenerationRequest) -> LeanCompletion:
        responses = {
            "synthetic_nat_refl": "```lean\nby rfl\n```",
            "synthetic_nat_add_zero": "by exact 0",
        }
        return LeanCompletion(
            text=responses[request.theorem_id],
            model_identifier="scripted-lean-repair-offline-provider",
            input_tokens=16,
            output_tokens=4,
        )

    def complete_repair(self, request: LeanRepairRequest) -> LeanCompletion:
        if request.theorem_id != "synthetic_nat_add_zero" or request.repair_iteration != 1:
            raise AssertionError("synthetic fixture expected one repair for synthetic_nat_add_zero")
        return LeanCompletion(
            text="by simpa using Nat.add_zero 3",
            model_identifier="scripted-lean-repair-offline-provider",
            input_tokens=32,
            output_tokens=6,
        )


def synthetic_compiler(source: str) -> CompilerOutcome:
    """A deterministic stand-in for independent Lean compilation in this smoke test."""
    accepted = "by rfl" in source or "by simpa using Nat.add_zero 3" in source
    return CompilerOutcome(
        passed=accepted,
        stdout="synthetic evaluator accepted candidate" if accepted else "",
        stderr="" if accepted else "synthetic Lean diagnostic: proof body did not typecheck",
        exit_code=0 if accepted else 1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    project_root = args.project_root

    tasks = load_tasks_jsonl(project_root / "data" / "processed" / "phase_06_synthetic_tasks.jsonl")
    config = LeanRepairRunConfig(
        run_id="phase_06_synthetic_lean_repair_v1",
        model_identifier="scripted-lean-repair-offline-provider",
        prompt_id="phase_06_lean_repair_prompt_v1",
        max_repair_iterations=1,
        seed=20260919,
    )
    attempts = generate_verify_repair(tasks, ScriptedLeanRepairProvider(), config, synthetic_compiler)

    result_root = project_root / "results"
    write_lean_repair_attempt_records(result_root / "phase_06_synthetic_attempts.jsonl", attempts)
    write_lean_repair_evidence(
        result_root / "phase_06_synthetic_lean_repair.json",
        config=config,
        attempts=attempts,
        synthetic_demo=True,
    )
    print(f"Wrote Phase 06 synthetic attempts: {result_root / 'phase_06_synthetic_attempts.jsonl'}")
    print(f"Wrote Phase 06 synthetic repair evidence: {result_root / 'phase_06_synthetic_lean_repair.json'}")
    print("This is a Lean-repair control smoke test, not a miniF2F or model-performance result.")


if __name__ == "__main__":
    main()
