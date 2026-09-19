"""Validate the static Phase 05 policy boundary before a run is staged."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.sympy_counterexamples import SYMPY_WORKFLOW_POLICY  # noqa: E402


def require(text: str, expected: str, label: str) -> None:
    if expected not in text:
        raise AssertionError(f"missing {label}: {expected}")


def main() -> None:
    config = PROJECT_ROOT / "configs" / "phase_05_sympy_counterexamples.yaml"
    prompt = PROJECT_ROOT / "prompts" / "phase_05_llm_sympy_prompt.md"
    module = PROJECT_ROOT / "src" / "formal_math" / "sympy_counterexamples.py"
    requirements = PROJECT_ROOT / "requirements-phase05.txt"
    fixture = PROJECT_ROOT / "data" / "processed" / "phase_05_synthetic_tasks.jsonl"
    for path in (config, prompt, module, requirements, fixture):
        if not path.is_file():
            raise AssertionError(f"required Phase 05 file is missing: {path}")

    configuration = config.read_text(encoding="utf-8")
    require(configuration, "arm: llm_sympy", "SymPy arm")
    require(configuration, "lean_feedback_during_generation: forbidden", "Lean-feedback prohibition")
    require(configuration, "compiler_feedback_during_generation: forbidden", "compiler-feedback prohibition")
    require(configuration, "arithmetic: exact_rational_only", "exact arithmetic policy")
    require(configuration, "development_split: valid", "development split")
    require(configuration, "test_split_locked: true", "locked test split")
    require(requirements.read_text(encoding="utf-8"), "sympy", "SymPy dependency")
    if SYMPY_WORKFLOW_POLICY["lean_feedback_during_generation"] != "forbidden":
        raise AssertionError("Phase 05 must not expose Lean feedback during generation")

    source = module.read_text(encoding="utf-8")
    if "evaluate_batch(" in source:
        raise AssertionError("Phase 05 generator must not evaluate candidates during generation")
    print("PASS: Phase 05 SymPy counterexample workflow is locked and valid")


if __name__ == "__main__":
    main()
