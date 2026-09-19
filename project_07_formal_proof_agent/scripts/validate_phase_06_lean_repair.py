"""Validate the Phase 06 Lean repair policy before a run is staged."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.lean_repair import LEAN_REPAIR_POLICY, PINNED_LEAN_TOOLCHAIN  # noqa: E402


def require(text: str, expected: str, label: str) -> None:
    if expected not in text:
        raise AssertionError(f"missing {label}: {expected}")


def main() -> None:
    config = PROJECT_ROOT / "configs" / "phase_06_lean_repair.yaml"
    prompt = PROJECT_ROOT / "prompts" / "phase_06_lean_repair_prompt.md"
    module = PROJECT_ROOT / "src" / "formal_math" / "lean_repair.py"
    fixture = PROJECT_ROOT / "data" / "processed" / "phase_06_synthetic_tasks.jsonl"
    for path in (config, prompt, module, fixture):
        if not path.is_file():
            raise AssertionError(f"required Phase 06 file is missing: {path}")

    configuration = config.read_text(encoding="utf-8")
    require(configuration, "arm: llm_lean_repair", "Lean repair arm")
    require(configuration, "initial_generation_feedback: none", "initial no-feedback policy")
    require(
        configuration,
        "lean_feedback_during_generation: allowed_compiler_diagnostics_and_static_policy",
        "Lean feedback policy",
    )
    require(configuration, "theorem_declaration_edits: forbidden", "immutable theorem policy")
    require(configuration, "max_repair_iterations: 2", "bounded repair budget")
    require(configuration, "development_split: valid", "development split")
    require(configuration, "test_split_locked: true", "locked test split")
    require(configuration, PINNED_LEAN_TOOLCHAIN, "pinned Lean toolchain")

    if LEAN_REPAIR_POLICY["theorem_declaration_edits"] != "forbidden":
        raise AssertionError("Phase 06 must forbid theorem declaration edits")
    source = module.read_text(encoding="utf-8")
    require(source, "evaluate_candidate(", "independent evaluation call")
    require(source, "LeanSubprocessCompiler", "pinned Lean compiler factory")
    print("PASS: Phase 06 Lean generate-verify-repair workflow is locked and valid")


if __name__ == "__main__":
    main()
