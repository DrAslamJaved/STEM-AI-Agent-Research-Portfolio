"""Validate the locked Phase 04 LLM-only baseline contract."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.llm_only_baseline import LLM_ONLY_POLICY  # noqa: E402


def require(text: str, fragment: str, description: str) -> None:
    if fragment not in text:
        raise AssertionError(f"missing {description}: {fragment!r}")


def main() -> None:
    config = PROJECT_ROOT / "configs" / "phase_04_llm_only_baseline.yaml"
    prompt = PROJECT_ROOT / "prompts" / "phase_04_llm_only_proof_prompt.md"
    module = PROJECT_ROOT / "src" / "formal_math" / "llm_only_baseline.py"
    fixture = PROJECT_ROOT / "data" / "processed" / "phase_04_synthetic_tasks.jsonl"
    for path in (config, prompt, module, fixture):
        if not path.is_file():
            raise AssertionError(f"required Phase 04 file is missing: {path}")

    configuration = config.read_text(encoding="utf-8")
    require(configuration, "arm: llm_only", "LLM-only arm")
    require(configuration, "verification_feedback_during_generation: forbidden", "no-feedback policy")
    require(configuration, "tool_calls_during_generation: forbidden", "tool-free policy")
    require(configuration, "evaluation_timing: post_generation_only", "post-generation evaluation")
    require(configuration, "development_split: valid", "development split")
    require(configuration, "test_split_locked: true", "locked test split")
    if LLM_ONLY_POLICY["repair_iterations"] != 0:
        raise AssertionError("LLM-only baseline must have zero repair iterations")

    source = module.read_text(encoding="utf-8")
    if "evaluate_batch(" in source:
        raise AssertionError("baseline generator must not evaluate candidates during generation")
    print("PASS: Phase 04 LLM-only baseline contract is locked and valid")


if __name__ == "__main__":
    main()
