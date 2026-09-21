"""Validate the Phase 09 real-study implementation without calling a model."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.live_study import ARM_POLICY, EXPECTED_ARMS, PINNED_MINIF2F_COMMIT  # noqa: E402


def require(text: str, phrase: str, label: str) -> None:
    if phrase not in text:
        raise AssertionError(f"missing {label}: {phrase}")


def main() -> None:
    required = (
        PROJECT_ROOT / "configs" / "phase_09_live_study.yaml",
        PROJECT_ROOT / "docs" / "phase_09_live_study.md",
        PROJECT_ROOT / "PHASE_09_INSTALL.md",
        PROJECT_ROOT / "prompts" / "phase_09_live_study_protocol.md",
        PROJECT_ROOT / "src" / "formal_math" / "live_study.py",
        PROJECT_ROOT / "scripts" / "prepare_phase_09_tasks.py",
        PROJECT_ROOT / "scripts" / "preregister_phase_09_study.py",
        PROJECT_ROOT / "scripts" / "run_phase_09_live_study.py",
        PROJECT_ROOT / "scripts" / "record_phase_09_human_review.py",
        PROJECT_ROOT / "scripts" / "provision_phase_09_canonical_runtime.ps1",
        PROJECT_ROOT / "tests" / "test_phase_09_live_study.py",
    )
    for path in required:
        if not path.is_file():
            raise AssertionError(f"required Phase 09 file is missing: {path}")

    config = (PROJECT_ROOT / "configs" / "phase_09_live_study.yaml").read_text(encoding="utf-8")
    require(config, PINNED_MINIF2F_COMMIT, "frozen miniF2F commit")
    require(config, "development_split: valid", "development split")
    require(config, "final_split: test", "locked final split")
    require(config, "test_split_requires_explicit_confirmation: true", "test confirmation")
    require(config, "human_review_required_before_release: true", "human review gate")
    for arm in EXPECTED_ARMS:
        require(config, f"  {arm}:", f"registered arm {arm}")
    source = (PROJECT_ROOT / "src" / "formal_math" / "live_study.py").read_text(encoding="utf-8")
    for token in ("extract_minif2f_tasks", "OpenAIResponsesProvider", "run_live_study", "final_release_manifest"):
        require(source, token, f"live-study capability {token}")
    if ARM_POLICY["llm_lean_repair"]["max_repair_iterations"] != 2:
        raise AssertionError("Phase 09 must retain the registered two-iteration Lean repair budget")
    print("PASS: Phase 09 real-study protocol is installed; no model call was made.")


if __name__ == "__main__":
    main()
