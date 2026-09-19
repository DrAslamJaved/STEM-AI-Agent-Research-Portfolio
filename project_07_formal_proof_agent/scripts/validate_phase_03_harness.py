"""Validate Phase 03's repository-controlled evaluation-harness contract."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_harness import FORBIDDEN_SHORTCUTS, VALID_ARMS  # noqa: E402


def require(text: str, fragment: str, description: str) -> None:
    if fragment not in text:
        raise AssertionError(f"missing {description}: {fragment!r}")


def main() -> None:
    config = PROJECT_ROOT / "configs" / "phase_03_evaluation_harness.yaml"
    prompt = PROJECT_ROOT / "prompts" / "phase_03_candidate_format.md"
    module = PROJECT_ROOT / "src" / "formal_math" / "evaluation_harness.py"
    for path in (config, prompt, module):
        if not path.is_file():
            raise AssertionError(f"required Phase 03 file is missing: {path}")

    configuration = config.read_text(encoding="utf-8")
    require(configuration, "test_split_locked: true", "locked test-split policy")
    require(configuration, "development_split: valid", "development split")
    require(configuration, "final_evaluation_split: test", "final evaluation split")
    require(configuration, "independent_posthoc_lean_compilation", "primary validator")
    for arm in sorted(VALID_ARMS):
        require(configuration, f"- {arm}", f"generation arm {arm}")
    for shortcut in sorted(FORBIDDEN_SHORTCUTS):
        require(configuration, f"- {shortcut}", f"shortcut policy {shortcut}")

    print("PASS: Phase 03 evaluation-harness contract is locked and valid")


if __name__ == "__main__":
    main()
