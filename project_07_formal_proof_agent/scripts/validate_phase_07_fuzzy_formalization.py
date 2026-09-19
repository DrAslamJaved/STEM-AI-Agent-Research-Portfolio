"""Validate the static Phase 07 fuzzy-formalization contract before staging."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.fuzzy_formalization import (  # noqa: E402
    PINNED_LEAN_TOOLCHAIN,
    THEOREM_INVENTORY,
    build_formalization_manifest,
    source_is_ascii,
    source_path,
    static_policy_violations,
)


def require(text: str, expected: str, label: str) -> None:
    if expected not in text:
        raise AssertionError(f"missing {label}: {expected}")


def main() -> None:
    config = PROJECT_ROOT / "configs" / "phase_07_fuzzy_formalization.yaml"
    prompt = PROJECT_ROOT / "prompts" / "phase_07_fuzzy_formalization.md"
    lean_source_path = source_path(PROJECT_ROOT)
    compiler = PROJECT_ROOT / "scripts" / "compile_phase_07_lean.py"
    for path in (config, prompt, lean_source_path, compiler):
        if not path.is_file():
            raise AssertionError(f"required Phase 07 file is missing: {path}")

    configuration = config.read_text(encoding="utf-8")
    require(configuration, "development_split: valid", "development split")
    require(configuration, "test_split_locked: true", "locked test split")
    require(configuration, "primary_validator: independent_lean_compilation", "formal validator")
    require(configuration, PINNED_LEAN_TOOLCHAIN, "pinned Lean toolchain")
    require(configuration, "universe: singleton", "minimal formalization scope")
    require(configuration, "source_encoding: ASCII", "portable source encoding")
    require(configuration, "universal_identity_is_false", "counterexample claim")

    lean_source = lean_source_path.read_text(encoding="utf-8")
    violations = static_policy_violations(lean_source)
    if violations:
        raise AssertionError(f"Lean source contains prohibited token(s): {', '.join(violations)}")
    if not source_is_ascii(lean_source):
        raise AssertionError("Lean source must be ASCII for the pinned Windows Lean 3 compiler")
    for theorem_name in THEOREM_INVENTORY:
        require(lean_source, theorem_name, "registered Lean theorem")
    require(
        lean_source,
        "fuzzy_jaccard fuzzy_member.quarter fuzzy_member.half = similarity_value.one -> false",
        "exact counterexample",
    )

    manifest = build_formalization_manifest(PROJECT_ROOT)
    if manifest["formal_validity_decision"] != "independent_lean_compilation":
        raise AssertionError("formal validity policy changed")
    print("PASS: Phase 07 ASCII fuzzy formalization is locked and ready for pinned Lean compilation")


if __name__ == "__main__":
    main()
