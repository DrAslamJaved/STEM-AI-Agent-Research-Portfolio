"""Provenance helpers for Project 07 Phase 07 fuzzy formalization evidence.

The Lean source is the formal object.  This Python module deliberately does
not infer mathematical validity: it records source integrity, creates the
pinned compiler command, and validates compiler evidence produced elsewhere.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping


PINNED_MINIF2F_COMMIT = "f0dcc8b59e630fba00ba9569ca6714700e0a8801"
PINNED_LEAN_TOOLCHAIN = "leanprover-community/lean:3.42.1"
PINNED_MATHLIB_REVISION = "cb2b02fff213ed6f65bebd64446baac64137dcda"
LEAN_SOURCE_RELATIVE_PATH = Path("formalizations/lean3/FuzzySimilarity.lean")
THEOREM_INVENTORY = (
    "fuzzy_jaccard_zero_zero",
    "fuzzy_jaccard_refl",
    "fuzzy_jaccard_symm",
    "quarter_has_exact_value",
    "half_has_exact_value",
    "fuzzy_jaccard_counterexample_value",
    "fuzzy_jaccard_counterexample_not_one",
)
FORBIDDEN_LEAN_TOKENS = ("sorry", "admit", "axiom")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without altering it."""
    return sha256(path.read_bytes()).hexdigest()


def source_path(project_root: Path) -> Path:
    """Return the registered Lean source path for this phase."""
    return project_root / LEAN_SOURCE_RELATIVE_PATH


def static_policy_violations(lean_source: str) -> tuple[str, ...]:
    """Detect prohibited proof shortcuts before invoking Lean.

    This is intentionally a simple lexical safeguard.  It does not establish
    theorem validity; that decision belongs exclusively to Lean compilation.
    """
    lowered = lean_source.lower()
    return tuple(
        token
        for token in FORBIDDEN_LEAN_TOKENS
        if re.search(rf"\b{re.escape(token)}\b", lowered)
    )


def source_is_ascii(lean_source: str) -> bool:
    """Require a portable ASCII Lean source for the pinned Windows compiler."""
    return lean_source.isascii()


def build_lean_compile_command(elan_path: Path, lean_source_path: Path) -> tuple[str, ...]:
    """Build the exact Phase 02-pinned Lean command for this source."""
    return (
        str(elan_path),
        "run",
        PINNED_LEAN_TOOLCHAIN,
        "lean",
        str(lean_source_path),
    )


def build_formalization_manifest(project_root: Path) -> dict[str, Any]:
    """Build compact, non-performance provenance for the Lean source."""
    lean_source_path = source_path(project_root)
    if not lean_source_path.is_file():
        raise FileNotFoundError(f"registered Lean source is missing: {lean_source_path}")
    source_text = lean_source_path.read_text(encoding="utf-8")
    violations = static_policy_violations(source_text)
    if violations:
        raise ValueError(f"registered Lean source violates static policy: {', '.join(violations)}")
    if not source_is_ascii(source_text):
        raise ValueError("registered Lean source must remain ASCII for the pinned Windows Lean 3 runner")
    return {
        "schema_version": 1,
        "phase": "07",
        "scope": "singleton-universe exact fuzzy Jaccard table on the denominator-four membership grid",
        "formal_validity_decision": "independent_lean_compilation",
        "source": {
            "path": LEAN_SOURCE_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(lean_source_path),
        },
        "pinned_toolchain": {
            "benchmark_commit": PINNED_MINIF2F_COMMIT,
            "lean_toolchain": PINNED_LEAN_TOOLCHAIN,
            "mathlib_revision": PINNED_MATHLIB_REVISION,
        },
        "theorem_inventory": list(THEOREM_INVENTORY),
        "counterexample": {
            "false_claim": "every registered fuzzy-member pair has similarity 1",
            "witness": {"a": "1/4", "b": "1/2", "value": "1/2"},
            "certification": "Lean theorem fuzzy_jaccard_counterexample_not_one",
        },
        "compile_evidence": "results/phase_07_lean_compile.json is created only by the pinned compiler runner",
    }


def write_formalization_manifest(path: Path, project_root: Path) -> dict[str, Any]:
    """Write a deterministic source manifest for audit before compilation."""
    payload = build_formalization_manifest(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_compile_evidence(evidence: Mapping[str, Any], project_root: Path) -> None:
    """Reject evidence that is not tied to the registered source and pins."""
    source_digest = sha256_file(source_path(project_root))
    required = {
        "benchmark_commit": PINNED_MINIF2F_COMMIT,
        "lean_toolchain": PINNED_LEAN_TOOLCHAIN,
        "mathlib_revision": PINNED_MATHLIB_REVISION,
        "source_sha256": source_digest,
    }
    for key, expected in required.items():
        actual = evidence.get(key)
        if actual != expected:
            raise ValueError(f"compile evidence {key} mismatch: expected {expected!r}, got {actual!r}")
    if evidence.get("formal_validity_decision") != "independent_lean_compilation":
        raise ValueError("compile evidence does not use independent Lean compilation")
    if evidence.get("compilation_passed") is True:
        if evidence.get("status") != "compiled" or evidence.get("compiler_exit_code") != 0:
            raise ValueError("a passing compilation record must have status=compiled and exit code 0")
    elif evidence.get("status") == "compiled":
        raise ValueError("compiled status requires compilation_passed=true")


def is_formally_valid(evidence: Mapping[str, Any], project_root: Path) -> bool:
    """Return true only for a source-matched successful independent compilation."""
    try:
        validate_compile_evidence(evidence, project_root)
    except ValueError:
        return False
    return evidence.get("compilation_passed") is True
