from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.fuzzy_formalization import (  # noqa: E402
    PINNED_LEAN_TOOLCHAIN,
    PINNED_MATHLIB_REVISION,
    PINNED_MINIF2F_COMMIT,
    THEOREM_INVENTORY,
    build_formalization_manifest,
    build_lean_compile_command,
    is_formally_valid,
    sha256_file,
    source_is_ascii,
    source_path,
    static_policy_violations,
    validate_compile_evidence,
    write_formalization_manifest,
)


class Phase07FuzzyFormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lean_source_path = source_path(PROJECT_ROOT)
        self.lean_source = self.lean_source_path.read_text(encoding="utf-8")

    def compile_evidence(self, **overrides: object) -> dict[str, object]:
        evidence: dict[str, object] = {
            "formal_validity_decision": "independent_lean_compilation",
            "benchmark_commit": PINNED_MINIF2F_COMMIT,
            "lean_toolchain": PINNED_LEAN_TOOLCHAIN,
            "mathlib_revision": PINNED_MATHLIB_REVISION,
            "source_sha256": sha256_file(self.lean_source_path),
            "status": "compiled",
            "compilation_passed": True,
            "compiler_exit_code": 0,
        }
        evidence.update(overrides)
        return evidence

    def test_configuration_locks_pinned_toolchain_and_test_split(self) -> None:
        config = (PROJECT_ROOT / "configs" / "phase_07_fuzzy_formalization.yaml").read_text(encoding="utf-8")
        self.assertIn(PINNED_MINIF2F_COMMIT, config)
        self.assertIn(PINNED_LEAN_TOOLCHAIN, config)
        self.assertIn(PINNED_MATHLIB_REVISION, config)
        self.assertIn("development_split: valid", config)
        self.assertIn("test_split_locked: true", config)

    def test_configuration_declares_independent_lean_as_primary_validator(self) -> None:
        config = (PROJECT_ROOT / "configs" / "phase_07_fuzzy_formalization.yaml").read_text(encoding="utf-8")
        self.assertIn("primary_validator: independent_lean_compilation", config)
        self.assertIn("generated_manifest_is_not_proof_evidence: true", config)

    def test_lean_source_contains_every_registered_theorem(self) -> None:
        for theorem_name in THEOREM_INVENTORY:
            self.assertIn(theorem_name, self.lean_source)

    def test_lean_source_uses_explicit_zero_union_convention(self) -> None:
        self.assertIn("fuzzy_member.zero fuzzy_member.zero := similarity_value.one", self.lean_source)
        self.assertIn("fuzzy_jaccard_zero_zero", self.lean_source)

    def test_lean_source_contains_exact_grid_counterexample(self) -> None:
        self.assertIn(
            "fuzzy_jaccard fuzzy_member.quarter fuzzy_member.half = similarity_value.half",
            self.lean_source,
        )
        self.assertIn(
            "fuzzy_jaccard fuzzy_member.quarter fuzzy_member.half = similarity_value.one -> false",
            self.lean_source,
        )

    def test_lean_source_is_ascii_for_pinned_windows_compiler(self) -> None:
        self.assertTrue(source_is_ascii(self.lean_source))
        self.assertNotRegex(self.lean_source, r"(?m)^import\s")

    def test_lean_source_has_no_prohibited_shortcuts(self) -> None:
        self.assertEqual(static_policy_violations(self.lean_source), ())

    def test_static_policy_matches_whole_tokens_only(self) -> None:
        self.assertEqual(static_policy_violations("-- axioms are discussed"), ())
        self.assertEqual(static_policy_violations("by sorry"), ("sorry",))
        self.assertEqual(static_policy_violations("by admit"), ("admit",))
        self.assertEqual(static_policy_violations("axiom unsound : False"), ("axiom",))

    def test_manifest_matches_registered_source_digest(self) -> None:
        manifest = json.loads(
            (PROJECT_ROOT / "results" / "phase_07_formalization_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["source"]["path"], "formalizations/lean3/FuzzySimilarity.lean")
        self.assertEqual(manifest["source"]["sha256"], sha256_file(self.lean_source_path))
        self.assertEqual(manifest["theorem_inventory"], list(THEOREM_INVENTORY))

    def test_manifest_declares_counterexample_not_performance_metric(self) -> None:
        manifest = build_formalization_manifest(PROJECT_ROOT)
        self.assertEqual(manifest["counterexample"]["witness"]["a"], "1/4")
        self.assertEqual(manifest["counterexample"]["witness"]["value"], "1/2")
        self.assertEqual(manifest["formal_validity_decision"], "independent_lean_compilation")
        self.assertNotIn("pass_at_k", manifest)

    def test_manifest_writer_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            written = write_formalization_manifest(output, PROJECT_ROOT)
            persisted = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(persisted, written)
        self.assertEqual(written, build_formalization_manifest(PROJECT_ROOT))

    def test_pinned_compile_command_is_exact(self) -> None:
        command = build_lean_compile_command(Path("C:/Users/example/.elan/bin/elan.exe"), self.lean_source_path)
        self.assertEqual(command[1:4], ("run", PINNED_LEAN_TOOLCHAIN, "lean"))
        self.assertEqual(command[-1], str(self.lean_source_path))

    def test_successful_source_matched_compile_is_formally_valid(self) -> None:
        evidence = self.compile_evidence()
        validate_compile_evidence(evidence, PROJECT_ROOT)
        self.assertTrue(is_formally_valid(evidence, PROJECT_ROOT))

    def test_compile_evidence_rejects_mismatched_source_or_pins(self) -> None:
        for override in (
            {"source_sha256": "0" * 64},
            {"benchmark_commit": "unregistered"},
            {"lean_toolchain": "leanprover/lean4:stable"},
        ):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    validate_compile_evidence(self.compile_evidence(**override), PROJECT_ROOT)
                self.assertFalse(is_formally_valid(self.compile_evidence(**override), PROJECT_ROOT))

    def test_compiled_status_requires_true_and_zero_exit_code(self) -> None:
        with self.assertRaises(ValueError):
            validate_compile_evidence(
                self.compile_evidence(compilation_passed=True, compiler_exit_code=1), PROJECT_ROOT
            )
        with self.assertRaises(ValueError):
            validate_compile_evidence(
                self.compile_evidence(compilation_passed=False, status="compiled", compiler_exit_code=0),
                PROJECT_ROOT,
            )

    def test_failed_compilation_is_not_formally_valid(self) -> None:
        evidence = self.compile_evidence(
            compilation_passed=False,
            status="failed_compile",
            compiler_exit_code=1,
        )
        validate_compile_evidence(evidence, PROJECT_ROOT)
        self.assertFalse(is_formally_valid(evidence, PROJECT_ROOT))

    def test_compiler_runner_uses_subprocess_and_records_source_digest(self) -> None:
        runner = (PROJECT_ROOT / "scripts" / "compile_phase_07_lean.py").read_text(encoding="utf-8")
        self.assertIn("subprocess.run(", runner)
        self.assertIn("pinned_commit(lean_cwd)", runner)
        self.assertIn("source_sha256", runner)
        self.assertIn("compiler_cwd_kind", runner)
        self.assertIn("source_is_ascii", runner)

    def test_prompt_forbids_theorem_edits_and_shortcuts(self) -> None:
        prompt = (PROJECT_ROOT / "prompts" / "phase_07_fuzzy_formalization.md").read_text(encoding="utf-8")
        self.assertIn("Do not change its definition, imports, hypotheses, theorem statement", prompt)
        self.assertIn("Do not use sorry, admit, or axiom", prompt)
        self.assertIn("registered exact-grid witness", prompt)
        self.assertIn("ASCII Lean 3 syntax only", prompt)


if __name__ == "__main__":
    unittest.main()
