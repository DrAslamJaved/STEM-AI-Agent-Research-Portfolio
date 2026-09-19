from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_harness import (  # noqa: E402
    Candidate,
    CompilerOutcome,
    Task,
    assert_split_is_allowed,
    evaluate_batch,
    evaluate_candidate,
    load_tasks_jsonl,
    static_rejection_reasons,
    write_evaluation_bundle,
)


class CountingCompiler:
    def __init__(self, passed: bool = True) -> None:
        self.calls = 0
        self.passed = passed

    def __call__(self, source: str) -> CompilerOutcome:
        self.calls += 1
        return CompilerOutcome(passed=self.passed, stdout=source, exit_code=0 if self.passed else 1)


class Phase03EvaluationHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.valid_task = Task("valid_refl", "valid", "example : (1 : Nat) = 1 :=")
        self.test_task = Task("test_refl", "test", "example : (1 : Nat) = 1 :=")

    def candidate(self, code: str, *, arm: str = "llm_only") -> Candidate:
        return Candidate("candidate_1", "valid_refl", arm, code, "phase_03_prompt")

    def test_development_mode_locks_test_split(self) -> None:
        with self.assertRaises(PermissionError):
            assert_split_is_allowed([self.test_task], "development")

    def test_final_evaluation_mode_locks_valid_split(self) -> None:
        with self.assertRaises(PermissionError):
            assert_split_is_allowed([self.valid_task], "final_evaluation")

    def test_static_scan_rejects_all_prohibited_shortcuts(self) -> None:
        self.assertEqual(static_rejection_reasons("by sorry\n-- admit\naxiom x : True"), ("admit", "axiom", "sorry"))

    def test_static_rejection_never_invokes_compiler(self) -> None:
        compiler = CountingCompiler()
        result = evaluate_candidate(self.valid_task, self.candidate("by admit"), compiler)
        self.assertEqual(result.status, "rejected_static")
        self.assertEqual(compiler.calls, 0)
        self.assertEqual(result.static_rejections, ("admit",))

    def test_clean_candidate_is_compiled_with_immutable_declaration(self) -> None:
        compiler = CountingCompiler()
        result = evaluate_candidate(self.valid_task, self.candidate("by rfl"), compiler)
        self.assertEqual(result.status, "compiled")
        self.assertEqual(compiler.calls, 1)
        self.assertIn("example : (1 : Nat) = 1 :=", result.compiler_stdout)
        self.assertIn("by rfl", result.compiler_stdout)

    def test_batch_rejects_unknown_theorem_id(self) -> None:
        unknown = Candidate("candidate_2", "not_in_manifest", "llm_sympy", "by rfl", "phase_03_prompt")
        with self.assertRaises(ValueError):
            evaluate_batch([self.valid_task], [unknown], CountingCompiler())

    def test_jsonl_manifest_and_bundle_capture_candidates_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "tasks.jsonl"
            manifest.write_text(
                json.dumps({"theorem_id": "valid_refl", "split": "valid", "declaration": "example : (1 : Nat) = 1 :="}) + "\n",
                encoding="utf-8",
            )
            tasks = load_tasks_jsonl(manifest)
            candidate = self.candidate("by rfl", arm="llm_lean_repair")
            results = evaluate_batch(tasks, [candidate], CountingCompiler(), mode="development")
            output = root / "evidence" / "bundle.json"
            write_evaluation_bundle(
                output,
                tasks=tasks,
                candidates=[candidate],
                results=results,
                mode="development",
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(payload["synthetic_demo"])
        self.assertEqual(payload["summary"], {"compiled": 1, "failed_compile": 0, "rejected_static": 0, "total": 1})
        self.assertEqual(payload["candidates"][0]["source_sha256"], candidate.source_sha256)
        self.assertTrue(payload["results"][0]["compile_passed"])

    def test_phase_03_configuration_locks_the_test_split(self) -> None:
        config = (PROJECT_ROOT / "configs" / "phase_03_evaluation_harness.yaml").read_text(encoding="utf-8")
        self.assertIn("test_split_locked: true", config)
        self.assertIn("final_evaluation_split: test", config)


if __name__ == "__main__":
    unittest.main()
