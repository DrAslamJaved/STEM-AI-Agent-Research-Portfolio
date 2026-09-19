from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_harness import CompilerOutcome, Task  # noqa: E402
from formal_math.lean_repair import (  # noqa: E402
    InitialGenerationRequest,
    LeanCompletion,
    LeanRepairRequest,
    LeanRepairRunConfig,
    PINNED_LEAN_TOOLCHAIN,
    build_initial_prompt,
    calculate_lean_repair_metrics,
    generate_verify_repair,
    make_pinned_lean3_compiler,
    write_lean_repair_attempt_records,
    write_lean_repair_evidence,
)


class RecordingProvider:
    def __init__(self, *, initial_add_zero: str = "by exact 0", repair_add_zero: str = "by simpa using Nat.add_zero 3") -> None:
        self.initial_add_zero = initial_add_zero
        self.repair_add_zero = repair_add_zero
        self.initial_requests: list[InitialGenerationRequest] = []
        self.repair_requests: list[LeanRepairRequest] = []

    def complete_initial(self, request: InitialGenerationRequest) -> LeanCompletion:
        self.initial_requests.append(request)
        responses = {
            "refl": "```lean\nby rfl\n```",
            "add_zero": self.initial_add_zero,
        }
        return LeanCompletion(text=responses[request.theorem_id], model_identifier="unit-provider")

    def complete_repair(self, request: LeanRepairRequest) -> LeanCompletion:
        self.repair_requests.append(request)
        return LeanCompletion(text=self.repair_add_zero, model_identifier="unit-provider")


class CountingCompiler:
    def __init__(self) -> None:
        self.sources: list[str] = []

    def __call__(self, source: str) -> CompilerOutcome:
        self.sources.append(source)
        accepted = "by rfl" in source or "by simpa using Nat.add_zero 3" in source
        return CompilerOutcome(
            passed=accepted,
            stdout="accepted" if accepted else "",
            stderr="synthetic Lean diagnostic: proof body did not typecheck" if not accepted else "",
            exit_code=0 if accepted else 1,
        )


class Phase06LeanRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = [
            Task("refl", "valid", "example : (2 : Nat) = 2 :="),
            Task("add_zero", "valid", "example : (3 : Nat) + 0 = 3 :="),
        ]
        self.config = LeanRepairRunConfig("unit-run", "unit-model", "phase_06_unit_prompt", 1, seed=7)

    def test_configuration_locks_bounded_lean_repair(self) -> None:
        config = (PROJECT_ROOT / "configs" / "phase_06_lean_repair.yaml").read_text(encoding="utf-8")
        self.assertIn("arm: llm_lean_repair", config)
        self.assertIn("theorem_declaration_edits: forbidden", config)
        self.assertIn("max_repair_iterations: 2", config)
        self.assertIn("test_split_locked: true", config)

    def test_initial_prompt_has_frozen_task_but_no_diagnostics(self) -> None:
        prompt = build_initial_prompt(self.tasks[0])
        self.assertIn("example : (2 : Nat) = 2 :=", prompt)
        self.assertIn("no compiler feedback", prompt)
        self.assertNotIn("Lean stderr", prompt)

    def test_test_split_is_blocked_before_provider_invocation(self) -> None:
        provider = RecordingProvider()
        with self.assertRaises(PermissionError):
            generate_verify_repair(
                [Task("held_out", "test", "example : True :=")],
                provider,
                self.config,
                CountingCompiler(),
            )
        self.assertEqual(provider.initial_requests, [])
        self.assertEqual(provider.repair_requests, [])

    def test_pinned_compiler_factory_uses_lean_3_42_1(self) -> None:
        compiler = make_pinned_lean3_compiler(Path(__file__), cwd=PROJECT_ROOT, timeout_seconds=12)
        self.assertEqual(
            compiler.command,
            (str(Path(__file__)), "run", PINNED_LEAN_TOOLCHAIN, "lean"),
        )
        self.assertEqual(compiler.cwd, PROJECT_ROOT)
        self.assertEqual(compiler.timeout_seconds, 12)

    def test_compiled_initial_attempt_does_not_trigger_repair(self) -> None:
        provider = RecordingProvider()
        attempts = generate_verify_repair([self.tasks[0]], provider, self.config, CountingCompiler())
        self.assertEqual(len(attempts), 1)
        self.assertTrue(attempts[0].evaluation.compile_passed)
        self.assertEqual(provider.repair_requests, [])

    def test_failed_attempt_passes_independent_diagnostic_to_repair(self) -> None:
        provider = RecordingProvider()
        attempts = generate_verify_repair([self.tasks[1]], provider, self.config, CountingCompiler())
        self.assertEqual(len(attempts), 2)
        request = provider.repair_requests[0]
        self.assertEqual(request.feedback.status, "failed_compile")
        self.assertEqual(request.feedback.compiler_exit_code, 1)
        self.assertIn("did not typecheck", request.feedback.compiler_stderr)
        self.assertIn("Previous Lean proof body", request.prompt)
        self.assertIn("by exact 0", request.prompt)

    def test_repaired_attempt_compiles_and_records_parent_candidate(self) -> None:
        attempts = generate_verify_repair([self.tasks[1]], RecordingProvider(), self.config, CountingCompiler())
        self.assertFalse(attempts[0].evaluation.compile_passed)
        self.assertTrue(attempts[1].evaluation.compile_passed)
        self.assertEqual(attempts[1].candidate.iteration, 1)
        self.assertEqual(attempts[1].candidate.metadata["prior_candidate_id"], attempts[0].candidate.candidate_id)

    def test_repair_budget_stops_after_configured_limit(self) -> None:
        provider = RecordingProvider(repair_add_zero="by exact 0")
        attempts = generate_verify_repair([self.tasks[1]], provider, self.config, CountingCompiler())
        self.assertEqual(len(attempts), 2)
        self.assertFalse(attempts[-1].evaluation.compile_passed)
        self.assertEqual(len(provider.repair_requests), 1)

    def test_static_rejection_is_recorded_before_repair(self) -> None:
        provider = RecordingProvider(initial_add_zero="by sorry")
        compiler = CountingCompiler()
        attempts = generate_verify_repair([self.tasks[1]], provider, self.config, compiler)
        self.assertEqual(attempts[0].evaluation.status, "rejected_static")
        self.assertEqual(attempts[1].feedback.static_rejections, ("sorry",))
        self.assertEqual(len(compiler.sources), 1)
        self.assertTrue(attempts[1].evaluation.compile_passed)

    def test_candidate_ids_are_stable_for_same_run(self) -> None:
        first = generate_verify_repair(self.tasks, RecordingProvider(), self.config, CountingCompiler())
        second = generate_verify_repair(self.tasks, RecordingProvider(), self.config, CountingCompiler())
        self.assertEqual(
            [attempt.candidate.candidate_id for attempt in first],
            [attempt.candidate.candidate_id for attempt in second],
        )

    def test_metrics_report_initial_and_post_repair_compile_rates(self) -> None:
        attempts = generate_verify_repair(self.tasks, RecordingProvider(), self.config, CountingCompiler())
        metrics = calculate_lean_repair_metrics(attempts, max_repair_iterations=1)
        self.assertEqual(metrics["compile_at_1"], 0.5)
        self.assertEqual(metrics["compile_after_lean_repair"], 1.0)
        self.assertEqual(metrics["repaired_theorem_count"], 1)
        self.assertEqual(metrics["mean_repair_iterations_executed"], 0.5)

    def test_attempt_records_and_evidence_preserve_compiler_feedback(self) -> None:
        attempts = generate_verify_repair(self.tasks, RecordingProvider(), self.config, CountingCompiler())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "attempts.jsonl"
            evidence = root / "lean_repair.json"
            write_lean_repair_attempt_records(records, attempts)
            write_lean_repair_evidence(
                evidence,
                config=self.config,
                attempts=attempts,
                synthetic_demo=True,
            )
            rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
            payload = json.loads(evidence.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 3)
        self.assertIsNone(rows[0]["repair_feedback"])
        self.assertIn("compiler_stderr", rows[-1]["repair_feedback"])
        self.assertEqual(payload["metrics"]["compile_after_lean_repair"], 1.0)
        self.assertEqual(payload["pinned_toolchain"]["lean_toolchain"], PINNED_LEAN_TOOLCHAIN)


if __name__ == "__main__":
    unittest.main()
