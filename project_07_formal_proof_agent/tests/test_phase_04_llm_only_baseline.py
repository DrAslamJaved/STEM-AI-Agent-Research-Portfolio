from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_harness import CompilerOutcome, Task, evaluate_batch  # noqa: E402
from formal_math.llm_only_baseline import (  # noqa: E402
    BaselineRunConfig,
    Completion,
    GenerationRequest,
    build_llm_only_prompt,
    calculate_baseline_metrics,
    generate_llm_only_candidates,
    normalise_proof_body,
    write_baseline_evidence,
    write_generation_records,
)


class RecordingProvider:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def complete(self, request: GenerationRequest) -> Completion:
        self.requests.append(request)
        responses = {
            ("refl", 0): "```lean\nby rfl\n```",
            ("refl", 1): "by exact 0",
            ("add_zero", 0): "by exact 0",
            ("add_zero", 1): "by simpa using Nat.add_zero 3",
        }
        return Completion(text=responses[(request.theorem_id, request.sample_index)], model_identifier="recording")


def fake_compiler(source: str) -> CompilerOutcome:
    passed = "by rfl" in source or "by simpa using Nat.add_zero 3" in source
    return CompilerOutcome(passed=passed, exit_code=0 if passed else 1)


class Phase04LLMOnlyBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = [
            Task("refl", "valid", "example : (2 : Nat) = 2 :="),
            Task("add_zero", "valid", "example : (3 : Nat) + 0 = 3 :="),
        ]
        self.config = BaselineRunConfig("unit-run", "unit-model", "phase_04_unit_prompt", 2, seed=7)

    def test_prompt_contains_frozen_theorem_but_no_prior_response(self) -> None:
        prompt = build_llm_only_prompt(self.tasks[0])
        self.assertIn("example : (2 : Nat) = 2 :=", prompt)
        self.assertIn("no feedback", prompt)
        self.assertNotIn("synthetic evaluator rejected candidate", prompt)

    def test_test_split_is_blocked_before_generation(self) -> None:
        test_task = Task("held_out", "test", "example : True :=")
        with self.assertRaises(PermissionError):
            generate_llm_only_candidates([test_task], RecordingProvider(), self.config)

    def test_generated_candidates_are_llm_only_and_have_no_repair_iteration(self) -> None:
        generated = generate_llm_only_candidates(self.tasks, RecordingProvider(), self.config)
        self.assertEqual(len(generated), 4)
        self.assertTrue(all(item.candidate.arm == "llm_only" for item in generated))
        self.assertTrue(all(item.candidate.iteration == 0 for item in generated))
        self.assertTrue(all(item.candidate.metadata["generation_feedback"] == "none" for item in generated))

    def test_provider_never_receives_a_previous_completion_or_evaluation(self) -> None:
        provider = RecordingProvider()
        generate_llm_only_candidates(self.tasks, provider, self.config)
        self.assertEqual(len(provider.requests), 4)
        for request in provider.requests:
            self.assertNotIn("synthetic evaluator rejected candidate", request.prompt)
            self.assertFalse(hasattr(request, "compiler_stderr"))

    def test_candidate_ids_are_stable_for_the_same_run(self) -> None:
        first = generate_llm_only_candidates(self.tasks, RecordingProvider(), self.config)
        second = generate_llm_only_candidates(self.tasks, RecordingProvider(), self.config)
        self.assertEqual([item.candidate.candidate_id for item in first], [item.candidate.candidate_id for item in second])

    def test_normalise_proof_body_removes_markdown_fence(self) -> None:
        self.assertEqual(normalise_proof_body("```lean\nby rfl\n```"), "by rfl")

    def test_normalise_proof_body_preserves_multiline_by_proof(self) -> None:
        raw = "Candidate proof:\nby\n  rfl"
        self.assertEqual(normalise_proof_body(raw), "by\n  rfl")

    def test_posthoc_metrics_report_compile_at_1_and_observed_pass_at_k(self) -> None:
        generated = generate_llm_only_candidates(self.tasks, RecordingProvider(), self.config)
        results = evaluate_batch(self.tasks, [item.candidate for item in generated], fake_compiler, mode="development")
        metrics = calculate_baseline_metrics(generated, results, ks=(1, 2))
        self.assertEqual(metrics["compile_at_1"], 0.5)
        self.assertEqual(metrics["observed_pass_at_k"], {"1": 0.5, "2": 1.0})

    def test_records_and_evidence_capture_generation_before_posthoc_results(self) -> None:
        generated = generate_llm_only_candidates(self.tasks, RecordingProvider(), self.config)
        results = evaluate_batch(self.tasks, [item.candidate for item in generated], fake_compiler, mode="development")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "candidates.jsonl"
            evidence = root / "baseline.json"
            write_generation_records(records, generated)
            write_baseline_evidence(
                evidence,
                config=self.config,
                generated=generated,
                results=results,
                synthetic_demo=True,
            )
            rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
            payload = json.loads(evidence.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 4)
        self.assertIn("raw_completion_sha256", rows[0])
        self.assertTrue(payload["generation_completed_before_evaluation"])
        self.assertEqual(payload["metrics"]["observed_pass_at_k"]["2"], 1.0)

    def test_phase_04_configuration_locks_no_feedback_generation(self) -> None:
        config = (PROJECT_ROOT / "configs" / "phase_04_llm_only_baseline.yaml").read_text(encoding="utf-8")
        self.assertIn("verification_feedback_during_generation: forbidden", config)
        self.assertIn("test_split_locked: true", config)


if __name__ == "__main__":
    unittest.main()
