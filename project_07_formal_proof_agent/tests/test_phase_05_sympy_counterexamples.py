from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_harness import CompilerOutcome, Task, evaluate_batch  # noqa: E402
from formal_math.sympy_counterexamples import (  # noqa: E402
    SympyCompletion,
    SympyGenerationRequest,
    SympyRunConfig,
    SympyContext,
    calculate_sympy_metrics,
    diagnose_symbolic_equality,
    exact_rational_grid,
    find_exact_counterexample,
    find_singleton_jaccard_identity_counterexample,
    generate_llm_sympy_candidates,
    singleton_jaccard,
    write_sympy_generation_records,
    write_sympy_workflow_evidence,
)


SYMPY_AVAILABLE = importlib.util.find_spec("sympy") is not None


class RecordingProvider:
    def __init__(self) -> None:
        self.requests: list[SympyGenerationRequest] = []

    def complete(self, request: SympyGenerationRequest) -> SympyCompletion:
        self.requests.append(request)
        responses = {
            ("identity", 0): "```lean\nby rfl\n```",
            ("identity", 1): "by rfl",
            ("fuzzy", 0): "by exact 0",
            ("fuzzy", 1): "by norm_num",
        }
        return SympyCompletion(text=responses[(request.theorem_id, request.iteration)], model_identifier="unit")


def fake_compiler(source: str) -> CompilerOutcome:
    passed = "by rfl" in source or "by norm_num" in source
    return CompilerOutcome(passed=passed, exit_code=0 if passed else 1)


class Phase05ContractTests(unittest.TestCase):
    def test_configuration_locks_sympy_only_generation(self) -> None:
        config = (PROJECT_ROOT / "configs" / "phase_05_sympy_counterexamples.yaml").read_text(encoding="utf-8")
        self.assertIn("arm: llm_sympy", config)
        self.assertIn("lean_feedback_during_generation: forbidden", config)
        self.assertIn("arithmetic: exact_rational_only", config)
        self.assertIn("test_split_locked: true", config)

    def test_sympy_dependency_is_declared(self) -> None:
        requirements = (PROJECT_ROOT / "requirements-phase05.txt").read_text(encoding="utf-8")
        self.assertIn("sympy", requirements.lower())

    def test_test_split_is_blocked_before_provider_invocation(self) -> None:
        provider = RecordingProvider()
        config = SympyRunConfig("unit", "unit", "prompt", 1)
        with self.assertRaises(PermissionError):
            generate_llm_sympy_candidates(
                [Task("held_out", "test", "example : True :=")],
                provider,
                config,
                {},
            )
        self.assertEqual(provider.requests, [])


@unittest.skipUnless(SYMPY_AVAILABLE, "SymPy is required; install requirements-phase05.txt")
class Phase05SympyWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = [
            Task("identity", "valid", "example : (2 : Nat) = 2 :="),
            Task("fuzzy", "valid", "example : (1 : Rat) / 4 < 1 / 2 :=", imports=("tactic",)),
        ]
        self.identity = diagnose_symbolic_equality(
            "binomial", "(x + y)**2", "x**2 + 2*x*y + y**2", variables=("x", "y")
        )
        self.witness = find_singleton_jaccard_identity_counterexample()
        self.contexts = {
            "identity": SympyContext(diagnostics=(self.identity,)),
            "fuzzy": SympyContext(counterexamples=(self.witness,)),
        }
        self.config = SympyRunConfig("unit-sympy", "unit-model", "phase_05_unit_prompt", 1, seed=7)

    def test_symbolic_identity_has_zero_residual(self) -> None:
        self.assertTrue(self.identity.equivalent_under_sympy_simplification)
        self.assertEqual(self.identity.residual, "0")
        self.assertEqual(self.identity.to_record()["formal_validity_decision"], "not_decided_by_sympy")

    def test_symbolic_non_identity_has_nonzero_residual(self) -> None:
        diagnostic = diagnose_symbolic_equality("bad_identity", "x + y", "x*y", variables=("x", "y"))
        self.assertFalse(diagnostic.equivalent_under_sympy_simplification)
        self.assertNotEqual(diagnostic.residual, "0")

    def test_exact_grid_contains_no_floats(self) -> None:
        grid = exact_rational_grid((1, 2, 3, 4), 4)
        self.assertEqual([str(value) for value in grid], ["1/4", "1/2", "3/4", "1"])
        self.assertTrue(all(value.is_Rational for value in grid))

    def test_singleton_jaccard_records_explicit_zero_convention(self) -> None:
        self.assertEqual(str(singleton_jaccard(0, 0)), "1")
        self.assertEqual(str(singleton_jaccard("1/4", "1/2")), "1/2")

    def test_exact_counterexample_refutes_false_nonzero_identity_claim(self) -> None:
        self.assertEqual(self.witness.assignments, {"a": "1/4", "b": "1/2"})
        self.assertEqual(self.witness.observed_value, "1/2")
        self.assertEqual(self.witness.to_record()["arithmetic"], "exact_rational")

    def test_missing_grid_witness_is_not_a_proof(self) -> None:
        witness = find_exact_counterexample(
            claim_id="always_true_on_grid",
            claim_description="A fixture claim that holds on the declared grid.",
            variables=("a",),
            grid=exact_rational_grid((1, 2), 2),
            claim_holds=lambda _values: True,
            observed_value=lambda _values: 0,
        )
        self.assertIsNone(witness)

    def test_provider_receives_sympy_context_but_no_lean_diagnostics(self) -> None:
        provider = RecordingProvider()
        generated = generate_llm_sympy_candidates(self.tasks, provider, self.config, self.contexts)
        self.assertEqual(len(generated), 4)
        self.assertTrue(all(item.candidate.arm == "llm_sympy" for item in generated))
        self.assertTrue(all(item.candidate.metadata["lean_feedback"] == "forbidden" for item in generated))
        for request in provider.requests:
            self.assertIn("SymPy-only diagnostic context", request.sympy_context)
            self.assertIn("no Lean compiler diagnostics", request.prompt)
            self.assertFalse(hasattr(request, "compiler_stderr"))

    def test_posthoc_metrics_and_evidence_preserve_generation_order(self) -> None:
        generated = generate_llm_sympy_candidates(self.tasks, RecordingProvider(), self.config, self.contexts)
        results = evaluate_batch(self.tasks, [item.candidate for item in generated], fake_compiler, mode="development")
        metrics = calculate_sympy_metrics(generated, results)
        self.assertEqual(metrics["compile_at_1"], 0.5)
        self.assertEqual(metrics["compile_after_sympy_repair"], 1.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "candidates.jsonl"
            evidence = root / "workflow.json"
            write_sympy_generation_records(records, generated)
            write_sympy_workflow_evidence(
                evidence,
                config=self.config,
                generated=generated,
                results=results,
                synthetic_demo=True,
            )
            rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
            payload = json.loads(evidence.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 4)
        self.assertIn("sympy_context_sha256", rows[0])
        self.assertTrue(payload["generation_completed_before_evaluation"])
        self.assertEqual(payload["metrics"]["compile_after_sympy_repair"], 1.0)


if __name__ == "__main__":
    unittest.main()
