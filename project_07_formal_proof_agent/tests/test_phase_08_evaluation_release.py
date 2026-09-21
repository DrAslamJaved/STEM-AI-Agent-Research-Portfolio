from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_release import (  # noqa: E402
    ARM_POLICY,
    EXPECTED_ARMS,
    FORMAL_VALIDITY_DECISION,
    FinalAttempt,
    build_release_bundle,
    calculate_arm_metrics,
    calculate_paired_comparisons,
    load_attempts_jsonl,
    validate_final_attempts,
    validate_release_manifest,
    write_attempts_jsonl,
    write_release_bundle,
    write_release_report,
)


def release_manifest(*, synthetic: bool = True, approved: bool = False) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "08",
        "run_id": "test-phase-08",
        "evaluation_split": "test",
        "test_split_locked": True,
        "formal_validity_decision": FORMAL_VALIDITY_DECISION,
        "samples_per_theorem": 1,
        "max_repair_iterations": 2,
        "model_identifier": "unit-test-model",
        "seed": 7,
        "synthetic_demo": synthetic,
        "arms": {arm: dict(policy) for arm, policy in ARM_POLICY.items()},
        "provenance": {
            "benchmark_commit": "f0dcc8b59e630fba00ba9569ca6714700e0a8801",
            "lean_toolchain": "leanprover-community/lean:3.42.1",
            "mathlib_revision": "cb2b02fff213ed6f65bebd64446baac64137dcda",
            "task_manifest_sha256": "a" * 64,
            "candidate_records_sha256": "b" * 64,
        },
        "human_review": {"approved": approved, "reviewer": "unit test"},
    }


def fixture_attempts() -> list[FinalAttempt]:
    records = [
        ("refl", "proof", "llm_only", 0, "compiled"),
        ("add_zero", "proof", "llm_only", 0, "failed_compile"),
        ("fuzzy_counterexample", "refutation", "llm_only", 0, "compiled"),
        ("refl", "proof", "llm_sympy", 0, "compiled"),
        ("add_zero", "proof", "llm_sympy", 0, "compiled"),
        ("fuzzy_counterexample", "refutation", "llm_sympy", 0, "compiled"),
        ("refl", "proof", "llm_lean_repair", 0, "compiled"),
        ("add_zero", "proof", "llm_lean_repair", 0, "failed_compile"),
        ("add_zero", "proof", "llm_lean_repair", 1, "compiled"),
        ("fuzzy_counterexample", "refutation", "llm_lean_repair", 0, "compiled"),
    ]
    return [
        FinalAttempt(
            candidate_id=f"{arm}-{theorem_id}-{iteration}",
            theorem_id=theorem_id,
            arm=arm,
            split="test",
            sample_index=1,
            iteration=iteration,
            status=status,
            kind=kind,
            elapsed_seconds=0.5,
            input_tokens=10,
            output_tokens=5,
            estimated_cost_usd=0.001,
        )
        for theorem_id, kind, arm, iteration, status in records
    ]


class Phase08EvaluationReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = release_manifest()
        self.attempts = fixture_attempts()

    def test_registers_exactly_the_three_study_arms(self) -> None:
        validate_release_manifest(self.manifest)
        self.assertEqual(set(self.manifest["arms"]), set(EXPECTED_ARMS))

    def test_rejects_unlocked_or_development_manifest(self) -> None:
        broken = release_manifest()
        broken["evaluation_split"] = "valid"
        with self.assertRaisesRegex(ValueError, "locked test split"):
            validate_release_manifest(broken)

        broken = release_manifest()
        broken["test_split_locked"] = False
        with self.assertRaisesRegex(ValueError, "test_split_locked"):
            validate_release_manifest(broken)

    def test_rejects_changed_arm_policy(self) -> None:
        broken = release_manifest()
        arms = broken["arms"]
        assert isinstance(arms, dict)
        arm = arms["llm_only"]
        assert isinstance(arm, dict)
        arm["lean_feedback"] = "allowed"
        with self.assertRaisesRegex(ValueError, "arms.llm_only.lean_feedback"):
            validate_release_manifest(broken)

    def test_rejects_malformed_provenance_digest(self) -> None:
        broken = release_manifest()
        provenance = broken["provenance"]
        assert isinstance(provenance, dict)
        provenance["candidate_records_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(ValueError, "candidate_records_sha256"):
            validate_release_manifest(broken)

    def test_requires_identical_theorem_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical theorem set"):
            validate_final_attempts(self.attempts[:-1], self.manifest)

    def test_rejects_development_records_and_unregistered_repairs(self) -> None:
        development = list(self.attempts)
        development[0] = FinalAttempt(
            **{**development[0].to_record(), "split": "valid"}
        )
        with self.assertRaisesRegex(PermissionError, "test-split"):
            validate_final_attempts(development, self.manifest)

        unregistered_repair = list(self.attempts)
        unregistered_repair[0] = FinalAttempt(
            **{**unregistered_repair[0].to_record(), "iteration": 1}
        )
        with self.assertRaisesRegex(ValueError, "must not contain repair iterations"):
            validate_final_attempts(unregistered_repair, self.manifest)

    def test_metrics_separate_initial_and_repaired_compilation(self) -> None:
        checked = validate_final_attempts(self.attempts, self.manifest)
        metrics = calculate_arm_metrics(checked, samples_per_theorem=1)
        self.assertEqual(metrics["llm_only"]["compile_at_1"], 0.666667)
        self.assertEqual(metrics["llm_lean_repair"]["compile_at_1"], 0.666667)
        self.assertEqual(
            metrics["llm_lean_repair"]["compile_after_allowed_workflow"], 1.0
        )
        self.assertEqual(metrics["llm_sympy"]["refutation_success_rate"], 1.0)
        self.assertEqual(metrics["llm_lean_repair"]["attempt_count"], 4)
        self.assertGreater(metrics["llm_only"]["estimated_cost_usd"], 0.0)

    def test_paired_comparison_is_by_theorem_identity(self) -> None:
        comparisons = calculate_paired_comparisons(self.attempts)
        sympy = comparisons["llm_sympy"]
        self.assertEqual(sympy["paired_outcomes"]["llm_sympy_only"], 1)
        self.assertEqual(sympy["paired_outcomes"]["both_compiled"], 2)
        self.assertEqual(sympy["discordant_pair_count"], 1)
        self.assertEqual(sympy["exact_mcnemar_two_sided_p_value"], 1.0)

    def test_synthetic_bundle_cannot_be_released(self) -> None:
        bundle = build_release_bundle(self.manifest, self.attempts)
        self.assertFalse(bundle["release_gate"]["release_ready"])
        self.assertIn("synthetic_demo=true", bundle["release_gate"]["blocking_conditions"][0])
        self.assertIn("Synthetic fixtures", bundle["evidence_boundary"])

    def test_real_bundle_requires_human_approval(self) -> None:
        not_approved = release_manifest(synthetic=False, approved=False)
        self.assertFalse(build_release_bundle(not_approved, self.attempts)["release_gate"]["release_ready"])

        approved = release_manifest(synthetic=False, approved=True)
        self.assertTrue(build_release_bundle(approved, self.attempts)["release_gate"]["release_ready"])

    def test_normalizes_nested_harness_record(self) -> None:
        attempt = FinalAttempt.from_record(
            {
                "candidate": {
                    "candidate_id": "nested-1",
                    "theorem_id": "refl",
                    "arm": "llm_only",
                    "sample_index": 1,
                    "iteration": 0,
                },
                "evaluation": {
                    "arm": "llm_only",
                    "split": "test",
                    "status": "compiled",
                    "elapsed_seconds": 1.25,
                    "static_rejections": [],
                },
                "input_tokens": 4,
                "output_tokens": 2,
                "estimated_cost_usd": 0.02,
            }
        )
        self.assertTrue(attempt.compiled)
        self.assertEqual(attempt.elapsed_seconds, 1.25)
        self.assertEqual(attempt.output_tokens, 2)

    def test_writers_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempts_path = root / "attempts.jsonl"
            bundle_path = root / "bundle.json"
            report_path = root / "report.md"
            write_attempts_jsonl(attempts_path, self.attempts)
            reloaded = load_attempts_jsonl(attempts_path)
            bundle = build_release_bundle(self.manifest, reloaded)
            write_release_bundle(bundle_path, bundle)
            write_release_report(report_path, bundle)
            persisted = json.loads(bundle_path.read_text(encoding="utf-8"))
            report = report_path.read_text(encoding="utf-8")
            self.assertEqual(persisted, bundle)
            self.assertIn("Synthetic smoke report", report)
            self.assertNotIn(b"\r\n", attempts_path.read_bytes())
            self.assertNotIn(b"\r\n", bundle_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
