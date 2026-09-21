from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_harness import CompilerOutcome, Task  # noqa: E402
from formal_math.evaluation_release import FinalAttempt, calculate_arm_metrics  # noqa: E402
from formal_math.live_study import (  # noqa: E402
    LiveStudyPlan,
    OpenAIResponsesProvider,
    PINNED_LEAN_TOOLCHAIN,
    PINNED_MATHLIB_REVISION,
    PINNED_MINIF2F_COMMIT,
    ProviderCompletion,
    StudyTask,
    build_fuzzy_study_tasks,
    extract_minif2f_tasks,
    load_task_manifest,
    load_study_plan,
    run_live_study,
    sha256_file,
    validate_plan_for_tasks,
    write_real_run_bundle,
    write_study_plan,
    write_task_manifest,
)


class ScriptedProvider:
    def __init__(self, model_identifier: str) -> None:
        self.model_identifier = model_identifier
        self.calls: list[str] = []

    def complete(self, *, prompt: str, seed: int | None, request_label: str) -> ProviderCompletion:
        self.calls.append(request_label)
        if ":llm_lean_repair:" in request_label and request_label.endswith(":0"):
            text = "by exact 0"
        else:
            text = "by rfl"
        return ProviderCompletion(
            text=text,
            model_identifier=self.model_identifier,
            input_tokens=12,
            output_tokens=4,
            response_id=f"response-{len(self.calls)}",
        )


class TrackingCompiler:
    def __init__(self, provider: ScriptedProvider) -> None:
        self.provider = provider
        self.provider_call_counts: list[int] = []

    def __call__(self, source: str) -> CompilerOutcome:
        self.provider_call_counts.append(len(self.provider.calls))
        passed = "by rfl" in source
        return CompilerOutcome(
            passed=passed,
            stdout="" if passed else "",
            stderr="" if passed else "synthetic type error",
            exit_code=0 if passed else 1,
        )


def study_task(*, split: str = "valid") -> StudyTask:
    return StudyTask(
        task=Task(
            theorem_id="study_refl",
            split=split,
            declaration="theorem study_refl : (2 : Nat) = 2 :=",
            source="unit-test",
        ),
        kind="proof",
        source_path="unit-test.lean",
        source_sha256="a" * 64,
    )


def plan_for(tasks_path: Path, *, split: str = "valid") -> LiveStudyPlan:
    return LiveStudyPlan(
        run_id="phase09-unit-test",
        evaluation_mode="final_evaluation" if split == "test" else "development",
        model_identifier="unit-test-model",
        samples_per_theorem=1,
        base_seed=17,
        max_repair_iterations=2,
        task_manifest_sha256=sha256_file(tasks_path),
        benchmark_commit=PINNED_MINIF2F_COMMIT,
        lean_toolchain=PINNED_LEAN_TOOLCHAIN,
        mathlib_revision=PINNED_MATHLIB_REVISION,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_estimated_cost_usd=1.0,
        send_seed_to_provider=False,
    )


class Phase09LiveStudyTests(unittest.TestCase):
    def test_task_preamble_is_compiled_but_declaration_remains_frozen(self) -> None:
        task = Task(
            theorem_id="with_preamble",
            split="valid",
            imports=("minif2f_import",),
            preamble="def project07_helper : Nat := 2",
            declaration="theorem with_preamble : project07_helper = 2 :=",
        )
        source = task.render_source("by rfl")
        self.assertIn("def project07_helper", source)
        self.assertIn("theorem with_preamble", source)
        self.assertTrue(source.endswith("by rfl\n"))

    def test_extractor_strips_reference_proof_bodies_and_selects_reproducibly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "lean" / "src"
            source_path.mkdir(parents=True)
            statements = "\n".join(
                f"theorem theorem_{index} : True :=\nbegin\n  trivial\nend\n"
                for index in range(244)
            )
            (source_path / "valid.lean").write_text(
                "import minif2f_import\n\nopen_locale big_operators\n\n" + statements,
                encoding="utf-8",
            )
            first = extract_minif2f_tasks(root, split="valid", limit=3, selection_seed=4)
            second = extract_minif2f_tasks(root, split="valid", limit=3, selection_seed=4)
            self.assertEqual([item.task.theorem_id for item in first], [item.task.theorem_id for item in second])
            self.assertEqual(len(first), 3)
            self.assertTrue(all("trivial" not in item.task.declaration for item in first))

    def test_fuzzy_suite_contains_a_refutation_without_preloaded_theorem_answers(self) -> None:
        tasks = build_fuzzy_study_tasks(PROJECT_ROOT, split="test")
        self.assertEqual(len(tasks), 4)
        refutations = [task for task in tasks if task.kind == "refutation"]
        self.assertEqual(len(refutations), 1)
        self.assertIn("quarter", refutations[0].task.declaration)
        self.assertNotIn("fuzzy_jaccard_refl", refutations[0].task.preamble)

    def test_task_and_plan_writers_are_lf_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks_path = root / "tasks.jsonl"
            write_task_manifest(tasks_path, [study_task()])
            self.assertNotIn(b"\r\n", tasks_path.read_bytes())
            plan = plan_for(tasks_path)
            plan_path = root / "plan.json"
            write_study_plan(plan_path, plan)
            self.assertNotIn(b"\r\n", plan_path.read_bytes())
            self.assertEqual(load_task_manifest(tasks_path)[0].task.theorem_id, "study_refl")
            self.assertEqual(load_study_plan(plan_path), plan)

    def test_plan_rejects_a_split_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tasks_path = Path(directory) / "tasks.jsonl"
            write_task_manifest(tasks_path, [study_task(split="test")])
            development_plan = plan_for(tasks_path, split="valid")
            with self.assertRaisesRegex(PermissionError, "requires only 'valid'"):
                validate_plan_for_tasks(development_plan, load_task_manifest(tasks_path))

    def test_three_arms_preserve_closed_generation_and_bounded_lean_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tasks_path = Path(directory) / "tasks.jsonl"
            write_task_manifest(tasks_path, [study_task()])
            tasks = load_task_manifest(tasks_path)
            provider = ScriptedProvider("unit-test-model")
            compiler = TrackingCompiler(provider)
            journal_path = Path(directory) / "generation_journal.jsonl"
            records, ledger = run_live_study(
                tasks,
                provider=provider,
                compiler=compiler,
                plan=plan_for(tasks_path),
                generation_journal_path=journal_path,
            )
            self.assertEqual(len(records), 4)
            self.assertEqual({record["candidate"]["arm"] for record in records}, {
                "llm_only", "llm_sympy", "llm_lean_repair"
            })
            lean_records = [record for record in records if record["candidate"]["arm"] == "llm_lean_repair"]
            self.assertEqual([record["iteration"] for record in lean_records], [0, 1])
            self.assertTrue(lean_records[-1]["evaluation"]["compile_passed"])
            # The baseline has one task/sample and must finish generation before
            # its first compiler invocation. SymPy likewise has no Lean feedback.
            self.assertGreaterEqual(compiler.provider_call_counts[0], 1)
            self.assertGreaterEqual(compiler.provider_call_counts[1], 2)
            self.assertGreater(ledger.total_estimated_cost_usd, 0.0)
            journal_lines = journal_path.read_bytes().splitlines()
            self.assertEqual(len(journal_lines), 4)
            self.assertNotIn(b"\r\n", journal_path.read_bytes())
            attempts = [FinalAttempt.from_record(record) for record in records]
            metrics = calculate_arm_metrics(attempts, samples_per_theorem=1)
            self.assertEqual(metrics["llm_lean_repair"]["compile_after_allowed_workflow"], 1.0)
            self.assertGreater(metrics["llm_only"]["total_generation_seconds"], -1.0)

    def test_development_and_final_bundles_keep_their_evidence_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_tasks_path = root / "valid_tasks.jsonl"
            write_task_manifest(valid_tasks_path, [study_task()])
            provider = ScriptedProvider("unit-test-model")
            records, ledger = run_live_study(
                load_task_manifest(valid_tasks_path),
                provider=provider,
                compiler=TrackingCompiler(provider),
                plan=plan_for(valid_tasks_path),
            )
            development_artifacts = write_real_run_bundle(
                root / "development_run",
                plan=plan_for(valid_tasks_path),
                tasks_path=valid_tasks_path,
                records=records,
                ledger=ledger,
            )
            development = json.loads(development_artifacts["development_evidence"].read_text())
            self.assertFalse(development["release_ready"])

            test_tasks_path = root / "test_tasks.jsonl"
            write_task_manifest(test_tasks_path, [study_task(split="test")])
            final_plan = plan_for(test_tasks_path, split="test")
            final_provider = ScriptedProvider("unit-test-model")
            final_records, final_ledger = run_live_study(
                load_task_manifest(test_tasks_path),
                provider=final_provider,
                compiler=TrackingCompiler(final_provider),
                plan=final_plan,
            )
            final_artifacts = write_real_run_bundle(
                root / "final_run",
                plan=final_plan,
                tasks_path=test_tasks_path,
                records=final_records,
                ledger=final_ledger,
            )
            final_evidence = json.loads(final_artifacts["evidence"].read_text())
            self.assertFalse(final_evidence["release_gate"]["release_ready"])
            self.assertIn("human_review.approved is false", final_evidence["release_gate"]["blocking_conditions"])
            review_script = PROJECT_ROOT / "scripts" / "record_phase_09_human_review.py"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(review_script),
                    "--run-dir",
                    str(root / "final_run"),
                    "--reviewer",
                    "Unit Reviewer",
                    "--approval-note",
                    "Reviewed immutable attempts, hashes, and Lean outcomes.",
                    "--approve",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            approved_evidence = json.loads(final_artifacts["evidence"].read_text())
            self.assertTrue(approved_evidence["release_gate"]["release_ready"])

    def test_openai_provider_parses_response_without_persisting_the_api_key(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "id": "resp_123",
                        "model": "unit-test-model",
                        "output_text": "by rfl",
                        "usage": {"input_tokens": 11, "output_tokens": 3},
                    }
                ).encode("utf-8")

        captured: list[object] = []

        def transport(request: object, **_: object) -> FakeResponse:
            captured.append(request)
            return FakeResponse()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-key"}, clear=False):
            provider = OpenAIResponsesProvider(
                model_identifier="unit-test-model", transport=transport
            )
            completion = provider.complete(prompt="prove", seed=1, request_label="unit")
        self.assertEqual(completion.text, "by rfl")
        self.assertEqual(completion.input_tokens, 11)
        self.assertEqual(len(captured), 1)


if __name__ == "__main__":
    unittest.main()
