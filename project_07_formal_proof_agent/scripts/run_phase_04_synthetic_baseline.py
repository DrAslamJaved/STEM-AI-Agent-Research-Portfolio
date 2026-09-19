"""Run a deterministic Phase 04 LLM-only baseline smoke test.

No external model is called. The scripted provider is solely a reproducibility
fixture for proving that generation completes before post-hoc evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_harness import CompilerOutcome, evaluate_batch, load_tasks_jsonl  # noqa: E402
from formal_math.llm_only_baseline import (  # noqa: E402
    BaselineRunConfig,
    Completion,
    GenerationRequest,
    generate_llm_only_candidates,
    write_baseline_evidence,
    write_generation_records,
)


class ScriptedProvider:
    """Offline fixture whose responses are fixed before evaluation starts."""

    def complete(self, request: GenerationRequest) -> Completion:
        responses = {
            ("synthetic_nat_refl", 0): "```lean\nby rfl\n```",
            ("synthetic_nat_refl", 1): "by exact 0",
            ("synthetic_nat_add_zero", 0): "by exact 0",
            ("synthetic_nat_add_zero", 1): "by simpa using Nat.add_zero 3",
        }
        return Completion(
            text=responses[(request.theorem_id, request.sample_index)],
            model_identifier="scripted-offline-provider",
            input_tokens=12,
            output_tokens=4,
        )


def synthetic_compiler(source: str) -> CompilerOutcome:
    """Deterministic post-generation evaluator used only by this smoke run."""
    accepted = "by rfl" in source or "by simpa using Nat.add_zero 3" in source
    return CompilerOutcome(
        passed=accepted,
        stdout="synthetic evaluator accepted candidate" if accepted else "",
        stderr="" if accepted else "synthetic evaluator rejected candidate",
        exit_code=0 if accepted else 1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    project_root = args.project_root

    tasks = load_tasks_jsonl(project_root / "data" / "processed" / "phase_04_synthetic_tasks.jsonl")
    config = BaselineRunConfig(
        run_id="phase_04_synthetic_baseline_v1",
        model_identifier="scripted-offline-provider",
        prompt_id="phase_04_llm_only_prompt_v1",
        samples_per_theorem=2,
        seed=20260919,
    )
    generated = generate_llm_only_candidates(tasks, ScriptedProvider(), config)

    # Evaluation is deliberately invoked only after the full generation batch exists.
    results = evaluate_batch(
        tasks,
        [item.candidate for item in generated],
        synthetic_compiler,
        mode="development",
    )

    result_root = project_root / "results"
    write_generation_records(result_root / "phase_04_synthetic_candidates.jsonl", generated)
    write_baseline_evidence(
        result_root / "phase_04_synthetic_baseline.json",
        config=config,
        generated=generated,
        results=results,
        synthetic_demo=True,
    )
    print(f"Wrote Phase 04 synthetic candidates: {result_root / 'phase_04_synthetic_candidates.jsonl'}")
    print(f"Wrote Phase 04 synthetic baseline evidence: {result_root / 'phase_04_synthetic_baseline.json'}")
    print("This is a generation/evaluation-control smoke test, not a miniF2F or model-performance result.")


if __name__ == "__main__":
    main()
