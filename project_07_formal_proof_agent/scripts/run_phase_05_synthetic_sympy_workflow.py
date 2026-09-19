"""Run a deterministic Phase 05 LLM+SymPy workflow smoke test.

The provider and compiler are offline fixtures.  They demonstrate that exact
SymPy diagnostics and a bounded repair loop occur before independent evaluation;
they do not call an LLM or compile miniF2F.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_harness import CompilerOutcome, evaluate_batch, load_tasks_jsonl  # noqa: E402
from formal_math.sympy_counterexamples import (  # noqa: E402
    SympyCompletion,
    SympyGenerationRequest,
    SympyRunConfig,
    SympyContext,
    diagnose_symbolic_equality,
    find_singleton_jaccard_identity_counterexample,
    generate_llm_sympy_candidates,
    write_sympy_generation_records,
    write_sympy_workflow_evidence,
)


class ScriptedSympyProvider:
    """Offline responses fixed by theorem and iteration before evaluation begins."""

    def complete(self, request: SympyGenerationRequest) -> SympyCompletion:
        responses = {
            ("synthetic_symbolic_identity", 0): "```lean\nby rfl\n```",
            ("synthetic_symbolic_identity", 1): "by rfl",
            ("synthetic_fuzzy_witness", 0): "by exact 0",
            ("synthetic_fuzzy_witness", 1): "by norm_num",
        }
        return SympyCompletion(
            text=responses[(request.theorem_id, request.iteration)],
            model_identifier="scripted-sympy-offline-provider",
            input_tokens=24,
            output_tokens=4,
        )


def synthetic_compiler(source: str) -> CompilerOutcome:
    """Deterministic post-generation evaluator used only for the smoke run."""
    accepted = "by rfl" in source or "by norm_num" in source
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

    tasks = load_tasks_jsonl(project_root / "data" / "processed" / "phase_05_synthetic_tasks.jsonl")
    identity = diagnose_symbolic_equality(
        "binomial_expansion",
        "(x + y)**2",
        "x**2 + 2*x*y + y**2",
        variables=("x", "y"),
    )
    jaccard_witness = find_singleton_jaccard_identity_counterexample()
    contexts = {
        "synthetic_symbolic_identity": SympyContext(diagnostics=(identity,)),
        "synthetic_fuzzy_witness": SympyContext(counterexamples=(jaccard_witness,)),
    }
    config = SympyRunConfig(
        run_id="phase_05_synthetic_sympy_v1",
        model_identifier="scripted-sympy-offline-provider",
        prompt_id="phase_05_llm_sympy_prompt_v1",
        max_sympy_repair_iterations=1,
        seed=20260919,
    )
    generated = generate_llm_sympy_candidates(tasks, ScriptedSympyProvider(), config, contexts)

    # The evaluator is deliberately invoked only after the full candidate batch exists.
    results = evaluate_batch(
        tasks,
        [item.candidate for item in generated],
        synthetic_compiler,
        mode="development",
    )

    result_root = project_root / "results"
    write_sympy_generation_records(result_root / "phase_05_synthetic_candidates.jsonl", generated)
    write_sympy_workflow_evidence(
        result_root / "phase_05_synthetic_sympy_workflow.json",
        config=config,
        generated=generated,
        results=results,
        synthetic_demo=True,
    )
    print(f"Wrote Phase 05 synthetic candidates: {result_root / 'phase_05_synthetic_candidates.jsonl'}")
    print(f"Wrote Phase 05 synthetic workflow evidence: {result_root / 'phase_05_synthetic_sympy_workflow.json'}")
    print("This is a SymPy/control smoke test, not a miniF2F or model-performance result.")


if __name__ == "__main__":
    main()
