"""Write an immutable Phase 09 plan before a live model call is permitted."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.live_study import (  # noqa: E402
    LiveStudyPlan,
    PINNED_LEAN_TOOLCHAIN,
    PINNED_MATHLIB_REVISION,
    PINNED_MINIF2F_COMMIT,
    load_task_manifest,
    sha256_file,
    validate_plan_for_tasks,
    write_study_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples-per-theorem", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--input-usd-per-million-tokens", type=float, required=True)
    parser.add_argument("--output-usd-per-million-tokens", type=float, required=True)
    parser.add_argument("--maximum-estimated-cost-usd", type=float, required=True)
    parser.add_argument("--send-seed-to-provider", action="store_true")
    parser.add_argument("--final-evaluation", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite an existing study plan: {args.output}")
    tasks = load_task_manifest(args.tasks)
    split = tasks[0].task.split
    if args.final_evaluation != (split == "test"):
        raise SystemExit("--final-evaluation is required exactly for a locked test task manifest.")
    plan = LiveStudyPlan(
        run_id=args.run_id,
        evaluation_mode="final_evaluation" if args.final_evaluation else "development",
        model_identifier=args.model,
        samples_per_theorem=args.samples_per_theorem,
        base_seed=args.base_seed,
        max_repair_iterations=2,
        task_manifest_sha256=sha256_file(args.tasks),
        benchmark_commit=PINNED_MINIF2F_COMMIT,
        lean_toolchain=PINNED_LEAN_TOOLCHAIN,
        mathlib_revision=PINNED_MATHLIB_REVISION,
        input_usd_per_million_tokens=args.input_usd_per_million_tokens,
        output_usd_per_million_tokens=args.output_usd_per_million_tokens,
        maximum_estimated_cost_usd=args.maximum_estimated_cost_usd,
        send_seed_to_provider=args.send_seed_to_provider,
    )
    validate_plan_for_tasks(plan, tasks)
    write_study_plan(args.output, plan)
    print(f"Wrote immutable Phase 09 study plan: {args.output}")
    print(f"Mode: {plan.evaluation_mode}; tasks: {len(tasks)}; samples per theorem: {plan.samples_per_theorem}")


if __name__ == "__main__":
    main()
