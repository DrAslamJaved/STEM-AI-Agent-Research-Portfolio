"""Execute a pre-registered real Project 07 study using the pinned Lean compiler."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.live_study import (  # noqa: E402
    OpenAIResponsesProvider,
    load_study_plan,
    load_task_manifest,
    make_pinned_live_compiler,
    run_live_study,
    sha256_file,
    validate_plan_for_tasks,
    write_real_run_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--elan", type=Path, required=True)
    parser.add_argument("--lean-cwd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--provider-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--lean-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--confirm-final-evaluation",
        action="store_true",
        help="Required to spend budget on a locked miniF2F test-split run.",
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite an existing live-run directory: {args.output_dir}")
    tasks = load_task_manifest(args.tasks)
    plan = load_study_plan(args.plan)
    if sha256_file(args.tasks) != plan.task_manifest_sha256:
        raise SystemExit("Task manifest checksum does not match the immutable study plan.")
    validate_plan_for_tasks(plan, tasks)
    if plan.evaluation_mode == "final_evaluation" and not args.confirm_final_evaluation:
        raise SystemExit("Refusing locked test evaluation without --confirm-final-evaluation.")
    if plan.evaluation_mode == "development" and args.confirm_final_evaluation:
        raise SystemExit("--confirm-final-evaluation cannot be used with a development plan.")

    provider = OpenAIResponsesProvider(
        model_identifier=plan.model_identifier,
        api_key_env=args.api_key_env,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        timeout_seconds=args.provider_timeout_seconds,
        send_seed=plan.send_seed_to_provider,
    )
    compiler = make_pinned_live_compiler(
        args.elan,
        cwd=args.lean_cwd,
        timeout_seconds=args.lean_timeout_seconds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    journal_path = args.output_dir / "generation_journal.jsonl"
    try:
        records, ledger = run_live_study(
            tasks,
            provider=provider,
            compiler=compiler,
            plan=plan,
            generation_journal_path=journal_path,
        )
    except Exception:
        print(f"Run stopped. Any paid completion captured before failure is preserved in: {journal_path}")
        raise
    artifacts = write_real_run_bundle(
        args.output_dir,
        plan=plan,
        tasks_path=args.tasks,
        records=records,
        ledger=ledger,
    )
    for label, path in artifacts.items():
        print(f"{label}: {path}")
    print(f"Estimated cost: ${ledger.total_estimated_cost_usd:.6f} across {ledger.request_count} requests")
    if plan.evaluation_mode == "development":
        print("Development evidence is not a final miniF2F result.")
    else:
        print("The locked-test release remains blocked until a human review is recorded.")


if __name__ == "__main__":
    main()
