"""Create a frozen Phase 09 task manifest from the canonical miniF2F clone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.live_study import (  # noqa: E402
    PINNED_MINIF2F_COMMIT,
    build_fuzzy_study_tasks,
    extract_minif2f_tasks,
    git_revision,
    sha256_file,
    write_task_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--mini-f2f-root", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all 244 miniF2F theorems.")
    parser.add_argument("--selection-seed", type=int, default=20260921)
    parser.add_argument("--without-fuzzy-suite", action="store_true")
    parser.add_argument(
        "--final-evaluation",
        action="store_true",
        help="Required before materialising a locked test-split task manifest.",
    )
    args = parser.parse_args()

    if args.split == "test" and not args.final_evaluation:
        raise SystemExit("Refusing test tasks without --final-evaluation.")
    if args.split == "valid" and args.final_evaluation:
        raise SystemExit("--final-evaluation is valid only with --split test.")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite an existing task manifest: {args.output}")
    if git_revision(args.mini_f2f_root) != PINNED_MINIF2F_COMMIT:
        raise SystemExit("The miniF2F checkout does not match the frozen Project 07 commit.")

    tasks = extract_minif2f_tasks(
        args.mini_f2f_root,
        split=args.split,
        limit=args.limit,
        selection_seed=args.selection_seed,
    )
    if not args.without_fuzzy_suite:
        tasks.extend(build_fuzzy_study_tasks(args.project_root, split=args.split))
    write_task_manifest(args.output, tasks)
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance = {
        "schema_version": 1,
        "phase": "09",
        "benchmark_commit": PINNED_MINIF2F_COMMIT,
        "split": args.split,
        "final_evaluation": args.final_evaluation,
        "miniF2F_task_count": sum(task.task.source == "miniF2F_v1" for task in tasks),
        "fuzzy_task_count": sum(task.task.source == "project07_fuzzy_suite" for task in tasks),
        "task_count": len(tasks),
        "selection_seed": args.selection_seed,
        "task_manifest_sha256": sha256_file(args.output),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote frozen {args.split}-split task manifest: {args.output}")
    print(f"Task SHA-256: {provenance['task_manifest_sha256']}")
    print(f"Tasks: {provenance['task_count']} ({provenance['miniF2F_task_count']} miniF2F, "
          f"{provenance['fuzzy_task_count']} fuzzy)")


if __name__ == "__main__":
    main()
