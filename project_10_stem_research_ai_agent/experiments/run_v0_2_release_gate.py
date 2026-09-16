"""Evaluate the release gate for the committed real Davis DTI workflow artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stem_research_agent.davis_manuscript import _load_json
from stem_research_agent.v02_release import evaluate_v02_release


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v0.2 Davis release gate.")
    parser.add_argument("--model-report", default="results/v0_2_davis_model_baselines.json")
    parser.add_argument("--literature-ledger", default="results/v0_2_dti_literature_ledger.json")
    parser.add_argument("--draft", default="results/v0_2_davis_constrained_draft.md")
    parser.add_argument("--trace", default="results/v0_2_davis_constrained_draft_trace.json")
    parser.add_argument("--output", default="results/v0_2_davis_release_report.json")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--review-reason", required=True)
    parser.add_argument("--tests-passed", action="store_true")
    parser.add_argument("--compileall-passed", action="store_true")
    parser.add_argument("--diff-check-passed", action="store_true")
    args = parser.parse_args()
    report = evaluate_v02_release(
        _load_json(args.model_report), _load_json(args.literature_ledger), Path(args.draft).read_text(encoding="utf-8"),
        _load_json(args.trace), reviewer=args.reviewer, review_reason=args.review_reason,
        tests_passed=args.tests_passed, compileall_passed=args.compileall_passed, diff_check_passed=args.diff_check_passed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
