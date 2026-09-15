"""Render a Davis manuscript draft from committed model and evidence artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stem_research_agent.davis_manuscript import build_draft_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an evidence-constrained Davis DTI draft.")
    parser.add_argument("--model-report", default="results/v0_2_davis_model_baselines.json")
    parser.add_argument("--literature-ledger", default="results/v0_2_dti_literature_ledger.json")
    parser.add_argument("--output", default="results/v0_2_davis_constrained_draft.md")
    parser.add_argument("--trace-output", default="results/v0_2_davis_constrained_draft_trace.json")
    args = parser.parse_args()
    draft = build_draft_from_files(args.model_report, args.literature_ledger)
    output, trace = Path(args.output), Path(args.trace_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(draft.markdown, encoding="utf-8")
    trace.write_text(json.dumps(draft.trace, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(trace)


if __name__ == "__main__":
    main()
