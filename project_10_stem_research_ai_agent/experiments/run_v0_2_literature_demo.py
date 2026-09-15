"""Verify the declared Davis literature plan against Crossref and emit a ledger."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

from stem_research_agent.crossref import retrieve_verified_source
from stem_research_agent.literature_ledger import build_literature_ledger, load_literature_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Crossref-verified Davis literature ledger.")
    parser.add_argument("--plan", default="config/v0_2_dti_literature_plan.json")
    parser.add_argument("--output", default="results/v0_2_dti_literature_ledger.json")
    parser.add_argument("--mailto", default=None, help="Optional contact address for the Crossref User-Agent.")
    args = parser.parse_args()
    plan = load_literature_plan(args.plan)

    def retrieve(spec):
        return retrieve_verified_source(
            spec["source_id"], spec["doi"], expected_title=spec.get("expected_title"),
            expected_year=spec.get("expected_year"), mailto=args.mailto,
        )

    ledger = build_literature_ledger(plan, retrieve=retrieve)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ledger.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
