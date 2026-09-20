from __future__ import annotations

import argparse
import json

from causal_audit_agent.contracts import CausalQuestion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Causal assumption-audit agent")
    parser.add_argument("--question", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--estimand", default="ATE", choices=("ATE", "ATT", "CATE"))
    parser.add_argument("--population", default="study population")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    contract = CausalQuestion(
        question=args.question,
        treatment=args.treatment,
        outcome=args.outcome,
        estimand=args.estimand,
        population=args.population,
    )
    print(json.dumps(contract.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
