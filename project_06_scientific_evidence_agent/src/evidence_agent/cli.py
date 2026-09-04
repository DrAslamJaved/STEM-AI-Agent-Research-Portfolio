"""Small, explicit command-line interface for the staged project workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from evidence_agent import __version__


UNIMPLEMENTED_COMMANDS = {
    "validate-data": "Complete Phase 02 dataset acquisition and validation first.",
    "build-index": "Complete Phase 03 retrieval-baseline implementation first.",
    "evaluate": "Complete the relevant evaluation pipeline before running this command.",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser without performing I/O or model loading."""
    parser = argparse.ArgumentParser(
        prog="evidence-agent",
        description="Scientific Evidence Verification and Citation-Audit Agent",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("contract", help="Print the active project contract.")
    subparsers.add_parser("validate-data", help="Validate acquired SciFact data.")
    subparsers.add_parser("build-index", help="Build a retrieval index.")
    evaluate = subparsers.add_parser("evaluate", help="Run a configured evaluation.")
    evaluate.add_argument("--config", type=str, help="Path to a YAML experiment config.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-compatible exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "contract":
        print(
            json.dumps(
                {
                    "project": "scientific_evidence_agent",
                    "version": __version__,
                    "runtime_gold_fields_forbidden": ["evidence", "cited_doc_ids"],
                    "current_phase": "foundation",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(
        f"Command '{args.command}' is not available yet. "
        f"{UNIMPLEMENTED_COMMANDS[args.command]}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
