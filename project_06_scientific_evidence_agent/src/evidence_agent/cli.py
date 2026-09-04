"""Small, explicit command-line interface for the staged project workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from evidence_agent import __version__
from evidence_agent.data.acquisition import (
    DEFAULT_SCIFACT_URL,
    acquire_scifact,
    write_acquisition_manifest,
)
from evidence_agent.data.scifact import (
    validate_scifact_dataset,
    write_validation_report,
)


UNIMPLEMENTED_COMMANDS = {
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

    acquire = subparsers.add_parser(
        "acquire-data", help="Acquire the official SciFact data release."
    )
    acquire.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/scifact"),
        help="Directory for the immutable source archive and extracted release.",
    )
    acquire.add_argument(
        "--url",
        default=DEFAULT_SCIFACT_URL,
        help="Official SciFact archive URL.",
    )
    acquire.add_argument(
        "--provenance-path",
        type=Path,
        default=Path("validation/scifact_acquisition.json"),
        help="Path for the acquisition provenance manifest.",
    )

    validate = subparsers.add_parser(
        "validate-data", help="Validate an acquired SciFact release."
    )
    validate.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/scifact"),
        help="SciFact extraction directory or a parent containing it.",
    )
    validate.add_argument(
        "--report-path",
        type=Path,
        default=Path("validation/scifact_validation.json"),
        help="Path for the validation report.",
    )
    validate.add_argument(
        "--skip-cross-validation",
        action="store_true",
        help="Allow an intentionally reduced fixture without five-fold splits.",
    )
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

    if args.command == "acquire-data":
        manifest = acquire_scifact(output_dir=args.output_dir, url=args.url)
        write_acquisition_manifest(manifest, args.provenance_path)
        print(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "validate-data":
        summary = validate_scifact_dataset(
            data_dir=args.data_dir,
            require_cross_validation=not args.skip_cross_validation,
        )
        write_validation_report(summary, args.report_path)
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
        return 0

    print(
        f"Command '{args.command}' is not available yet. "
        f"{UNIMPLEMENTED_COMMANDS[args.command]}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
