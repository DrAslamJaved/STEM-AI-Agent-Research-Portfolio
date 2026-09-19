"""Aggregate a captured, locked-test Project 07 final evaluation.

This script does not generate candidates. It accepts pre-captured JSONL
records from the three pre-registered arms and produces an auditable release
bundle. It refuses synthetic manifests so a smoke fixture cannot be promoted
to a final result by filename alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_release import (  # noqa: E402
    build_release_bundle,
    load_attempts_jsonl,
    sha256_file,
    write_release_bundle,
    write_release_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("synthetic_demo") is True:
        raise SystemExit("Refusing a synthetic manifest in the final-evaluation runner.")
    provenance = manifest.get("provenance", {})
    if provenance.get("task_manifest_sha256") != sha256_file(args.task_manifest):
        raise SystemExit("Task-manifest checksum does not match the locked final manifest.")
    if provenance.get("candidate_records_sha256") != sha256_file(args.attempts):
        raise SystemExit("Candidate-record checksum does not match the locked final manifest.")
    attempts = load_attempts_jsonl(args.attempts)
    bundle = build_release_bundle(manifest, attempts)
    write_release_bundle(args.output, bundle)
    write_release_report(args.report, bundle)

    print(f"Wrote locked-test release evidence: {args.output}")
    print(f"Wrote release report: {args.report}")
    print(f"Release ready: {bundle['release_gate']['release_ready']}")


if __name__ == "__main__":
    main()
