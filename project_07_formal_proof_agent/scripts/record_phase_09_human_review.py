"""Record an explicit human approval after reviewing a locked Phase 09 run.

This command never approves a run implicitly.  It requires ``--approve`` and
an explanatory note, verifies the task and candidate hashes again, and then
rebuilds the Phase 08 release bundle.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from formal_math.evaluation_release import (  # noqa: E402
    build_release_bundle,
    load_attempts_jsonl,
    sha256_file,
    validate_release_manifest,
    write_release_bundle,
    write_release_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--approval-note", required=True)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()

    if not args.approve:
        raise SystemExit("Refusing to record approval without the explicit --approve flag.")
    if len(args.approval_note.strip()) < 20:
        raise SystemExit("Approval note must explain the review in at least 20 characters.")

    manifest_path = args.run_dir / "locked_evaluation_manifest.json"
    attempts_path = args.run_dir / "candidate_attempts.jsonl"
    evidence_path = args.run_dir / "release_evidence.json"
    report_path = args.run_dir / "release_report.md"
    if not all(path.is_file() for path in (manifest_path, attempts_path, evidence_path, report_path)):
        raise SystemExit("Run directory does not contain a complete locked-test evidence bundle.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("synthetic_demo") is True:
        raise SystemExit("Synthetic evidence can never be approved as a final result.")
    if manifest.get("evaluation_split") != "test" or manifest.get("test_split_locked") is not True:
        raise SystemExit("Human approval is permitted only for a locked test-split run.")
    provenance = manifest.get("provenance", {})
    if provenance.get("candidate_records_sha256") != sha256_file(attempts_path):
        raise SystemExit("Candidate records changed after the run; review cannot be recorded.")
    validate_release_manifest(manifest)

    review = {
        "schema_version": 1,
        "phase": "09",
        "approved": True,
        "reviewer": args.reviewer,
        "approval_note": args.approval_note.strip(),
        "reviewed_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "locked_manifest_sha256_before_approval": sha256(manifest_path.read_bytes()).hexdigest(),
        "candidate_records_sha256": sha256_file(attempts_path),
    }
    review_path = args.run_dir / "human_review.json"
    review_path.write_bytes(json.dumps(review, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    manifest["human_review"] = {
        "approved": True,
        "reviewer": args.reviewer,
        "review_record_sha256": sha256_file(review_path),
    }
    manifest_path.write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    attempts = load_attempts_jsonl(attempts_path)
    bundle = build_release_bundle(manifest, attempts)
    write_release_bundle(evidence_path, bundle)
    write_release_report(report_path, bundle)
    print(f"Wrote human review record: {review_path}")
    print(f"Release ready: {bundle['release_gate']['release_ready']}")


if __name__ == "__main__":
    main()
