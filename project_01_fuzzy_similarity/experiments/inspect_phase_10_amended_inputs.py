"""Fail-closed Phase 10 source-binding preflight.

This script performs no modelling.  Given the root of the validated
Project10_v02_clean worktree, it records hashes and verifies that the frozen
Project 02-compatible Davis inputs required by Phase 10 are actually present.
It explicitly rejects reliance on Project 10 v0.2's different label/split
regime by never accepting its outputs as substitutes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_LABEL = "interaction_kd_le_1000_nM"
EXPECTED_RANDOM_STATE = 20260830
REQUIRED_RAW = (
    "project_10_stem_research_ai_agent/data/raw/davis/ligands_can.txt",
    "project_10_stem_research_ai_agent/data/raw/davis/proteins.txt",
    "project_10_stem_research_ai_agent/data/raw/davis/Y",
)
REQUIRED_RECORDS = (
    "project_02_drug_target/reports/davis_binary_label_summary.json",
    "project_02_drug_target/reports/davis_split_audit.json",
    "project_02_drug_target/validation/davis_sha256.csv",
)
TABLE_CANDIDATES = (
    "project_02_drug_target/data/interim/davis_interactions_labeled.csv",
    "project_02_drug_target/data/interim/davis_interaction_table.csv",
    "project_02_drug_target/data/processed/davis_interaction_table.csv",
)
SPLIT_CANDIDATE = "project_02_drug_target/data/interim/davis_split_assignments.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def csv_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle), [])


def text_contains(path: Path, value: str) -> bool:
    return value in path.read_text(encoding="utf-8")


def json_has_random_state(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("random_state") == EXPECTED_RANDOM_STATE


def preflight(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root was not found: {source_root}")

    found: list[dict[str, Any]] = []
    missing: list[str] = []
    for relative in (*REQUIRED_RAW, *REQUIRED_RECORDS):
        path = source_root / relative
        if path.is_file():
            found.append(metadata(path, source_root))
        else:
            missing.append(relative)

    table_path = next((source_root / item for item in TABLE_CANDIDATES if (source_root / item).is_file()), None)
    split_path = source_root / SPLIT_CANDIDATE
    if table_path is None:
        missing.append("Project 02-compatible processed interaction table (approved candidate paths)")
    else:
        table = metadata(table_path, source_root)
        table["columns"] = csv_columns(table_path)
        found.append(table)
        if EXPECTED_LABEL not in table["columns"]:
            missing.append(f"{table['path']} with label column {EXPECTED_LABEL}")
    if split_path.is_file():
        split = metadata(split_path, source_root)
        split["columns"] = csv_columns(split_path)
        found.append(split)
    else:
        missing.append(SPLIT_CANDIDATE)

    label_record = source_root / REQUIRED_RECORDS[0]
    audit_record = source_root / REQUIRED_RECORDS[1]
    if label_record.is_file() and not text_contains(label_record, EXPECTED_LABEL):
        missing.append(f"label evidence for {EXPECTED_LABEL}")
    if audit_record.is_file() and not json_has_random_state(audit_record):
        missing.append(f"split audit with random_state {EXPECTED_RANDOM_STATE}")

    return {
        "study": "phase_10_external_dti",
        "amendment": "01_source_location",
        "status": "ready_for_runner_binding" if not missing else "incomplete_fail_closed",
        "source_root": source_root.as_posix(),
        "required_label": EXPECTED_LABEL,
        "required_random_state": EXPECTED_RANDOM_STATE,
        "excluded_project_10_v02_regime": {"label": "pKd_ge_7_0", "random_state": 20260915},
        "found_files": found,
        "missing_or_incompatible": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="perform provenance preflight only")
    parser.add_argument("--source-root", required=True, help="validated Project10_v02_clean root")
    parser.add_argument("--output", default="results/phase_10_amended_input_inventory.json")
    args = parser.parse_args()
    if not args.inspect:
        raise SystemExit("Refusing preflight without --inspect.")

    result = preflight(Path(args.source_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)
    if result["status"] != "ready_for_runner_binding":
        raise SystemExit("Fail-closed: frozen Phase 10 inputs are incomplete or incompatible.")


if __name__ == "__main__":
    main()
