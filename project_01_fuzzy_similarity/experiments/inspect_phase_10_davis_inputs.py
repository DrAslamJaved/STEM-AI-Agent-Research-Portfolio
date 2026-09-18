"""Inspect the local Project 02 Davis artefacts without running a model.

This provenance gate inventories raw, processed, and interim data files,
records SHA-256 hashes and delimited-file headers where applicable, and writes
a JSON manifest.  It never generates prediction or evaluation results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


STRUCTURED_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(path: Path) -> str:
    name = path.name.lower()
    if "raw" in {part.lower() for part in path.parts}:
        return "raw_source_file"
    if "split" in name or "assignment" in name:
        return "split_candidate"
    if any(token in name for token in ("pair", "interaction", "davis", "processed", "feature")):
        return "data_candidate"
    return "other_candidate"


def delimited_metadata(path: Path) -> dict[str, Any]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])
        row_count = sum(1 for _ in reader)
    return {"format": path.suffix.lower().lstrip("."), "columns": header, "data_rows": row_count}


def json_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        first_nonempty = next((line.strip() for line in handle if line.strip()), "")
    if not first_nonempty:
        return {"format": path.suffix.lower().lstrip("."), "first_record_kind": "empty"}
    if path.suffix.lower() == ".jsonl":
        try:
            first_record = json.loads(first_nonempty)
        except json.JSONDecodeError:
            return {"format": "jsonl", "first_record_kind": "invalid_json"}
        return {
            "format": "jsonl",
            "first_record_kind": type(first_record).__name__,
            "first_record_keys": sorted(first_record) if isinstance(first_record, dict) else [],
        }
    try:
        first_value = json.loads(first_nonempty)
    except json.JSONDecodeError:
        return {"format": "json", "first_record_kind": "multiline_or_unreadable"}
    return {
        "format": "json",
        "top_level_kind": type(first_value).__name__,
        "top_level_keys": sorted(first_value) if isinstance(first_value, dict) else [],
    }


def file_metadata(path: Path, project_root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": path.relative_to(project_root).as_posix(),
        "role": classify(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if path.suffix.lower() in {".csv", ".tsv"}:
        metadata.update(delimited_metadata(path))
    elif path.suffix.lower() in {".json", ".jsonl"}:
        metadata.update(json_metadata(path))
    else:
        metadata.update({"format": "unparsed", "extension": path.suffix.lower()})
    return metadata


def inventory(project_root: Path) -> dict[str, Any]:
    project_two = project_root / "project_02_drug_target"
    if not project_two.is_dir():
        raise FileNotFoundError(f"Project 02 directory was not found: {project_two}")
    data_root = project_two / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Project 02 data directory was not found: {data_root}")

    files = sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    )
    if not files:
        raise FileNotFoundError("No Project 02 data artefacts were found")
    return {
        "status": "inspected_no_model_execution",
        "project_02_directory": project_two.as_posix(),
        "data_root": data_root.as_posix(),
        "files": [file_metadata(path, project_two) for path in files],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="create a provenance inventory only")
    parser.add_argument("--output", default="results/phase_10_input_inventory.json")
    args = parser.parse_args()
    if not args.inspect:
        raise SystemExit("Refusing inspection without --inspect.")

    portfolio_root = Path(__file__).resolve().parents[2]
    result = inventory(portfolio_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
