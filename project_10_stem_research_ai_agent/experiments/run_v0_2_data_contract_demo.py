"""Report the validated provenance and label prevalence of a local Davis copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stem_research_agent.davis import davis_records, load_davis_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical Davis source files.")
    parser.add_argument("--data-dir", required=True, help="Directory containing ligands_can.txt, proteins.txt and Y.")
    parser.add_argument("--output", default="results/v0_2_davis_data_report.json")
    parser.add_argument("--binder-pkd-cutoff", type=float, default=7.0)
    args = parser.parse_args()
    dataset = load_davis_dataset(args.data_dir)
    records = davis_records(dataset, binder_pkd_cutoff=args.binder_pkd_cutoff)
    result = {
        "source": dataset.source_report.to_dict(),
        "binder_pkd_cutoff": args.binder_pkd_cutoff,
        "positive_label_count": sum(row["label"] for row in records),
        "positive_label_rate": sum(row["label"] for row in records) / len(records),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
