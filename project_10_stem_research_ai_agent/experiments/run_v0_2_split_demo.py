"""Generate reproducible leakage diagnostics for the approved Davis source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stem_research_agent.davis import davis_records, load_davis_dataset
from stem_research_agent.dti_splits import (
    cold_drug_split,
    cold_target_split,
    pair_random_split,
    split_diagnostics,
    validate_split_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-aware Davis split diagnostics.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", default="config/v0_2_davis_manifest.json")
    parser.add_argument("--output", default="results/v0_2_davis_split_diagnostics.json")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    dataset = load_davis_dataset(args.data_dir, expected_sha256=manifest["sha256"])
    records = davis_records(dataset, binder_pkd_cutoff=float(manifest.get("binder_pkd_cutoff", 7.0)))
    pair_train, pair_test = pair_random_split(records, test_fraction=args.test_fraction, seed=args.seed)
    drug_train, drug_test, held_out_drugs = cold_drug_split(records, test_fraction=args.test_fraction, seed=args.seed)
    target_train, target_test, held_out_targets = cold_target_split(records, test_fraction=args.test_fraction, seed=args.seed)
    pair = split_diagnostics(pair_train, pair_test, split_name="pair_random")
    cold_drug = split_diagnostics(drug_train, drug_test, split_name="cold_drug")
    cold_target = split_diagnostics(target_train, target_test, split_name="cold_target")
    validate_split_diagnostics(pair)
    validate_split_diagnostics(cold_drug, cold_entity="drug")
    validate_split_diagnostics(cold_target, cold_entity="target")
    result = {
        "dataset_id": manifest["dataset_id"],
        "source_commit": manifest["source_commit"],
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "pair_random": pair,
        "cold_drug": {**cold_drug, "held_out_drug_ids": list(held_out_drugs)},
        "cold_target": {**cold_target, "held_out_target_ids": list(held_out_targets)},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
