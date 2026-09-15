"""Evaluate reproducible DTI baselines across approved Davis split conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stem_research_agent.davis import davis_records, load_davis_dataset
from stem_research_agent.dti_models import evaluate_dti_models
from stem_research_agent.dti_splits import cold_drug_split, cold_target_split, pair_random_split, split_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible Davis DTI baseline models.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", default="config/v0_2_davis_manifest.json")
    parser.add_argument("--output", default="results/v0_2_davis_model_baselines.json")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    dataset = load_davis_dataset(args.data_dir, expected_sha256=manifest["sha256"])
    records = davis_records(dataset, binder_pkd_cutoff=float(manifest.get("binder_pkd_cutoff", 7.0)))
    splits = {
        "pair_random": pair_random_split(records, test_fraction=args.test_fraction, seed=args.seed),
        "cold_drug": cold_drug_split(records, test_fraction=args.test_fraction, seed=args.seed)[:2],
        "cold_target": cold_target_split(records, test_fraction=args.test_fraction, seed=args.seed)[:2],
    }
    outcomes = {
        name: {
            "diagnostics": split_diagnostics(train, test, split_name=name),
            "models": evaluate_dti_models(train, test, seed=args.seed),
        }
        for name, (train, test) in splits.items()
    }
    result = {
        "dataset_id": manifest["dataset_id"],
        "source_commit": manifest["source_commit"],
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "feature_set": "fixed_smiles_and_protein_composition_v1",
        "outcomes": outcomes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
