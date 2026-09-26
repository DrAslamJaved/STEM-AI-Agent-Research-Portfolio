"""Load and audit the structure-input Matbench dielectric task for Phase 07."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .structure_graph import (DEFAULT_CUTOFF_ANGSTROM, DEFAULT_MAX_NEIGHBORS,
                              crystal_graph, graph_summary, structure_descriptors)


TASK = "matbench_dielectric"


def load_structure_task():
    from matbench.bench import MatbenchBenchmark

    benchmark = MatbenchBenchmark(autoload=False, subset=[TASK])
    task = next(iter(benchmark.tasks))
    task.load()
    if task.metadata["input_type"] != "structure" or task.metadata["task_type"] != "regression":
        raise ValueError("Unexpected Matbench task type")
    return task


def audit_fold(task, fold_number: int, limit: int = 64,
               cutoff_angstrom: float = DEFAULT_CUTOFF_ANGSTROM,
               max_neighbors: int = DEFAULT_MAX_NEIGHBORS) -> dict:
    if not isinstance(fold_number, int) or fold_number not in task.folds:
        raise ValueError(f"{fold_number!r} is not an official fold")
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    train_inputs, _ = task.get_train_and_val_data(fold_number)
    test_inputs = task.get_test_data(fold_number, include_target=False)
    output = {}
    for partition, inputs in (("train", train_inputs), ("test", test_inputs)):
        sample = list(inputs)[:limit]
        graphs = [crystal_graph(s, cutoff_angstrom, max_neighbors) for s in sample]
        descriptors = [structure_descriptors(s) for s in sample]
        output[partition] = {"records": len(inputs), "audited_graphs": len(graphs),
                             "graph": graph_summary(graphs),
                             "descriptor_dimension": len(descriptors[0]),
                             "finite_descriptors": bool(all(np.isfinite(d).all() for d in descriptors))}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--cutoff-angstrom", type=float, default=DEFAULT_CUTOFF_ANGSTROM,
                        help="periodic neighbour radius in angstrom")
    parser.add_argument("--max-neighbors", type=int, default=DEFAULT_MAX_NEIGHBORS)
    parser.add_argument("--output", type=Path, default=Path("results/phase07_structure_audit.json"))
    args = parser.parse_args()
    task = load_structure_task()
    audit = audit_fold(task, args.fold, args.limit, args.cutoff_angstrom, args.max_neighbors)
    output = {"phase": "07", "task": TASK, "benchmark": "Matbench v0.1",
              "target": task.metadata.get("target"), "unit": task.metadata.get("unit"),
              "fold": args.fold, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "versions": {name: version(name) for name in ("matbench", "pymatgen", "numpy", "scikit-learn")},
              "graph_protocol": {"cutoff_angstrom": args.cutoff_angstrom,
                                 "max_neighbors_per_site": args.max_neighbors,
                                 "node_feature": "atomic number", "edge_feature": "periodic distance"},
              "audit": audit}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved structure audit to {args.output}")


if __name__ == "__main__":
    main()
