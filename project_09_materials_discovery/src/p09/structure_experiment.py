"""Phase 08 fixed-protocol structure descriptor versus graph-model experiment."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .graph_model import (GraphTrainingConfig, fit_predict_graph_regressor,
                          require_torch)
from .structure_graph import crystal_graph, structure_descriptors
from .structure_task import TASK, load_structure_task


def _metrics(y_true, prediction) -> dict:
    return {"mae": float(mean_absolute_error(y_true, prediction)),
            "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
            "test_count": int(len(y_true))}


def _fold_data(task, fold: int):
    train_structures, train_targets = task.get_train_and_val_data(fold)
    test_structures, test_targets = task.get_test_data(fold, include_target=True)
    y_train = np.asarray(train_targets, dtype=float).reshape(-1)
    y_test = np.asarray(test_targets, dtype=float).reshape(-1)
    if not (len(train_structures) == len(y_train) and len(test_structures) == len(y_test)):
        raise ValueError("Matbench structures and target arrays disagree")
    return list(train_structures), y_train, list(test_structures), y_test


def run_fold(task, fold: int, seed: int, config: GraphTrainingConfig,
             rf_trees: int = 300) -> dict:
    """Evaluate both fixed models against the same official held-out fold."""
    if fold not in task.folds:
        raise ValueError(f"{fold!r} is not an official fold")
    train_structures, y_train, test_structures, y_test = _fold_data(task, fold)
    x_train = np.asarray([structure_descriptors(s) for s in train_structures])
    x_test = np.asarray([structure_descriptors(s) for s in test_structures])
    descriptor_model = RandomForestRegressor(n_estimators=rf_trees, random_state=seed,
                                             n_jobs=config.cpu_threads)
    descriptor_model.fit(x_train, y_train)
    descriptor_metrics = _metrics(y_test, descriptor_model.predict(x_test))

    train_graphs = [crystal_graph(s, config.cutoff_angstrom, config.max_neighbors)
                    for s in train_structures]
    test_graphs = [crystal_graph(s, config.cutoff_angstrom, config.max_neighbors)
                   for s in test_structures]
    graph_metrics = _metrics(y_test, fit_predict_graph_regressor(
        train_graphs, y_train, test_graphs, seed=seed, config=config))
    return {"fold": fold, "seed": seed, "train_count": int(len(y_train)),
            "models": {"descriptor_random_forest": descriptor_metrics,
                       "periodic_message_passing": graph_metrics}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="must remain 1; fixed for deterministic CPU training")
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("results/phase08_structure_comparison.json"))
    args = parser.parse_args()
    require_torch()
    if args.epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1 or args.rf_trees < 1:
        raise ValueError("epochs, batch size, CPU threads and RF trees must be positive")
    if args.cpu_threads != 1:
        raise ValueError("Phase 08 fixes CPU threads at one for reproducibility")
    task = load_structure_task()
    if len(set(args.folds)) != len(args.folds) or any(fold not in task.folds for fold in args.folds):
        raise ValueError(f"folds must be distinct official folds: {task.folds}")
    config = GraphTrainingConfig(epochs=args.epochs, batch_size=args.batch_size,
                                 cpu_threads=args.cpu_threads)
    runs = [run_fold(task, fold, args.seed, config, args.rf_trees) for fold in args.folds]
    output = {"phase": "08", "task": TASK, "benchmark": "Matbench v0.1",
              "target": task.metadata.get("target"), "unit": task.metadata.get("unit"),
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "versions": {name: version(name) for name in ("matbench", "pymatgen", "numpy", "scikit-learn", "torch")},
              "protocol": {"outer_split": "official Matbench folds", "seed": args.seed,
                           "descriptor_model": {"algorithm": "RandomForestRegressor", "trees": args.rf_trees},
                           "graph_model": config.as_dict()},
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(runs)} structure-model comparisons to {args.output}")


if __name__ == "__main__":
    main()
