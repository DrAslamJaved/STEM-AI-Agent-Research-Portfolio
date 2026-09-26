"""Official Matbench v0.1 entry point; run with p09-experiment."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .core import POLICIES, Settings, crossing_summary, simulate


TASK = "matbench_expt_gap"


def features_and_groups(inputs) -> tuple[np.ndarray, np.ndarray]:
    """Element fraction vector and transparent weighted elemental summaries."""
    from pymatgen.core import Composition, Element

    rows, groups = [], []
    for value in inputs:
        comp = Composition(str(value))
        fractions = comp.fractional_composition.get_el_amt_dict()
        vec = np.zeros(118, dtype=float)
        zs, weights, masses = [], [], []
        for symbol, amount in fractions.items():
            elem = Element(symbol)
            vec[elem.Z - 1] = amount
            zs.append(elem.Z)
            weights.append(amount)
            masses.append(float(elem.atomic_mass))
        weights = np.asarray(weights)
        zs = np.asarray(zs)
        masses = np.asarray(masses)
        mean_z = float(np.dot(weights, zs))
        mean_mass = float(np.dot(weights, masses))
        rows.append(np.r_[vec, len(weights), mean_z,
                          np.sqrt(np.dot(weights, (zs - mean_z) ** 2)),
                          mean_mass, np.sqrt(np.dot(weights, (masses - mean_mass) ** 2))])
        groups.append(comp.reduced_formula)
    return np.asarray(rows), np.asarray(groups)


def load_task():
    from matbench.bench import MatbenchBenchmark

    benchmark = MatbenchBenchmark(autoload=False, subset=[TASK])
    task = next(iter(benchmark.tasks))
    task.load()
    if task.metadata["input_type"] != "composition" or task.metadata["task_type"] != "regression":
        raise ValueError("Unexpected Matbench task type")
    return task


def exclude_test_compositions(x_train, y_train, train_groups, test_groups):
    """Group-exclusive robustness filter; the official test rows stay fixed."""
    keep = ~np.isin(train_groups, np.unique(test_groups))
    return x_train[keep], y_train[keep], train_groups[keep], int(np.count_nonzero(~keep))


def run(task, folds: list[int], seeds: list[int], settings: Settings,
        target_mae: float, split_mode: str = "official", include_diversity: bool = False) -> dict:
    if split_mode not in ("official", "group_exclusive"):
        raise ValueError("Unknown split mode")
    available = list(task.folds)
    rows, audits = [], []
    for fold_number in folds:
        fold = f"fold_{fold_number}"
        if fold_number not in available:
            raise ValueError(f"{fold_number} not in official folds: {available}")
        train_inputs, train_targets = task.get_train_and_val_data(fold_number)
        test_inputs, test_targets = task.get_test_data(fold_number, include_target=True)
        x_train, train_groups = features_and_groups(train_inputs)
        x_test, test_groups = features_and_groups(test_inputs)
        y_train = np.asarray(train_targets, dtype=float)
        y_test = np.asarray(test_targets, dtype=float)
        overlap = set(train_groups).intersection(test_groups)
        removed = 0
        if split_mode == "group_exclusive":
            x_train, y_train, train_groups, removed = exclude_test_compositions(
                x_train, y_train, train_groups, test_groups)
            assert not set(train_groups).intersection(test_groups)
        audits.append({"fold": fold, "train_count": len(y_train), "test_count": len(y_test),
                       "unique_train_compositions": len(set(train_groups)),
                       "original_overlapping_train_test_compositions": len(overlap),
                       "original_overlapping_test_records": int(sum(g in overlap for g in test_groups)),
                       "removed_training_records": removed})
        for seed in seeds:
            fold_rows = simulate(x_train, y_train, train_groups, x_test, y_test,
                                 seed=seed, settings=settings, include_diversity=include_diversity)
            rows.extend({"fold": fold, **r} for r in fold_rows)
    return {
        "task": TASK, "benchmark": "Matbench v0.1", "target_unit": "eV",
        "split_mode": split_mode,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {name: version(name) for name in ("matbench", "pymatgen", "numpy", "scikit-learn")},
        "settings": {"folds": folds, "seeds": seeds, "budgets": settings.budgets,
                     "initial": settings.initial, "calibration_fraction": settings.calibration_fraction,
                     "members": settings.members, "trees": settings.trees, "alpha": settings.alpha,
                     "target_mae_ev": target_mae, "include_diversity": include_diversity,
                     "diversity_weight": 0.5 if include_diversity else None,
                     "diversity_features": "first 118 element fractions" if include_diversity else None},
        "split_audits": audits, "checkpoints": rows,
        "threshold_summary": crossing_summary(rows, target_mae,
                                                POLICIES if include_diversity else POLICIES[:2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23])
    parser.add_argument("--budgets", nargs="+", type=int, default=[200, 400, 800, 1600])
    parser.add_argument("--target-mae", type=float, default=0.60)
    parser.add_argument("--split-mode", choices=["official", "group_exclusive"], default="official")
    parser.add_argument("--include-diversity", action="store_true", help="include locked uncertainty plus composition-distance acquisition")
    parser.add_argument("--output", type=Path, default=Path("results/pilot.json"))
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("folds and seeds must not contain duplicates")
    settings = Settings(budgets=tuple(args.budgets), initial=args.budgets[0])
    if not np.isfinite(args.target_mae) or args.target_mae <= 0:
        parser.error("target MAE must be positive and finite")
    result = run(load_task(), args.folds, args.seeds, settings, args.target_mae,
                 args.split_mode, args.include_diversity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(result['checkpoints'])} model checkpoints to {args.output}")


if __name__ == "__main__":
    main()
