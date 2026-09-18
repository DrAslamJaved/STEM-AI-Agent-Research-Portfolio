"""Execute the approved Phase 10R external Davis study only with ``--execute``.

This runner constructs its own table and split assignments from the newly frozen
raw source.  It intentionally cannot read Phase 10 CSV files or Project 10 v0.2
predictions, labels, or split diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SOURCE_COMMIT = "a546a8433a6822e958f36171c4356ad6f414d623"
RAW_RELATIVE = Path("project_10_stem_research_ai_agent/data/raw/davis")
MANIFEST_RELATIVE = Path("project_10_stem_research_ai_agent/config/v0_2_davis_manifest.json")
SPLIT_SEED = 20260918
BOOTSTRAP_SEED = 20260919
K_VALUES = (3, 5, 9, 15, 25)
METHODS = ("theta1", "jaccard", "dice", "cosine")
PROBABILITY_TOLERANCE = 1e-12
SMILES = "#%()+-.0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz"
AMINO = "ACDEFGHIKLMNPQRSTVWY"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def manifest_hashes(manifest: Path) -> set[str]:
    if not manifest.is_file():
        return set()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if "sha256" in key.lower() and isinstance(item, str):
                    found.add(item.lower())
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return found


def load_raw(source_root: Path) -> tuple[list[str], list[str], np.ndarray, dict[str, Any]]:
    raw = source_root / RAW_RELATIVE
    ligand_path, protein_path, affinity_path = raw / "ligands_can.txt", raw / "proteins.txt", raw / "Y"
    required = (ligand_path, protein_path, affinity_path)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("Phase 10R requires ligands_can.txt, proteins.txt, and Y at the approved raw location")
    hashes = [sha256(path) for path in required]
    known = manifest_hashes(source_root / MANIFEST_RELATIVE)
    if known and any(value not in known for value in hashes):
        raise ValueError("raw Davis hash does not occur in the retained manifest")
    ligands_map = json.loads(ligand_path.read_text(encoding="utf-8"))
    proteins_map = json.loads(protein_path.read_text(encoding="utf-8"))
    if not isinstance(ligands_map, dict) or not isinstance(proteins_map, dict):
        raise ValueError("ligands_can.txt and proteins.txt must be JSON mappings")
    with affinity_path.open("rb") as handle:
        affinity = np.asarray(pickle.load(handle, encoding="latin1"), dtype=float)
    ligands, proteins = list(ligands_map.values()), list(proteins_map.values())
    if affinity.shape != (68, 442) or len(ligands) != 68 or len(proteins) != 442:
        raise ValueError("expected 68 ligands, 442 proteins, and a Y matrix of shape (68, 442)")
    if not np.isfinite(affinity).all() or not (affinity > 0).all():
        raise ValueError("Y must be finite and strictly positive; no filtering or imputation is permitted")
    if not all(isinstance(value, str) and value for value in (*ligands, *proteins)):
        raise ValueError("every source SMILES and protein sequence must be a non-empty string")
    return ligands, proteins, affinity, {"source_commit": SOURCE_COMMIT, "raw_files": [file_record(path, source_root) for path in required], "manifest_checked": bool(known)}


def frequency(text: str, alphabet: str) -> list[float]:
    counts = Counter(text)
    total = len(text)
    return [counts[char] / total for char in alphabet] + [sum(count for char, count in counts.items() if char not in alphabet) / total]


def construct(ligands: list[str], proteins: list[str], affinity: np.ndarray) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    features: list[list[float]] = []
    for drug_index, smiles in enumerate(ligands):
        drug_vector = frequency(smiles, SMILES)
        for target_index, sequence in enumerate(proteins):
            target_vector = frequency(sequence, AMINO)
            kd = float(affinity[drug_index, target_index])
            rows.append({"observed_pair_index": len(rows), "drug_index": drug_index, "target_index": target_index, "smiles": smiles, "sequence": sequence, "kd_nM": kd, "interaction_kd_le_1000_nM": int(kd <= 1000.0)})
            features.append(drug_vector + target_vector)
    matrix = np.asarray(features, dtype=float)
    if len(rows) != 30056 or matrix.shape[0] != 30056 or not ((matrix >= 0).all() and (matrix <= 1).all()):
        raise ValueError("constructed table does not satisfy the frozen Phase 10R dimensions or membership bounds")
    split = len(SMILES) + 1
    if not np.allclose(matrix[:, :split].sum(axis=1), 1.0, atol=1e-12) or not np.allclose(matrix[:, split:].sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("fuzzy component cardinalities are not one")
    canonical = json.dumps(rows, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"rows": rows, "features": matrix, "table_sha256": hashlib.sha256(canonical).hexdigest()}


def make_splits(rows: list[dict[str, Any]], labels: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

    indices = np.arange(len(rows))
    groups_drug = np.asarray([row["drug_index"] for row in rows])
    groups_target = np.asarray([row["target_index"] for row in rows])
    output: dict[str, dict[str, np.ndarray]] = {}
    for name, groups in (("cold_drug", groups_drug), ("cold_target", groups_target)):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
        train, test = next(iter(splitter.split(indices, labels, groups)))
        output[name] = {"train": train, "test": test, "groups": groups}
    pair_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SPLIT_SEED)
    train, test = next(iter(pair_splitter.split(indices, labels)))
    output["pair_random"] = {"train": train, "test": test, "groups": np.arange(len(rows))}
    for name, split in output.items():
        train, test = split["train"], split["test"]
        if set(train) & set(test) or len(train) + len(test) != len(rows) or len(np.unique(labels[test])) != 2:
            raise ValueError(f"{name} does not form a valid two-class non-overlapping partition")
        if name.startswith("cold_") and set(split["groups"][train]) & set(split["groups"][test]):
            raise ValueError(f"{name} has held-out entity leakage")
    return output


def top_order(scores: np.ndarray, maximum_k: int) -> np.ndarray:
    """Top scores descending; exact score ties resolve by ascending train-row index."""
    threshold = np.partition(scores, len(scores) - maximum_k)[len(scores) - maximum_k]
    above = np.flatnonzero(scores > threshold)
    above = above[np.argsort(-scores[above], kind="stable")]
    equal = np.flatnonzero(scores == threshold)
    return np.concatenate((above, equal))[:maximum_k]


def bounded_probability(value: float) -> float:
    """Accept only mathematically valid probabilities, allowing round-off at 0/1."""
    if not math.isfinite(value) or value < -PROBABILITY_TOLERANCE or value > 1.0 + PROBABILITY_TOLERANCE:
        raise ValueError(f"weighted-kNN probability is outside [0, 1] beyond numerical tolerance: {value!r}")
    return min(1.0, max(0.0, value))


def predictions(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> dict[str, dict[int, np.ndarray]]:
    """Return all four weighted-kNN predictions, sharing the equivalent fuzzy ranking."""
    if len(train_x) < max(K_VALUES):
        raise ValueError("training partition is smaller than the largest pre-specified k")
    result = {method: {k: np.empty(len(test_x), dtype=float) for k in K_VALUES} for method in METHODS}
    train_sums, train_norms = train_x.sum(axis=1), np.linalg.norm(train_x, axis=1)
    for row_index, vector in enumerate(test_x):
        delta = np.minimum(train_x, vector).sum(axis=1)
        left = np.maximum(vector - train_x, 0.0).sum(axis=1)
        right = np.maximum(train_x - vector, 0.0).sum(axis=1)
        alpha, beta = np.maximum(left, right), np.minimum(left, right)
        theta = np.divide(delta * delta, delta * delta + alpha * beta + delta * (alpha + beta), out=np.zeros_like(delta), where=(delta * delta + alpha * beta + delta * (alpha + beta)) != 0)
        jaccard = np.divide(delta, np.maximum(train_x, vector).sum(axis=1), out=np.zeros_like(delta), where=np.maximum(train_x, vector).sum(axis=1) != 0)
        dice = np.divide(2 * delta, train_sums + vector.sum(), out=np.zeros_like(delta), where=(train_sums + vector.sum()) != 0)
        cosine = np.divide(train_x @ vector, train_norms * np.linalg.norm(vector), out=np.zeros_like(delta), where=(train_norms * np.linalg.norm(vector)) != 0)
        shared_order, cosine_order = top_order(theta, max(K_VALUES)), top_order(cosine, max(K_VALUES))
        for name, scores, order in (("theta1", theta, shared_order), ("jaccard", jaccard, shared_order), ("dice", dice, shared_order), ("cosine", cosine, cosine_order)):
            for k in K_VALUES:
                neighbours = order[:k]
                denominator = scores[neighbours].sum()
                raw_probability = (scores[neighbours] @ train_y[neighbours]) / denominator if denominator else float(train_y.mean())
                result[name][k][row_index] = bounded_probability(float(raw_probability))
    return result


def inner_folds(indices: np.ndarray, labels: np.ndarray, groups: np.ndarray, split_name: str) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if split_name == "pair_random":
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED).split(indices, labels[indices])
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED).split(indices, labels[indices], groups[indices])


def select_and_threshold(features: np.ndarray, labels: np.ndarray, outer_train: np.ndarray, groups: np.ndarray, split_name: str) -> tuple[dict[str, int], dict[str, float]]:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    oof = {method: {k: np.full(len(outer_train), np.nan) for k in K_VALUES} for method in METHODS}
    for fold_number, (train_local, valid_local) in enumerate(inner_folds(outer_train, labels, groups, split_name), start=1):
        print(f"{split_name}: completed inner fold {fold_number}/5", flush=True)
        train, valid = outer_train[train_local], outer_train[valid_local]
        fold = predictions(features[train], labels[train], features[valid])
        for method in METHODS:
            for k in K_VALUES:
                oof[method][k][valid_local] = fold[method][k]
    selected: dict[str, int] = {}
    thresholds: dict[str, float] = {}
    y = labels[outer_train]
    for method in METHODS:
        selected[method] = min(K_VALUES, key=lambda k: (-average_precision_score(y, oof[method][k]), k))
        precision, recall, thresholds_raw = precision_recall_curve(y, oof[method][selected[method]])
        f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1], out=np.zeros(len(thresholds_raw)), where=(precision[:-1] + recall[:-1]) != 0)
        thresholds[method] = float(thresholds_raw[int(np.argmax(f1))]) if len(thresholds_raw) else 0.5
    return selected, thresholds


def metric_summary(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

    probability = np.asarray([bounded_probability(float(value)) for value in probability])
    predicted = probability >= threshold
    return {"ap": float(average_precision_score(y, probability)), "roc_auc": float(roc_auc_score(y, probability)), "brier": float(brier_score_loss(y, probability)), "precision": float(precision_score(y, predicted, zero_division=0)), "recall": float(recall_score(y, predicted, zero_division=0)), "f1": float(f1_score(y, predicted, zero_division=0))}


def bootstrap_ap(y: np.ndarray, first: np.ndarray, second: np.ndarray, units: np.ndarray, seed: int, grouped: bool) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score

    rng, unique = np.random.default_rng(seed), np.unique(units)
    values: list[float] = []
    discarded = 0
    while len(values) < 10_000:
        if grouped:
            sampled = rng.choice(unique, size=len(unique), replace=True)
            chosen = np.concatenate([np.flatnonzero(units == item) for item in sampled])
        else:
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            chosen = np.concatenate((rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)))
        if len(np.unique(y[chosen])) != 2:
            discarded += 1
            continue
        values.append(float(average_precision_score(y[chosen], first[chosen]) - average_precision_score(y[chosen], second[chosen])))
    return {"difference": float(average_precision_score(y, first) - average_precision_score(y, second)), "bootstrap_ci_95_lower": float(np.quantile(values, 0.025)), "bootstrap_ci_95_upper": float(np.quantile(values, 0.975)), "accepted_resamples": len(values), "discarded_resamples": discarded}


def execute(source_root: Path, output: Path) -> None:
    ligands, proteins, affinity, provenance = load_raw(source_root)
    built = construct(ligands, proteins, affinity)
    rows, features = built["rows"], built["features"]
    labels = np.asarray([row["interaction_kd_le_1000_nM"] for row in rows], dtype=int)
    splits = make_splits(rows, labels)
    print("validated raw source, constructed table, and frozen split assignments", flush=True)
    result: dict[str, Any] = {"study": "phase_10r", "status": "EXPERIMENTALLY_SUPPORTED", "raw_provenance": provenance, "construction": {"n_pairs": len(rows), "n_drugs": 68, "n_targets": 442, "label": "interaction_kd_le_1000_nM", "positive_prevalence": float(labels.mean()), "table_sha256": built["table_sha256"]}, "split_seed": SPLIT_SEED, "splits": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    assignment_final = output.parent / "phase_10r_split_assignments.csv"
    prediction_final = output.parent / "phase_10r_test_predictions.csv"
    assignment_path = assignment_final.with_name(assignment_final.name + ".partial")
    prediction_path = prediction_final.with_name(prediction_final.name + ".partial")
    json_partial = output.with_name(output.name + ".partial")
    with assignment_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("observed_pair_index", "split", "partition"))
        writer.writeheader()
        for name, split in splits.items():
            for partition in ("train", "test"):
                for index in split[partition]: writer.writerow({"observed_pair_index": int(index), "split": name, "partition": partition})
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "observed_pair_index", "method", "probability", "label"))
        writer.writeheader()
        for split_offset, (name, split) in enumerate(splits.items()):
            train, test, groups = split["train"], split["test"], split["groups"]
            print(f"starting {name}", flush=True)
            selected, thresholds = select_and_threshold(features, labels, train, groups, name)
            predicted = predictions(features[train], labels[train], features[test])
            y = labels[test]
            entry: dict[str, Any] = {"n_train": len(train), "n_test": len(test), "training_prevalence": float(labels[train].mean()), "test_prevalence": float(y.mean()), "methods": {}, "comparisons": {}}
            final = {method: predicted[method][selected[method]] for method in METHODS}
            for method, probability in final.items():
                entry["methods"][method] = {"selected_k": selected[method], "threshold": thresholds[method], "metrics": metric_summary(y, probability, thresholds[method])}
                for pair, value in zip(test, probability): writer.writerow({"split": name, "observed_pair_index": int(pair), "method": method, "probability": float(value), "label": int(labels[pair])})
            unit = groups[test]
            for comparator_index, comparator in enumerate(("jaccard", "dice", "cosine")):
                analysis = bootstrap_ap(y, final["theta1"], final[comparator], unit, BOOTSTRAP_SEED + 100 * split_offset + comparator_index, name != "pair_random")
                analysis["decision"] = "better" if analysis["difference"] >= 0.02 and analysis["bootstrap_ci_95_lower"] > 0 else "no_supported_advantage"
                entry["comparisons"][comparator] = analysis
            result["splits"][name] = entry
            print(f"completed {name}")
    result["split_assignments_sha256"] = sha256(assignment_path)
    result["prediction_file_sha256"] = sha256(prediction_path)
    json_partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(assignment_path, assignment_final)
    os.replace(prediction_path, prediction_final)
    os.replace(json_partial, output)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the approved Phase 10R study")
    parser.add_argument("--source-root", default=r"G:\Research\STEM\Project10_v02_clean")
    parser.add_argument("--output", default="results/phase_10r_external_dti.json")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing Phase 10R execution without --execute.")
    execute(Path(args.source_root), Path(args.output))


if __name__ == "__main__":
    main()
