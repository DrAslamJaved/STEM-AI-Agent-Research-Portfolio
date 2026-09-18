"""Execute the approved Phase 09 robustness study only with ``--execute``.

The study deliberately keeps fuzzy sigma-count cardinality fixed while adding
zero-sum bounded perturbations to non-collinear cluster prototypes.  It writes
replicate-level results and paired inferential summaries to JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzy_similarity import (  # noqa: E402
    Parameters,
    fuzzy_cosine,
    fuzzy_dice,
    fuzzy_jaccard,
    rational_similarity,
)


CENTRES = (
    (0.70, 0.70, 0.30, 0.30, 0.70, 0.70, 0.30, 0.30, 0.50, 0.50),
    (0.70, 0.30, 0.70, 0.30, 0.70, 0.30, 0.70, 0.30, 0.50, 0.50),
    (0.70, 0.30, 0.30, 0.70, 0.30, 0.70, 0.70, 0.30, 0.50, 0.50),
)
EPSILONS = (0.00, 0.04, 0.08, 0.12, 0.16, 0.20)
N_PER_CLUSTER = 100
N_REPLICATIONS = 100
N_CLUSTERS = 3
BOOTSTRAP_RESAMPLES = 10_000
PERMUTATION_RESAMPLES = 10_000
CARDINALITY_TOLERANCE = 1e-12

THETA_1 = Parameters(1, 0, 0, 0, 0, 1, 1, 1, 0, 1)
METHODS: dict[str, Callable[[Iterable[float], Iterable[float]], float]] = {
    "theta1": lambda u, v: rational_similarity(u, v, THETA_1),
    "jaccard": fuzzy_jaccard,
    "dice": fuzzy_dice,
    "cosine": fuzzy_cosine,
}


def zero_sum_perturbation(rng: random.Random, epsilon: float, dimension: int) -> list[float]:
    """Return bounded noise with zero sum and maximum absolute value epsilon."""
    if epsilon == 0.0:
        return [0.0] * dimension
    raw = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
    mean = sum(raw) / dimension
    centered = [value - mean for value in raw]
    maximum = max(abs(value) for value in centered)
    if maximum == 0.0:  # A theoretical safeguard for a constant draw.
        return [0.0] * dimension
    return [epsilon * value / maximum for value in centered]


def generate_dataset(seed: int, epsilon: float) -> tuple[list[list[float]], list[int]]:
    """Generate one cardinality-preserving labelled fuzzy dataset."""
    rng = random.Random(seed)
    samples: list[list[float]] = []
    labels: list[int] = []
    for label, centre in enumerate(CENTRES):
        target = sum(centre)
        for _ in range(N_PER_CLUSTER):
            noise = zero_sum_perturbation(rng, epsilon, len(centre))
            vector = [base + perturbation for base, perturbation in zip(centre, noise)]
            if not all(0.0 <= value <= 1.0 for value in vector):
                raise ValueError("generated membership is outside [0, 1]")
            if not math.isclose(sum(vector), target, abs_tol=CARDINALITY_TOLERANCE):
                raise ValueError("generated vector does not preserve sigma-count cardinality")
            samples.append(vector)
            labels.append(label)
    return samples, labels


def dataset_fingerprint(samples: list[list[float]]) -> str:
    """Return a compact reproducibility fingerprint without storing all samples."""
    canonical = json.dumps(samples, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def adjusted_rand_index(true_labels: list[int], predicted_labels: list[int]) -> float:
    """Calculate ARI without a third-party dependency."""
    n = len(true_labels)
    if n != len(predicted_labels) or n < 2:
        raise ValueError("label vectors must have equal length of at least two")
    total_pairs = n * (n - 1) // 2
    true_counts = Counter(true_labels)
    predicted_counts = Counter(predicted_labels)
    joint_counts = Counter(zip(true_labels, predicted_labels))
    choose_two = lambda count: count * (count - 1) // 2
    index = sum(choose_two(count) for count in joint_counts.values())
    expected = (
        sum(choose_two(count) for count in true_counts.values())
        * sum(choose_two(count) for count in predicted_counts.values())
        / total_pairs
    )
    maximum = (
        sum(choose_two(count) for count in true_counts.values())
        + sum(choose_two(count) for count in predicted_counts.values())
    ) / 2
    if maximum == expected:
        return 1.0 if true_labels == predicted_labels else 0.0
    return (index - expected) / (maximum - expected)


def average_linkage(similarity: list[list[float]], n_clusters: int = N_CLUSTERS) -> list[int]:
    """Deterministic average-linkage clustering, merging largest similarity first."""
    n_samples = len(similarity)
    active = {index: 1 for index in range(n_samples)}
    members = {index: [index] for index in range(n_samples)}
    scores: dict[tuple[int, int], float] = {}
    heap: list[tuple[float, int, int]] = []

    def add_score(left: int, right: int, value: float) -> None:
        key = (min(left, right), max(left, right))
        scores[key] = value
        heapq.heappush(heap, (-value, key[0], key[1]))

    for left in range(n_samples):
        for right in range(left + 1, n_samples):
            add_score(left, right, similarity[left][right])

    next_cluster = n_samples
    while len(active) > n_clusters:
        while True:
            negative_score, left, right = heapq.heappop(heap)
            key = (left, right)
            if left in active and right in active and scores.get(key) == -negative_score:
                break
        left_size = active.pop(left)
        right_size = active.pop(right)
        merged = next_cluster
        next_cluster += 1
        members[merged] = members.pop(left) + members.pop(right)
        active[merged] = left_size + right_size
        for other, other_size in list(active.items()):
            if other == merged:
                continue
            left_score = scores[(min(left, other), max(left, other))]
            right_score = scores[(min(right, other), max(right, other))]
            add_score(
                merged,
                other,
                (left_size * left_score + right_size * right_score) / (left_size + right_size),
            )

    labels = [0] * n_samples
    for label, cluster_members in enumerate(members.values()):
        for item in cluster_members:
            labels[item] = label
    return labels


def run_method(
    samples: list[list[float]], labels: list[int], similarity_function: Callable[[Iterable[float], Iterable[float]], float]
) -> dict[str, object]:
    """Cluster one dataset under one similarity and return pre-registered metrics."""
    n_samples = len(samples)
    matrix = [[1.0] * n_samples for _ in range(n_samples)]
    within: list[float] = []
    between: list[float] = []
    for left in range(n_samples):
        for right in range(left + 1, n_samples):
            value = similarity_function(samples[left], samples[right])
            matrix[left][right] = matrix[right][left] = value
            (within if labels[left] == labels[right] else between).append(value)
    predicted = average_linkage(matrix)
    cluster_sizes = [predicted.count(label) for label in range(N_CLUSTERS)]
    if any(size == 0 for size in cluster_sizes):
        raise ValueError("clustering returned an empty cluster")
    within_mean = sum(within) / len(within)
    between_mean = sum(between) / len(between)
    return {
        "ari": adjusted_rand_index(labels, predicted),
        "within_mean": within_mean,
        "between_mean": between_mean,
        "separation_gap": within_mean - between_mean,
        "cluster_sizes": cluster_sizes,
    }


def descriptive_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Return population descriptive summaries required by the protocol."""
    output: dict[str, dict[str, float]] = {}
    for metric in ("ari", "within_mean", "between_mean", "separation_gap"):
        values = [float(row[metric]) for row in rows]
        mean = sum(values) / len(values)
        output[metric] = {
            "mean": mean,
            "std": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
            "min": min(values),
            "max": max(values),
        }
    return output


def percentile(values: list[float], probability: float) -> float:
    """Linearly interpolated quantile for a non-empty list."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def paired_analysis(differences: list[float], bootstrap_seed: int, permutation_seed: int) -> dict[str, float]:
    """Calculate the approved bootstrap CI and two-sided sign-flip p-value."""
    if not differences:
        raise ValueError("paired analysis needs at least one difference")
    observed_mean = sum(differences) / len(differences)
    bootstrap_rng = random.Random(bootstrap_seed)
    bootstrap_means = [
        sum(bootstrap_rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    permutation_rng = random.Random(permutation_seed)
    extreme = 0
    for _ in range(PERMUTATION_RESAMPLES):
        signed_mean = sum(
            value if permutation_rng.getrandbits(1) else -value for value in differences
        ) / len(differences)
        if abs(signed_mean) >= abs(observed_mean):
            extreme += 1
    return {
        "mean_difference": observed_mean,
        "bootstrap_ci_95_lower": percentile(bootstrap_means, 0.025),
        "bootstrap_ci_95_upper": percentile(bootstrap_means, 0.975),
        "permutation_p_value_two_sided": (extreme + 1) / (PERMUTATION_RESAMPLES + 1),
    }


def holm_adjustment(raw_p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm-adjusted p-values keyed by comparator name."""
    ordered = sorted(raw_p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    total = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (total - index) * p_value)
        running_maximum = max(running_maximum, candidate)
        adjusted[name] = running_maximum
    return adjusted


def conclusion(analysis: dict[str, float]) -> str:
    """Apply the pre-registered one-condition advantage rule."""
    supported = (
        analysis["mean_difference"] >= 0.02
        and analysis["bootstrap_ci_95_lower"] > 0.0
        and analysis["holm_adjusted_permutation_p_value"] < 0.05
    )
    return "supported_advantage_on_this_condition" if supported else "no_supported_advantage"


def condition_seed(condition_index: int, replication_index: int) -> int:
    return 20261000 + 100 * condition_index + replication_index


def run_study() -> dict[str, object]:
    """Run all 600 fixed replications and produce a JSON-serializable result."""
    conditions: list[dict[str, object]] = []
    for condition_index, epsilon in enumerate(EPSILONS):
        method_rows: dict[str, list[dict[str, object]]] = {name: [] for name in METHODS}
        fingerprints: list[str] = []
        for replication_index in range(N_REPLICATIONS):
            seed = condition_seed(condition_index, replication_index)
            samples, labels = generate_dataset(seed, epsilon)
            fingerprints.append(dataset_fingerprint(samples))
            for name, function in METHODS.items():
                metrics = run_method(samples, labels, function)
                method_rows[name].append(
                    {
                        "replication": replication_index,
                        "seed": seed,
                        "dataset_sha256": fingerprints[-1],
                        **metrics,
                    }
                )

        comparisons: dict[str, dict[str, float | str]] = {}
        raw_p_values: dict[str, float] = {}
        for comparator in ("jaccard", "dice", "cosine"):
            differences = [
                float(theta["ari"]) - float(other["ari"])
                for theta, other in zip(method_rows["theta1"], method_rows[comparator])
            ]
            analysis = paired_analysis(
                differences,
                bootstrap_seed=20261099 + 1_000 * condition_index + len(comparisons),
                permutation_seed=20261100 + 1_000 * condition_index + len(comparisons),
            )
            comparisons[comparator] = analysis
            raw_p_values[comparator] = analysis["permutation_p_value_two_sided"]
        adjusted = holm_adjustment(raw_p_values)
        for comparator, adjusted_p_value in adjusted.items():
            comparisons[comparator]["holm_adjusted_permutation_p_value"] = adjusted_p_value
            comparisons[comparator]["decision"] = conclusion(comparisons[comparator])

        conditions.append(
            {
                "condition_index": condition_index,
                "epsilon": epsilon,
                "seeds": [condition_seed(condition_index, index) for index in range(N_REPLICATIONS)],
                "dataset_sha256": fingerprints,
                "methods": {
                    name: {"summary": descriptive_summary(rows), "replications": rows}
                    for name, rows in method_rows.items()
                },
                "paired_ari_comparisons": comparisons,
            }
        )
        print(f"completed epsilon={epsilon:.2f} ({condition_index + 1}/{len(EPSILONS)})", flush=True)

    return {
        "status": "executed",
        "protocol": "phase_09_robustness_protocol",
        "theta1_parameters": [1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
        "n_conditions": len(EPSILONS),
        "epsilons": list(EPSILONS),
        "n_replications_per_condition": N_REPLICATIONS,
        "n_samples_per_replication": N_PER_CLUSTER * N_CLUSTERS,
        "dimension": len(CENTRES[0]),
        "samples_per_cluster": N_PER_CLUSTER,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "permutation_resamples": PERMUTATION_RESAMPLES,
        "cardinality_target": 5.0,
        "conditions": conditions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the approved study")
    parser.add_argument("--output", default="results/phase_09_robustness.json")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing execution without --execute.")
    result = run_study()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
