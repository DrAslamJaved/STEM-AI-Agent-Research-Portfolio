"""Selection and evaluation on arrays; no Matbench dependency in this module."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import pairwise_distances


POLICIES = ("random", "uncertainty", "uncertainty_diversity")


@dataclass(frozen=True)
class Settings:
    budgets: tuple[int, ...] = (200, 400, 800, 1600)
    initial: int = 200
    calibration_fraction: float = 0.15
    members: int = 5
    trees: int = 64
    alpha: float = 0.10

    def validate(self, n_train: int) -> None:
        if not (0 < self.calibration_fraction < 0.5):
            raise ValueError("calibration_fraction must lie in (0, 0.5)")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must lie in (0, 1)")
        if self.members < 2 or self.trees < 1:
            raise ValueError("at least two members and one tree are required")
        if not self.budgets or self.initial != self.budgets[0] or tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("budgets must increase strictly and start at initial")
        if self.initial < 2 or self.budgets[-1] >= n_train:
            raise ValueError("budgets must fit within the training fold after calibration")


def conformal_radius(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal absolute-residual quantile."""
    errors = np.asarray(residuals, dtype=float)
    if errors.ndim != 1 or len(errors) == 0 or not np.all(np.isfinite(errors)):
        raise ValueError("residuals must be a nonempty finite 1-D array")
    rank = ceil((len(errors) + 1) * (1 - alpha))
    if rank > len(errors):
        return float("inf")
    return float(np.partition(errors, rank - 1)[rank - 1])


def make_partition(groups: np.ndarray, n: int, seed: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep equivalent compositions together when reserving calibration data."""
    groups = np.asarray(groups)
    if len(groups) != n or len(np.unique(groups)) < 2:
        raise ValueError("need at least two composition groups")
    pool, cal = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(np.zeros(n), groups=groups))
    assert not set(groups[pool]).intersection(groups[cal])
    return pool, cal


def _fit(x: np.ndarray, y: np.ndarray, selected: np.ndarray, seed: int, settings: Settings):
    single = RandomForestRegressor(n_estimators=settings.trees, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    single.fit(x[selected], y[selected])
    rng = np.random.default_rng(seed + 899)
    ensemble = []
    for i in range(settings.members):
        sample = rng.choice(selected, size=len(selected), replace=True)
        model = RandomForestRegressor(n_estimators=settings.trees, min_samples_leaf=2, n_jobs=-1, random_state=seed + i + 1000)
        model.fit(x[sample], y[sample])
        ensemble.append(model)
    return single, ensemble


def _pred(models: list, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.stack([m.predict(x) for m in models])
    return predictions.mean(axis=0), predictions.std(axis=0, ddof=1)


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, scaled to [0, 1]."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - (counts + 1) / 2)[inverse] / max(len(values) - 1, 1))


def diversity_scores(x_train: np.ndarray, selected: np.ndarray, remaining: np.ndarray,
                     disagreement: np.ndarray) -> np.ndarray:
    """Equal-weight rank of disagreement and nearest labelled composition distance.

    Only the 118 element-fraction columns contribute to distance. This function
    sees features and model disagreement; it never receives pool targets.
    """
    if x_train.shape[1] < 118:
        raise ValueError("diversity acquisition requires 118 element-fraction columns")
    if len(disagreement) != len(remaining):
        raise ValueError("disagreement must match remaining pool")
    nearest = pairwise_distances(x_train[remaining, :118], x_train[selected, :118]).min(axis=1)
    return 0.5 * (_percentile_ranks(disagreement) + _percentile_ranks(nearest))


def simulate(
    x_train: np.ndarray, y_train: np.ndarray, train_groups: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray, *, seed: int, settings: Settings,
    include_diversity: bool = False,
) -> list[dict]:
    """Run paired policies. y_test is consulted only while computing checkpoint metrics."""
    x_train, x_test = np.asarray(x_train, float), np.asarray(x_test, float)
    y_train, y_test = np.asarray(y_train, float), np.asarray(y_test, float)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("train and test feature matrices must have the same columns")
    if len(x_train) != len(y_train) or len(x_test) != len(y_test) or not len(y_test):
        raise ValueError("feature and target row counts differ or test is empty")
    if not all(np.all(np.isfinite(a)) for a in (x_train, x_test, y_train, y_test)):
        raise ValueError("features and targets must be finite")
    settings.validate(len(y_train))
    pool, cal = make_partition(train_groups, len(y_train), seed, settings.calibration_fraction)
    if settings.budgets[-1] > len(pool):
        raise ValueError(f"largest budget {settings.budgets[-1]} exceeds available pool {len(pool)}")
    rng = np.random.default_rng(seed)
    initial = rng.choice(pool, size=settings.initial, replace=False)
    output = []
    if include_diversity and x_train.shape[1] < 118:
        raise ValueError("diversity acquisition requires 118 element-fraction columns")
    for policy in POLICIES if include_diversity else POLICIES[:2]:
        selected = initial.copy()
        remaining = np.setdiff1d(pool, selected)
        policy_rng = np.random.default_rng(seed + 2048)
        for step, budget in enumerate(settings.budgets):
            assert len(selected) == budget
            single, ensemble = _fit(x_train, y_train, selected, seed + step, settings)
            center_cal, _ = _pred(ensemble, x_train[cal])
            radius = conformal_radius(np.abs(y_train[cal] - center_cal), settings.alpha)
            center_test, spread_test = _pred(ensemble, x_test)
            predictions = {
                "mean": np.full(len(y_test), float(np.mean(y_train[selected]))),
                "forest": single.predict(x_test),
                "ensemble": center_test,
            }
            for name, prediction in predictions.items():
                row = {
                    "seed": int(seed), "policy": policy, "budget": int(budget), "model": name,
                    "mae_ev": float(mean_absolute_error(y_test, prediction)),
                    "rmse_ev": float(np.sqrt(mean_squared_error(y_test, prediction))),
                    "test_count": len(y_test), "calibration_count": len(cal),
                }
                if name == "ensemble":
                    row.update({
                        "coverage_90": float(np.mean(np.abs(y_test - prediction) <= radius)),
                        "interval_width_ev": float(2 * radius),
                        "mean_disagreement_ev": float(np.mean(spread_test)),
                        "selected_train_positions": selected.tolist(),
                        "calibration_train_positions": cal.tolist(),
                    })
                output.append(row)
            if step + 1 == len(settings.budgets):
                continue
            batch = settings.budgets[step + 1] - budget
            if policy == "random":
                chosen = policy_rng.choice(remaining, size=batch, replace=False)
            else:
                _, spread = _pred(ensemble, x_train[remaining])
                score = (diversity_scores(x_train, selected, remaining, spread)
                         if policy == "uncertainty_diversity" else spread)
                tie_order = policy_rng.permutation(len(remaining))
                chosen = remaining[tie_order[np.argsort(-score[tie_order], kind="stable")[:batch]]]
            selected = np.concatenate([selected, chosen])
            remaining = np.setdiff1d(remaining, chosen)
    return output


def crossing_summary(rows: list[dict], target_mae: float,
                     policies: tuple[str, ...] = POLICIES[:2]) -> list[dict]:
    """First scheduled budget at or below the locked MAE, or null."""
    if not np.isfinite(target_mae) or target_mae <= 0:
        raise ValueError("target MAE must be positive and finite")
    keys = sorted({(r["fold"], r["seed"]) for r in rows})
    summary = []
    for fold, seed in keys:
        cross = {}
        for policy in policies:
            eligible = sorted(r["budget"] for r in rows if r["fold"] == fold and r["seed"] == seed
                              and r["policy"] == policy and r["model"] == "ensemble"
                              and r["mae_ev"] <= target_mae)
            cross[policy] = eligible[0] if eligible else None
        saved = (cross["random"] - cross["uncertainty"]
                 if cross["random"] is not None and cross["uncertainty"] is not None else None)
        entry = {"fold": fold, "seed": seed, "first_budget": cross,
                 "labels_saved_vs_random": saved}
        if "uncertainty_diversity" in policies:
            hybrid = cross["uncertainty_diversity"]
            entry["labels_saved_diversity_vs_random"] = (
                cross["random"] - hybrid if cross["random"] is not None and hybrid is not None else None)
        summary.append(entry)
    return summary
