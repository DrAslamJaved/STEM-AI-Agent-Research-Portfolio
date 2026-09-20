from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd


ALLOWED_SCENARIOS = (
    "randomized",
    "observed_confounding",
    "nonlinear",
    "heterogeneous",
    "hidden_confounding",
    "positivity_stress",
    "null",
)


@dataclass(frozen=True)
class SyntheticDataset:
    """Observed synthetic data accompanied by oracle causal-effect truth."""

    data: pd.DataFrame
    true_ate: float
    true_ite: np.ndarray
    scenario: str
    seed: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def generate_synthetic(
    scenario: str = "observed_confounding", n: int = 1000, seed: int = 202608
) -> SyntheticDataset:
    """Generate a reproducible structural-causal-model benchmark dataset.

    The returned frame contains only variables available to an estimator. The
    hidden-confounding scenario uses an unobserved variable in both treatment
    assignment and the outcome equation but intentionally omits it from the
    observed frame. ``true_ite`` records Y(1) - Y(0) under common exogenous
    noise, and ``true_ate`` is its finite-sample mean.
    """
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError("n must be an integer")
    if n < 50:
        raise ValueError("n must be at least 50")
    if not isinstance(scenario, str):
        raise TypeError("scenario must be a string")
    if scenario not in ALLOWED_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    n = int(n)
    seed = int(seed)
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    hidden_u = rng.normal(size=n)
    logits = np.zeros(n)
    if scenario != "randomized":
        logits = 0.8 * x1 - 0.5 * x2
    if scenario == "nonlinear":
        logits += 0.8 * np.sin(x1) + 0.3 * x2**2
    if scenario == "hidden_confounding":
        logits += 1.1 * hidden_u
    if scenario == "positivity_stress":
        logits = 5.0 * x1
    propensity = _sigmoid(logits)
    treatment = rng.binomial(1, propensity)
    tau = np.full(n, 0.0 if scenario == "null" else 2.0)
    if scenario == "heterogeneous":
        tau = 1.0 + 0.8 * x1
    baseline = 1.5 * x1 - x2
    if scenario == "nonlinear":
        baseline += np.sin(2 * x1) + 0.5 * x2**2
    if scenario == "hidden_confounding":
        baseline += 1.2 * hidden_u
    outcome = baseline + tau * treatment + rng.normal(scale=1.0, size=n)
    frame = pd.DataFrame(
        {"X1": x1, "X2": x2, "T": treatment, "Y": outcome, "propensity": propensity}
    )
    return SyntheticDataset(frame, float(np.mean(tau)), tau, scenario, seed)
