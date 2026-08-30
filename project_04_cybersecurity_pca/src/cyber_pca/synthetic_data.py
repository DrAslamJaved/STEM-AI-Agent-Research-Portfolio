"""Deterministic synthetic cybersecurity network-flow data."""

from __future__ import annotations

from numbers import Integral
from typing import Final

import numpy as np
import pandas as pd
from numpy.random import Generator


FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "duration_ms",
    "packets_in",
    "packets_out",
    "bytes_in",
    "bytes_out",
    "syn_count",
    "ack_count",
    "connection_rate",
    "unique_dest_ports",
    "failed_logins",
)

ATTACK_TYPES: Final[tuple[str, ...]] = (
    "port_scan",
    "dos",
    "brute_force",
    "exfiltration",
)

OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "flow_id",
    *FEATURE_COLUMNS,
    "is_anomaly",
    "scenario",
)

DEFAULT_NORMAL_OBSERVATIONS: Final[int] = 4000
DEFAULT_ATTACK_OBSERVATIONS_PER_TYPE: Final[int] = 250
DEFAULT_RANDOM_SEED: Final[int] = 42


def _validate_positive_integer(
    value: object,
    *,
    name: str,
) -> int:
    """Validate a positive integer generation argument."""

    if isinstance(value, bool) or not isinstance(
        value,
        Integral,
    ):
        raise TypeError(f"{name} must be an integer.")

    integer_value = int(value)

    if integer_value <= 0:
        raise ValueError(f"{name} must be positive.")

    return integer_value


def _validate_random_seed(value: object) -> int:
    """Validate a nonnegative integer random seed."""

    if isinstance(value, bool) or not isinstance(
        value,
        Integral,
    ):
        raise TypeError("random_seed must be an integer.")

    integer_value = int(value)

    if integer_value < 0:
        raise ValueError(
            "random_seed must be nonnegative."
        )

    return integer_value


def _generate_normal_features(
    rng: Generator,
    n_observations: int,
) -> pd.DataFrame:
    """Generate correlated normal network-flow features."""

    activity = rng.lognormal(
        mean=2.2,
        sigma=0.35,
        size=n_observations,
    )

    inbound_balance = rng.normal(
        loc=1.0,
        scale=0.10,
        size=n_observations,
    )

    outbound_balance = rng.normal(
        loc=0.92,
        scale=0.12,
        size=n_observations,
    )

    packets_in = np.maximum(
        1.0,
        activity * 2.8 * inbound_balance
        + rng.normal(
            loc=0.0,
            scale=1.0,
            size=n_observations,
        ),
    )

    packets_out = np.maximum(
        1.0,
        activity * 2.4 * outbound_balance
        + rng.normal(
            loc=0.0,
            scale=1.0,
            size=n_observations,
        ),
    )

    inbound_payload = rng.lognormal(
        mean=6.1,
        sigma=0.22,
        size=n_observations,
    )

    outbound_payload = rng.lognormal(
        mean=6.0,
        sigma=0.24,
        size=n_observations,
    )

    bytes_in = np.maximum(
        64.0,
        packets_in
        * inbound_payload
        * rng.normal(
            loc=1.0,
            scale=0.04,
            size=n_observations,
        ),
    )

    bytes_out = np.maximum(
        64.0,
        packets_out
        * outbound_payload
        * rng.normal(
            loc=1.0,
            scale=0.04,
            size=n_observations,
        ),
    )

    syn_count = np.maximum(
        0.0,
        activity * 0.24
        + rng.normal(
            loc=0.0,
            scale=0.30,
            size=n_observations,
        ),
    )

    ack_count = np.maximum(
        0.0,
        syn_count * 0.94
        + rng.normal(
            loc=0.0,
            scale=0.22,
            size=n_observations,
        ),
    )

    connection_rate = np.maximum(
        0.01,
        activity * 0.16
        + rng.normal(
            loc=0.0,
            scale=0.08,
            size=n_observations,
        ),
    )

    duration_ms = np.maximum(
        1.0,
        180.0
        + activity * 42.0
        + rng.normal(
            loc=0.0,
            scale=45.0,
            size=n_observations,
        ),
    )

    unique_dest_ports = (
        rng.poisson(
            lam=2.0,
            size=n_observations,
        )
        + 1
    ).astype(np.float64)

    failed_logins = rng.binomial(
        n=2,
        p=0.04,
        size=n_observations,
    ).astype(np.float64)

    return pd.DataFrame(
        {
            "duration_ms": duration_ms,
            "packets_in": packets_in,
            "packets_out": packets_out,
            "bytes_in": bytes_in,
            "bytes_out": bytes_out,
            "syn_count": syn_count,
            "ack_count": ack_count,
            "connection_rate": connection_rate,
            "unique_dest_ports": unique_dest_ports,
            "failed_logins": failed_logins,
        },
        dtype=np.float64,
    )


def _generate_attack_features(
    rng: Generator,
    n_observations: int,
    attack_type: str,
) -> pd.DataFrame:
    """Generate one stochastic attack scenario."""

    features = _generate_normal_features(
        rng,
        n_observations,
    )

    if attack_type == "port_scan":
        features["duration_ms"] *= rng.uniform(
            0.25,
            0.70,
            size=n_observations,
        )
        features["unique_dest_ports"] += rng.lognormal(
            mean=4.0,
            sigma=0.35,
            size=n_observations,
        )
        features["syn_count"] += rng.lognormal(
            mean=3.2,
            sigma=0.30,
            size=n_observations,
        )
        features["connection_rate"] *= rng.uniform(
            4.0,
            8.0,
            size=n_observations,
        )
        features["bytes_in"] *= rng.uniform(
            0.10,
            0.35,
            size=n_observations,
        )
        features["bytes_out"] *= rng.uniform(
            0.10,
            0.35,
            size=n_observations,
        )
        features["ack_count"] *= rng.uniform(
            0.20,
            0.60,
            size=n_observations,
        )

    elif attack_type == "dos":
        features["duration_ms"] *= rng.uniform(
            0.20,
            0.80,
            size=n_observations,
        )
        features["packets_in"] *= rng.uniform(
            7.0,
            12.0,
            size=n_observations,
        )
        features["packets_out"] *= rng.uniform(
            5.0,
            10.0,
            size=n_observations,
        )
        features["bytes_in"] *= rng.uniform(
            5.0,
            10.0,
            size=n_observations,
        )
        features["bytes_out"] *= rng.uniform(
            4.0,
            8.0,
            size=n_observations,
        )
        features["syn_count"] *= rng.uniform(
            8.0,
            14.0,
            size=n_observations,
        )
        features["ack_count"] *= rng.uniform(
            2.0,
            5.0,
            size=n_observations,
        )
        features["connection_rate"] *= rng.uniform(
            10.0,
            20.0,
            size=n_observations,
        )

    elif attack_type == "brute_force":
        features["duration_ms"] *= rng.uniform(
            1.0,
            2.0,
            size=n_observations,
        )
        features["failed_logins"] += (
            rng.poisson(
                lam=12.0,
                size=n_observations,
            )
            + 5.0
        )
        features["connection_rate"] *= rng.uniform(
            3.0,
            6.0,
            size=n_observations,
        )
        features["syn_count"] *= rng.uniform(
            2.0,
            4.0,
            size=n_observations,
        )
        features["unique_dest_ports"] = np.clip(
            features["unique_dest_ports"],
            1.0,
            3.0,
        )

    elif attack_type == "exfiltration":
        features["duration_ms"] *= rng.uniform(
            1.5,
            3.0,
            size=n_observations,
        )
        features["packets_out"] *= rng.uniform(
            2.0,
            4.0,
            size=n_observations,
        )
        features["bytes_in"] *= rng.uniform(
            0.70,
            1.30,
            size=n_observations,
        )
        features["bytes_out"] *= rng.uniform(
            12.0,
            25.0,
            size=n_observations,
        )

    else:
        raise ValueError(
            f"Unsupported attack type: {attack_type}"
        )

    return features.astype(np.float64)


def _add_labels(
    features: pd.DataFrame,
    *,
    scenario: str,
    is_anomaly: int,
) -> pd.DataFrame:
    """Attach evaluation metadata to feature rows."""

    labelled = features.copy()
    labelled["is_anomaly"] = np.full(
        len(labelled),
        is_anomaly,
        dtype=np.int64,
    )
    labelled["scenario"] = scenario

    return labelled


def generate_synthetic_network_data(
    n_normal: int = DEFAULT_NORMAL_OBSERVATIONS,
    n_attack_per_type: int = (
        DEFAULT_ATTACK_OBSERVATIONS_PER_TYPE
    ),
    random_seed: int = DEFAULT_RANDOM_SEED,
    *,
    shuffle: bool = True,
) -> pd.DataFrame:
    """Generate a deterministic synthetic network-flow dataset."""

    validated_normal_count = _validate_positive_integer(
        n_normal,
        name="n_normal",
    )

    validated_attack_count = _validate_positive_integer(
        n_attack_per_type,
        name="n_attack_per_type",
    )

    validated_seed = _validate_random_seed(random_seed)

    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a boolean.")

    rng = np.random.default_rng(validated_seed)

    frames = [
        _add_labels(
            _generate_normal_features(
                rng,
                validated_normal_count,
            ),
            scenario="normal",
            is_anomaly=0,
        )
    ]

    for attack_type in ATTACK_TYPES:
        frames.append(
            _add_labels(
                _generate_attack_features(
                    rng,
                    validated_attack_count,
                    attack_type,
                ),
                scenario=attack_type,
                is_anomaly=1,
            )
        )

    dataset = pd.concat(
        frames,
        axis=0,
        ignore_index=True,
    )

    if shuffle:
        shuffled_indices = rng.permutation(len(dataset))
        dataset = dataset.iloc[
            shuffled_indices
        ].reset_index(drop=True)

    dataset.insert(
        0,
        "flow_id",
        np.arange(
            len(dataset),
            dtype=np.int64,
        ),
    )

    return dataset.loc[:, OUTPUT_COLUMNS].copy()
