# Synthetic Cybersecurity Dataset Contract

## 1. Purpose

The synthetic dataset provides a deterministic development environment for
testing the cybersecurity PCA anomaly-detection workflow.

It is designed to validate:

- data-generation logic;
- leakage-safe preprocessing;
- PCA fitting on normal traffic;
- reconstruction-error calculation;
- normal-only threshold calibration;
- anomaly classification;
- evaluation metrics;
- reproducibility.

Synthetic results must not be presented as evidence of real-world intrusion
detection performance.

## 2. Observation unit

Each row represents one aggregated network-flow observation collected during a
fixed monitoring interval.

The dataset contains numerical traffic features, one binary evaluation label,
and one human-readable scenario label.

## 3. Default dataset composition

| Scenario | Observations | Anomaly label |
|---|---:|---:|
| Normal traffic | 4,000 | 0 |
| Port scan | 250 | 1 |
| Denial of service | 250 | 1 |
| Brute force | 250 | 1 |
| Data exfiltration | 250 | 1 |
| Total | 5,000 | — |

The default random seed is `42`.

## 4. Dataset schema

| Column | Type | Model feature | Description |
|---|---|---:|---|
| `flow_id` | integer | No | Unique row identifier assigned after deterministic shuffling |
| `duration_ms` | float64 | Yes | Flow duration in milliseconds |
| `packets_in` | float64 | Yes | Number of inbound packets |
| `packets_out` | float64 | Yes | Number of outbound packets |
| `bytes_in` | float64 | Yes | Inbound byte volume |
| `bytes_out` | float64 | Yes | Outbound byte volume |
| `syn_count` | float64 | Yes | Number of TCP SYN events |
| `ack_count` | float64 | Yes | Number of TCP acknowledgement events |
| `connection_rate` | float64 | Yes | Connections observed per second |
| `unique_dest_ports` | float64 | Yes | Number of distinct destination ports |
| `failed_logins` | float64 | Yes | Number of failed authentication attempts |
| `is_anomaly` | integer | No | Ground-truth binary label: normal 0, anomaly 1 |
| `scenario` | string | No | Normal or attack-generation scenario |

The model feature order is fixed:

1. `duration_ms`;
2. `packets_in`;
3. `packets_out`;
4. `bytes_in`;
5. `bytes_out`;
6. `syn_count`;
7. `ack_count`;
8. `connection_rate`;
9. `unique_dest_ports`;
10. `failed_logins`.

The columns `flow_id`, `is_anomaly`, and `scenario` must never be supplied to
standardization or PCA.

## 5. Normal-traffic structure

Normal traffic will be generated from a small number of correlated latent
factors.

The intended relationships include:

- traffic intensity influences packet counts and connection rate;
- packet counts and payload size influence byte volumes;
- SYN and acknowledgement counts are correlated;
- inbound and outbound traffic are related but not identical;
- failed login counts remain small for most normal observations.

These correlations create a dominant approximately linear normal-traffic
subspace that PCA can learn.

Random noise must be included so that the data are not perfectly low rank.

## 6. Attack signatures

| Attack | Intended feature behaviour |
|---|---|
| Port scan | High `unique_dest_ports` and `syn_count`, elevated connection rate, and relatively small payload volumes |
| Denial of service | Very high connection rate, packet counts, and SYN activity |
| Brute force | High `failed_logins` and repeated connection activity, usually involving relatively few destination ports |
| Data exfiltration | Abnormally high `bytes_out` and a large outbound-to-inbound byte ratio |

Attack observations should remain stochastic. They must not be constant rows or
exact duplicates.

## 7. Generation requirements

The generator must:

1. use `numpy.random.default_rng`;
2. accept a user-specified integer random seed;
3. reproduce identical data for identical arguments and seed;
4. produce different numerical observations for different seeds;
5. generate finite values only;
6. prevent negative traffic values;
7. preserve the exact schema and column order;
8. assign unique sequential flow identifiers after shuffling;
9. deterministically shuffle the combined dataset;
10. validate observation-count arguments;
11. avoid reading or writing files inside the generator;
12. return a pandas DataFrame.

## 8. Label contract

The following mapping is mandatory:

| Scenario | `is_anomaly` |
|---|---:|
| `normal` | 0 |
| `port_scan` | 1 |
| `dos` | 1 |
| `brute_force` | 1 |
| `exfiltration` | 1 |

Labels exist only for evaluation. They must not influence standardization,
principal-component selection, PCA fitting, reconstruction, or threshold
calibration.

## 9. Validation requirements

Automated tests must verify:

1. exact schema and column order;
2. expected total and per-scenario row counts;
3. label-to-scenario consistency;
4. unique flow identifiers;
5. finite feature values;
6. nonnegative feature values;
7. float64 model-feature columns;
8. deterministic reproduction using the same seed;
9. changed numerical observations using a different seed;
10. absence of duplicate rows within each attack scenario;
11. statistically visible attack signatures;
12. rejection of invalid observation counts and invalid seeds.

Attack-signature tests should compare robust statistics such as medians rather
than exact randomly generated values.

## 10. Scope limitation

This dataset is a controlled mathematical test fixture. It simplifies real
network behaviour and does not reproduce every protocol, topology, attack
strategy, or form of concept drift.

The final cybersecurity evaluation will use UNSW-NB15 after the complete
synthetic workflow has passed its validation gates.
