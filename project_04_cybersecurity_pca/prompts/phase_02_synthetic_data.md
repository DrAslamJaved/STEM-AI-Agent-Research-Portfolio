# Phase 2 Implementation Specification

## Objective

Implement a deterministic synthetic cybersecurity network-flow dataset for
developing and validating the PCA anomaly-detection workflow.

## Scope

Phase 2 implements:

- a documented synthetic-data contract;
- configuration-controlled dataset defaults;
- correlated normal network traffic;
- port-scan observations;
- denial-of-service observations;
- brute-force observations;
- data-exfiltration observations;
- deterministic shuffling;
- binary anomaly labels;
- scenario labels;
- input validation;
- package exports;
- automated structural and statistical tests.

Phase 2 does not implement:

- preprocessing splits;
- feature standardization;
- PCA fitting on cybersecurity data;
- threshold calibration;
- anomaly prediction;
- final UNSW-NB15 evaluation.

## Default dataset

The default dataset contains:

- 4,000 normal observations;
- 250 port-scan observations;
- 250 denial-of-service observations;
- 250 brute-force observations;
- 250 data-exfiltration observations.

The total default size is 5,000 observations.

The default random seed is 42.

## Public API

The package must expose:

- `FEATURE_COLUMNS`;
- `ATTACK_TYPES`;
- `OUTPUT_COLUMNS`;
- `generate_synthetic_network_data`.

The generator must return a pandas DataFrame and must not read or write files.

## Feature contract

The ten model features are:

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

The identifier, label, and scenario columns must not be model features.

## Mathematical design

Normal observations must contain correlated feature structure so that PCA can
learn an approximately linear normal-traffic subspace.

Random noise must prevent the normal data from being perfectly low rank.

Attack scenarios must perturb interpretable feature combinations while
remaining stochastic.

## Validation requirements

Tests must verify:

- exact schema and column order;
- default and per-scenario row counts;
- label consistency;
- float64 model features;
- finite and nonnegative values;
- unique sequential identifiers;
- identical output for identical seeds;
- changed output for different seeds;
- absence of duplicate attack rows;
- statistically visible attack signatures;
- deterministic shuffling;
- rejection of invalid counts, seeds, and shuffle values;
- successful package exports.

Attack signatures must be tested using medians or other robust statistics,
rather than exact random values.

## Evidence policy

Record only commands actually executed and results actually observed.

Synthetic results validate software and experimental design. They must not be
reported as real-world intrusion-detection performance.
