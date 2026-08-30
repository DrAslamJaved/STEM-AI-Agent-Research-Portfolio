# Phase 3 Implementation Specification

## Objective

Implement deterministic, leakage-safe splitting and standardization for the
synthetic cybersecurity dataset.

## Scope

Phase 3 implements:

- a normal-only fitting split;
- a normal-only calibration split;
- a labelled test split;
- deterministic split membership;
- immutable raw and standardized split containers;
- normal-fitting-only feature scaling;
- flow-identifier alignment;
- input and leakage validation;
- public package exports.

Phase 3 does not implement PCA fitting, component selection, reconstruction
error, threshold calibration, anomaly prediction, or classification metrics.

## Default partition

Normal observations use these proportions:

- fitting: 0.60;
- calibration: 0.20;
- test: 0.20.

Every attack observation belongs to the test split.

For the default synthetic dataset, the expected shapes are:

- normal fitting: 2,400 rows and 13 columns;
- normal calibration: 800 rows and 13 columns;
- complete test: 1,800 rows and 13 columns.

## Public API

The package must expose:

- `RawDataSplits`;
- `StandardizedDataSplits`;
- `split_normal_calibration_test`;
- `standardize_splits`.

## Split invariants

The implementation must guarantee:

- normal-only fitting;
- normal-only calibration;
- attack-only evaluation access;
- disjoint split identifiers;
- complete identifier coverage;
- deterministic splitting;
- unchanged input data;
- preserved feature order;
- aligned scenario and anomaly labels.

## Standardization contract

The scaler must be `sklearn.preprocessing.StandardScaler`.

It must fit only on the ten numerical features from the normal fitting split.

The following columns must be excluded:

- `flow_id`;
- `is_anomaly`;
- `scenario`.

Fitting features must have means approximately zero and population standard
deviations approximately one.

Calibration and test observations must use the same fitted means and scales.

## Leakage validation

Tests must verify that extreme calibration or test observations do not change
fitted scaler statistics.

Tests must also verify:

- inverse transformation;
- zero-variance rejection;
- nonfinite-value rejection;
- invalid-label rejection;
- scenario consistency;
- identifier-overlap rejection;
- invalid split fractions;
- invalid seeds;
- invalid shuffle values.

## Evidence policy

Record only commands actually executed and outputs actually observed.

Labels may be used to construct and evaluate partitions, but must never be
included among the PCA input features.
