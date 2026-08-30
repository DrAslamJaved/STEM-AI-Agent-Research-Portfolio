# Phase 03 Agent Trace

## Date

2026-08-28

## Starting state

Branch: `feature/project-04-cyber-pca`

Starting commit: `ca2a645`

Starting commit title:
`Implement deterministic synthetic cybersecurity data`

Project 2 remained untracked and outside the Phase 3 scope.

## Scope

Implement deterministic normal-only fitting and calibration partitions,
labelled test partitioning, and leakage-safe standardization.

No PCA cybersecurity experiment, anomaly threshold, prediction, or final
classification evaluation was implemented.

## Contract validation

The preprocessing contract was created and checked by the existing package
documentation tests.

Observed result:

- tests passed: 5;
- failures: 0;
- exit code: 0.

## Configuration red-green cycle

Red result:

- failing test: `test_baseline_configuration_contract`;
- exception: `KeyError`;
- missing key: `preprocessing`;
- tests passed: 4;
- tests failed: 1;
- exit code: 1.

The preprocessing fractions, seed, attack assignment, scaler, excluded
metadata columns, and zero-variance rule were then added to
`configs/baseline.yaml`.

Green result:

- tests passed: 5;
- failures: 0;
- exit code: 0.

## Raw-splitting import red test

The initial preprocessing test failed during collection because
`cyber_pca.preprocessing` did not exist.

Observed result:

- collection errors: 1;
- exception: `ModuleNotFoundError`;
- exit code: 2.

## Raw-splitting implementation

`RawDataSplits` and `split_normal_calibration_test` were implemented.

Observed first-green result:

- focused tests passed: 4;
- failures: 0;
- exit code: 0.

Observed default partitions:

- fitting shape: 2,400 rows and 13 columns;
- calibration shape: 800 rows and 13 columns;
- test shape: 1,800 rows and 13 columns;
- fitting labels: normal only;
- calibration labels: normal only;
- test labels: normal and anomaly;
- test normal observations: 800;
- test attack observations: 1,000.

## Standardization import red test

The standardization test failed during collection because
`StandardizedDataSplits` had not yet been implemented.

Observed result:

- collection errors: 1;
- exception: `ImportError`;
- exit code: 2.

## Standardization implementation

`StandardizedDataSplits` and `standardize_splits` were implemented using
`sklearn.preprocessing.StandardScaler`.

The scaler fits only on the normal fitting feature matrix.

The first focused run collected only four tests because the six new
standardization tests had not been saved in the test file.

A collection check identified the omission. The tests were saved and
collection increased to ten tests.

Corrected result:

- tests collected: 10;
- tests passed: 10;
- failures: 0;
- exit code: 0.

## Expanded validation

Raw validation tests covered deterministic membership, input immutability,
schema order, duplicate identifiers, nonfinite values, invalid fractions,
invalid seeds, invalid shuffle values, and nonempty partition requirements.

Observed result:

- tests collected: 31;
- tests passed: 31;
- failures: 0;
- exit code: 0.

Further tests covered calibration leakage, zero variance, identifier overlap,
nonfinite calibration features, invalid scenarios, and invalid labels.

Observed result:

- tests collected: 38;
- tests passed: 38;
- failures: 0;
- exit code: 0.

## Public-export red-green cycle

The public-export red test failed during collection because preprocessing
symbols were not exported by `cyber_pca`.

Observed red result:

- collection errors: 1;
- exception: `ImportError`;
- exit code: 2.

The package interface was updated. The successful complete regression suite
subsequently verified the public exports.

## Complete regression evidence

Observed result:

- tests collected: 122;
- tests passed: 122;
- failures: 0;
- duration: 11.20 seconds;
- exit code: 0.

## Coverage evidence

Observed result:

- tests collected: 122;
- tests passed: 122;
- failures: 0;
- total coverage: 91.46%;
- required coverage: 90%;
- coverage gate: passed;
- duration: 8.41 seconds;
- exit code: 0.

Coverage artifacts:

- `reports/validation/phase_03_pytest.xml`;
- `reports/validation/phase_03_coverage.xml`.

## Numerical preprocessing evidence

Observed results:

- fitting shape: 2,400 rows and 13 columns;
- calibration shape: 800 rows and 13 columns;
- test shape: 1,800 rows and 13 columns;
- test labels: 800 normal and 1,000 anomaly;
- identifier union complete: true;
- maximum absolute standardized fitting mean:
  `4.100423704282245e-16`;
- maximum population-standard-deviation error:
  `2.220446049250313e-16`;
- same-seed fitting identifier hashes matched;
- different-seed fitting identifier hashes differed;
- feature count: 10.

Seed-42 fitting-identifier SHA-256:

`fd9763c2e230cdc89f1319b79e3d4a113f6fce28a47f7af03aedf9c85863e6c9`

Scaler SHA-256:

`0db3e4facfc81ced3acc5fdec3e1d859118837c8e24aa0566139fbbd0996f0be`

## Pending completion checks

The README update, final documentation validation, compilation check, staged
scope review, commit, push, and remote verification remain pending.

## Final Phase 3 validation

The README was updated to mark leakage-safe splitting and standardization
as completed and normal-only PCA fitting as the next phase.

Final verified results:

- tests collected: 122;
- tests passed: 122;
- failures: 0;
- total coverage: 91.46%;
- required coverage: 90%;
- coverage gate: passed;
- coverage exit code: 0;
- compilation exit code: 0;
- diff check: clean.

Staging, commit, push, and remote verification remain pending.
