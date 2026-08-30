# Phase 05 Agent Trace

## Date

2026-08-28

## Scope

Implement and validate standardized-space PCA reconstruction errors,
normal-calibration-only threshold selection, and strict binary anomaly
prediction.

No labelled evaluation metric, plot, UNSW-NB15 experiment, or agent reasoning
score was produced during this phase.

## Starting state

- branch: `feature/project-04-cyber-pca`;
- starting commit: `44bc1ca`;
- local/remote divergence: `0 0`;
- Project 2 remained untracked and outside the Phase 5 scope;
- selected PCA components from Phase 4: 5;
- explained-variance target: 0.95;
- threshold quantile: 0.99;
- threshold comparison: strictly greater than.

## Contract and configuration

The reconstruction-error contract was created at
`docs/reconstruction_error_contract.md`.

One initial package-integrity invocation returned exit code 120 without a test
summary. An explicit rerun with the project interpreter completed successfully:

- tests passed: 6;
- failures: 0;
- exit code: 0.

A configuration test initially failed with:

- failure: `KeyError: 'quantile_method'`;
- tests passed: 5;
- tests failed: 1;
- exit code: 1.

The baseline configuration was updated with:

- threshold method: quantile;
- threshold quantile: 0.99;
- quantile method: linear;
- comparison rule: strictly greater than.

The configuration rerun produced:

- tests passed: 6;
- failures: 0;
- exit code: 0.

## Detector module TDD

The detector import test initially failed with
`ModuleNotFoundError: cyber_pca.detector`:

- collection errors: 1;
- exit code: 2.

The detector API skeleton was then created. Its structural tests produced:

- tests passed: 2;
- failures: 0;
- exit code: 0.

### Reconstruction errors

Before implementation:

- tests passed: 2;
- tests failed: 4;
- failure: `NotImplementedError`;
- exit code: 1.

After implementing standardized-space mean squared reconstruction error:

- tests passed: 6;
- failures: 0;
- exit code: 0.

### Threshold calibration

Before implementation:

- tests passed: 6;
- tests failed: 3;
- failure: `NotImplementedError`;
- exit code: 1.

After implementing the normal-calibration-only linear quantile:

- tests passed: 9;
- failures: 0;
- exit code: 0.

### Prediction

Before implementation:

- tests passed: 9;
- tests failed: 3;
- failure: `NotImplementedError`;
- exit code: 1.

After implementing strict `error > threshold` prediction:

- tests passed: 12;
- failures: 0;
- exit code: 0.

## Boundary validation

The first threshold-boundary run produced:

- tests passed: 28;
- tests failed: 1;
- exit code: 1.

The failure occurred because pandas exposed a `ValueError` while converting
nonnumeric strings to float64. The public detector contract required a
`TypeError`.

The conversion was wrapped and translated into a clear public `TypeError`.
The combined detector and threshold-validation rerun produced:

- tests passed: 41;
- failures: 0;
- exit code: 0.

Prediction boundary tests then produced:

- tests passed: 65;
- failures: 0;
- exit code: 0.

## Public package interface

The package-export red test failed during collection because
`AnomalyThresholdResult` was not exported:

- collection errors: 1;
- exit code: 2.

The detector dataclasses and functions were exported through
`cyber_pca.__init__`.

The export inspection confirmed:

- `AnomalyThresholdResult`: available;
- `ReconstructionErrorSplits`: available;
- `calibrate_anomaly_threshold`: available;
- `compute_reconstruction_errors`: available;
- `predict_anomalies`: available;
- missing exports: none;
- inspection exit code: 0.

The first green export run found a missing final newline in
`src/cyber_pca/detector.py`:

- tests passed: 71;
- tests failed: 1;
- exit code: 1.

After that repair, the integrity test found a missing final newline in
`tests/test_detector.py`:

- tests passed: 71;
- tests failed: 1;
- exit code: 1.

A complete text-artifact newline audit was then performed. The rerun produced:

- tests passed: 72;
- failures: 0;
- exit code: 0.

## Synthetic smoke execution

The first smoke command used the incorrect function name
`standardize_data_splits`:

- failure: import error;
- exit code: 1;
- no detector results were produced.

Source inspection identified the validated Phase 3 function as
`standardize_splits`, which was already available at package root.

The corrected label-free smoke execution produced:

- selected components: 5;
- achieved explained variance: 0.95811145295725997;
- calibrated threshold: 0.19016111759041537;
- threshold quantile: 0.99;
- quantile method: linear;
- calibration count: 800;
- calibration errors strictly above threshold: 8;
- calibration errors equal to threshold: 0;
- test predictions equal to zero: 797;
- test predictions equal to one: 1,003;
- exit code: 0.

Reconstruction-error counts were:

- normal fitting: 2,400;
- normal calibration: 800;
- test: 1,800.

Verified reconstruction-error hashes were:

- normal fitting:
  `b4063736f26c3f515892c5008cea6d16fde7bcf9cdf16ff3d902c646bf6d77ec`;
- normal calibration:
  `4e5305df9bb5b2fe90bfb71eee418190ceeb6c0e9e0cb306f0ed528b6fae983a`;
- test:
  `2b820bbef5c44129032ed44ebe54d26874eacc0de72b5ef4fd74b2bd56f86dbb`;
- predictions:
  `e0a69fbfb430fc17336cc7425eb2110bd099f6fcdee8832ca169d801dc40b9e5`.

No anomaly or scenario label was used in threshold calibration or prediction.

## Integration validation

The label-blind integration suite verified:

- partition sizes and identifier alignment;
- finite and nonnegative errors;
- exact NumPy quantile agreement;
- strict prediction agreement;
- deterministic complete reruns;
- near-exact reconstruction with all components.

Result:

- tests passed: 6;
- failures: 0;
- exit code: 0.

## Complete regression and coverage

An initial complete run produced 237 passing tests but did not activate coverage
measurement because the command omitted an explicit `--cov` target. It generated
no Phase 5 coverage XML. That run was accepted as regression evidence only and
not as coverage evidence.

The corrected command explicitly used `--cov=cyber_pca`.

Verified pre-documentation results:

- tests collected: 237;
- tests passed: 237;
- failures: 0;
- errors: 0;
- skipped: 0;
- line coverage: 94.79%;
- branch coverage: 88.40%;
- combined coverage: 93.09%;
- required combined coverage: 90%;
- coverage gate: passed;
- pytest exit code: 0;
- evidence-validation exit code: 0;
- compilation exit code: 0;
- diff-check exit code: 0.

Evidence files:

- `reports/validation/phase_05_pytest.xml`;
- `reports/validation/phase_05_coverage.xml`.

## Post-documentation closure

The required Phase 5 prompt, agent trace, README evidence, and package
documentation test were added.

The focused documentation gate produced:

- tests passed: 8;
- failures: 0;
- exit code: 0.

The final post-documentation regression and coverage run produced:

- tests collected: 238;
- tests passed: 238;
- failures: 0;
- errors: 0;
- skipped: 0;
- line coverage: 94.79%;
- branch coverage: 88.40%;
- combined coverage: 93.09%;
- required combined coverage: 90%;
- coverage gate: passed;
- pytest exit code: 0;
- diff-check exit code: 0.

The final accepted Phase 5 regression count is therefore 238 tests.

## Pre-commit staged-diff repair

The first staged diff check failed with exit code 2 because line 574 of
`tests/test_detector_validation.py` contained trailing whitespace.

The three trailing spaces were removed without changing test logic.

Repair verification produced:

- inspected line 574: `'    )'`;
- line-inspection exit code: 0;
- focused validation tests passed: 53;
- focused validation failures: 0;
- focused validation exit code: 0;
- repaired staged diff-check exit code: 0;
- Project 2 staged files: 0.

No detector mathematics, threshold logic, prediction behaviour, or numerical
evidence changed during this formatting repair.

## Phase outcome

Phase 5 validated reconstruction-error calculation, calibration-only threshold
selection, strict anomaly prediction, deterministic execution, and defensive
input validation.

The 1,003 predicted anomalies are not yet a performance claim. Precision,
recall, F1, confusion-matrix counts, false positives, and false negatives remain
to be calculated in Phase 6.
