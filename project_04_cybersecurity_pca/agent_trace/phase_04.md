# Phase 04 Agent Trace

## Date

2026-08-28

## Scope

Integrate normal-only PCA fitting, minimum cumulative explained-variance
component selection, and PCA score generation with the leakage-safe synthetic
cybersecurity workflow.

Reconstruction-error thresholding and anomaly classification were not
implemented in this phase.

## Starting state

- Branch: `feature/project-04-cyber-pca`
- Starting commit: `d681fa5`
- Starting state synchronized with the remote feature branch.
- Untracked sibling directory `project_02_drug_target` remained outside the
  Project 4 scope.

## Documentation and configuration

Created `docs/pca_fitting_contract.md`.

Extended `configs/baseline.yaml` with:

- fitting split: `normal_fit_only`;
- component-selection rule:
  `minimum_cumulative_explained_variance`;
- explained-variance target: `0.95`;
- selected-model refitting;
- three score partitions;
- excluded identifier and label columns.

### Configuration red test

Actual result:

- 4 tests passed;
- 1 test failed;
- failure: missing `pca.fitting_split`;
- exit code: 1.

### Configuration green test

Actual result:

- 5 tests passed;
- exit code: 0.

## PCA workflow implementation

Created `src/cyber_pca/pca_workflow.py` with:

- `PCAFitResult`;
- `PCAScoreSplits`;
- `select_n_components`;
- `fit_normal_pca`;
- `transform_pca_splits`.

The workflow fits PCA only on standardized `normal_fit` observations.

### Import red test

Actual result:

- collection error:
  `ModuleNotFoundError: No module named 'cyber_pca.pca_workflow'`;
- exit code: 2.

### First implementation run

Actual result:

- 4 tests passed;
- 4 tests failed;
- each failure reported:
  `ValueError: assignment destination is read-only`.

Root cause:

The cumulative explained-variance array was marked read-only before its final
element was normalized to exactly `1.0`.

Correction:

The array is now finalized before `setflags(write=False)` is applied.

### Repaired workflow run

Actual result:

- 8 tests passed;
- exit code: 0.

## Numerical PCA inspection

Using the deterministic seed-42 synthetic dataset:

- selected components: 5;
- variance target: 0.95;
- achieved variance: 0.95811145295726;
- cumulative variance after four components: 0.92599114;
- cumulative variance after five components: 0.95811145;
- covariance symmetry error: 0.0;
- maximum eigenpair residual:
  `1.915134717478395e-15`;
- orthonormality error:
  `1.3322676295501878e-15`;
- maximum absolute PCA fitting mean:
  `4.100423704282245e-16`;
- fit-score shape: `(2400, 5)`;
- calibration-score shape: `(800, 5)`;
- test-score shape: `(1800, 5)`.

Complete eigenvalues:

1. 6.82544705
2. 1.00359571
3. 0.99724234
4. 0.43748616
5. 0.32133707
6. 0.17586870
7. 0.09764290
8. 0.07771211
9. 0.03904386
10. 0.02879251

## Component-selection boundary tests

### Initial boundary run

Actual result:

- 12 tests passed;
- 14 tests failed;
- exit code: 1.

Root cause:

The new tests passed `explained_variance_target` positionally even though the
public API intentionally defines it as keyword-only.

The four non-numeric-target tests initially passed for the wrong reason:
Python rejected the positional call before target validation was reached.

Correction:

Every test now passes
`explained_variance_target=<value>` explicitly. No production code was
changed.

### Corrected boundary run

Actual result:

- 26 tests passed;
- exit code: 0.

## Package interface

Added public exports for:

- `PCAFitResult`;
- `PCAScoreSplits`;
- `select_n_components`;
- `fit_normal_pca`;
- `transform_pca_splits`.

### Public-interface red test

Actual result:

- collection error:
  `ImportError: cannot import name 'PCAFitResult' from 'cyber_pca'`;
- exit code: 2.

### Public-interface green test

Actual result:

- 32 tests passed;
- all five public exports were present;
- missing exports: none;
- compilation exit code: 0;
- test exit code: 0;
- interface inspection exit code: 0.

## First complete coverage gate

Actual result:

- 149 tests passed;
- failures: 0;
- errors: 0;
- combined coverage: 89.05%;
- required coverage: 90%;
- exit code: 1.

The failure was caused only by the coverage threshold. Functional regression
tests all passed.

## Defensive validation tests

Created `tests/test_pca_workflow_validation.py`.

The tests cover:

- zero total explained variance;
- invalid standardized-split type;
- non-DataFrame partitions;
- empty partitions;
- incorrect feature columns;
- unnamed identifier indexes;
- missing and duplicate flow identifiers;
- nonfinite values;
- insufficient fitting observations;
- all three partition-overlap combinations;
- invalid fit-result type;
- unfitted PCA model;
- incorrect model feature count.

Focused result:

- 16 tests passed;
- compilation exit code: 0;
- test exit code: 0.

## Final regression and coverage gate

Actual result:

- tests collected: 165;
- tests passed: 165;
- failures: 0;
- errors: 0;
- skipped: 0;
- lines covered: 556 of 586;
- line coverage: 94.88%;
- branches covered: 167 of 190;
- branch coverage: 87.89%;
- combined coverage: 93.17%;
- required combined coverage: 90%;
- coverage exit code: 0;
- compilation exit code: 0;
- XML evidence-validation exit code: 0;
- diff-check exit code: 0.

Evidence files:

- `reports/validation/phase_04_pytest.xml`;
- `reports/validation/phase_04_coverage.xml`.

## Remaining uncovered PCA workflow guard

The internal consistency guard at `pca_workflow.py:321` remains uncovered.
It raises a `RuntimeError` only if two deterministic PCA fits on the identical
float64 fitting matrix produce inconsistent eigenvalues.

This is retained as a defensive runtime invariant. It is not bypassed or
excluded from coverage.

## Phase outcome

Phase 4 successfully established:

- normal-only PCA fitting;
- minimum 95% explained-variance component selection;
- deterministic retained-component refitting;
- leakage-safe transformation of every partition;
- preservation of flow identifiers;
- numerical validation at float64 precision;
- defensive input validation;
- public package exports;
- regression evidence above the required coverage threshold.

Commit and push status: TO BE EXECUTED/VERIFIED.
