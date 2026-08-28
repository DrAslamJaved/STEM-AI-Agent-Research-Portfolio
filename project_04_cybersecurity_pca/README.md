# Project 04: Agentic Cybersecurity Anomaly Detection Using PCA

## Objective

Develop and validate an agentic cybersecurity anomaly-detection workflow that
integrates:

- linear algebra;
- principal component analysis;
- machine learning;
- cybersecurity;
- software testing;
- AI agent evaluation.

## Research question

Can PCA reconstruction error reliably distinguish anomalous cybersecurity
observations from normal network traffic, and under what data and modelling
assumptions does this reasoning fail?

## Core hypothesis

If PCA is fitted using representative normal network traffic, the retained
principal components should represent the dominant linear correlation structure
of normal behaviour.

Observations that lie far from this learned normal subspace should have large
reconstruction errors and can therefore be treated as anomaly candidates.

A large reconstruction error is statistical evidence of unusual behaviour. It
is not proof that an observation is malicious.

## Workflow

```text
Raw cybersecurity data
↓
Data validation
↓
Leakage-safe preprocessing
↓
Feature standardization
↓
Covariance matrix
↓
Eigenvalue decomposition
↓
Principal components
↓
Dimensionality reduction
↓
Reconstruction
↓
Reconstruction error
↓
Normal-only threshold calibration
↓
Anomaly prediction
↓
Evaluation
↓
AI agent reasoning assessment
```

## Phase 1 verification evidence

The verified Phase 1 execution produced:

- 60 passing tests;
- 93.26% total coverage;
- seven passing independent mathematical checks;
- deterministic JSON validation output;
- successful package compilation;
- successful module and console entry points.

Evidence files:

```text
reports/validation/math_validation.json
reports/validation/phase_01_final_pytest.xml
reports/validation/phase_01_final_coverage.xml
agent_trace/phase_01.md
```

These results validate the mathematical and software foundation. They do not
represent cybersecurity anomaly-detection performance.

## Phase 2 verification evidence

Phase 2 implemented a deterministic synthetic network-flow dataset containing
normal traffic and four interpretable attack scenarios.

Verified evidence:

- default dataset size: 5,000 observations;
- model features: 10;
- attack scenarios: port scan, denial of service, brute force, and exfiltration;
- focused synthetic-data tests: 24 passed;
- complete regression suite: 84 passed;
- total coverage: 94.82%;
- duplicate flow identifiers: 0;
- missing values: 0;
- all model features use float64;
- same-seed dataset hashes matched;
- different-seed dataset hashes differed.

The verified seed-42 dataset SHA-256 is
`35005389b137bd472e44b44c987597b1b7e13b8fa88a4c099c110c50986e1561`.

Evidence files:

- `docs/synthetic_data_contract.md`;
- `tests/test_synthetic_data.py`;
- `reports/validation/phase_02_pytest.xml`;
- `reports/validation/phase_02_coverage.xml`;
- `agent_trace/phase_02.md`.

These results validate deterministic synthetic-data generation. They do not
represent real-world cybersecurity anomaly-detection performance.

## Phase 3 verification evidence

Phase 3 implemented deterministic, leakage-safe normal fitting, calibration,
and evaluation partitions together with normal-fitting-only feature
standardization.

Verified evidence:

- normal fitting observations: 2,400;
- normal calibration observations: 800;
- normal test observations: 800;
- attack test observations: 1,000;
- complete test observations: 1,800;
- focused preprocessing tests: 38 passed;
- complete regression suite: 122 passed;
- total coverage: 91.46%;
- overlapping split identifiers: 0;
- identifier union complete: true;
- maximum absolute standardized fitting mean:
  `4.100423704282245e-16`;
- maximum population-standard-deviation error:
  `2.220446049250313e-16`;
- same-seed fitting partitions matched;
- different-seed fitting partitions differed.

The seed-42 fitting-identifier SHA-256 is
`fd9763c2e230cdc89f1319b79e3d4a113f6fce28a47f7af03aedf9c85863e6c9`.

The fitted-scaler SHA-256 is
`0db3e4facfc81ced3acc5fdec3e1d859118837c8e24aa0566139fbbd0996f0be`.

The scaler uses only normal fitting features. Calibration observations, test
observations, scenario labels, anomaly labels, and flow identifiers do not
influence its fitted means or scales.

Evidence files:

- `docs/preprocessing_contract.md`;
- `tests/test_preprocessing.py`;
- `reports/validation/phase_03_pytest.xml`;
- `reports/validation/phase_03_coverage.xml`;
- `agent_trace/phase_03.md`.

## Phase 4 verification evidence

Phase 4 integrated the manual PCA implementation with the leakage-safe
standardized partitions. PCA is fitted only on normal fitting observations,
and the minimum component count satisfying the configured explained-variance
target is retained.

Verified evidence:

- PCA fitting observations: 2,400 normal observations;
- explained-variance target: 0.95;
- selected principal components: 5;
- achieved explained variance: `0.95811145295726`;
- cumulative variance with four components: `0.92599114`;
- cumulative variance with five components: `0.95811145`;
- fit-score shape: `(2400, 5)`;
- calibration-score shape: `(800, 5)`;
- test-score shape: `(1800, 5)`;
- covariance symmetry error: `0.0`;
- maximum eigenpair residual:
  `1.915134717478395e-15`;
- eigenvector orthonormality error:
  `1.3322676295501878e-15`;
- maximum absolute PCA fitting mean:
  `4.100423704282245e-16`;
- focused PCA-workflow tests: 42 passed;
- complete regression suite: 165 passed;
- failures, errors, and skipped tests: 0;
- line coverage: 94.88%;
- branch coverage: 87.89%;
- combined coverage: 93.17%.

Changing calibration or test values does not change the fitted PCA model.
Calibration observations, test observations, attack labels, scenario labels,
and flow identifiers do not influence PCA fitting or component selection.

Evidence files:

- `docs/pca_fitting_contract.md`;
- `tests/test_pca_workflow.py`;
- `tests/test_pca_workflow_validation.py`;
- `reports/validation/phase_04_pytest.xml`;
- `reports/validation/phase_04_coverage.xml`;
- `agent_trace/phase_04.md`.

These results validate normal-only PCA fitting and dimensionality reduction on
the deterministic synthetic dataset. Reconstruction-error thresholding,
anomaly classification, and real-world UNSW-NB15 evaluation remain to be
implemented.

## Phase 5 verification evidence

Phase 5 implemented standardized-space PCA reconstruction errors,
normal-calibration-only threshold selection, and strict binary anomaly
prediction.

Verified evidence:

- selected PCA components: 5;
- achieved explained variance: `0.95811145295725997`;
- normal fitting reconstruction errors: 2,400;
- normal calibration reconstruction errors: 800;
- test reconstruction errors: 1,800;
- calibrated threshold: `0.19016111759041537`;
- threshold quantile: 0.99;
- quantile method: linear;
- calibration errors strictly above threshold: 8;
- calibration errors equal to threshold: 0;
- test observations predicted normal: 797;
- test observations predicted anomalous: 1,003;
- complete regression suite: 238 passed;
- line coverage: 94.79%;
- branch coverage: 88.40%;
- combined coverage: 93.09%;
- failures, errors, and skipped tests: 0;
- full-component reconstruction: approximately exact;
- complete detector execution: deterministic.

Evidence files:

- `docs/reconstruction_error_contract.md`;
- `tests/test_detector.py`;
- `tests/test_detector_validation.py`;
- `tests/test_detector_integration.py`;
- `reports/validation/phase_05_pytest.xml`;
- `reports/validation/phase_05_coverage.xml`;
- `agent_trace/phase_05.md`.

The PCA model uses only normal fitting observations. The anomaly threshold uses
only normal calibration reconstruction errors. Test errors, anomaly labels, and
scenario labels do not influence either fitted object.

These results validate reconstruction-error computation and calibration-only
thresholding. They do not represent final anomaly-detection performance.

## Current implementation status

| Component | Status |
|---|---|
| Git feature branch | Completed |
| Isolated Python 3.12 environment | Completed |
| Project architecture | Completed |
| Python package configuration | Completed |
| Mathematical specification | Completed |
| Manual PCA implementation | Completed |
| Mathematical tests | Completed |
| Mathematical validation CLI | Completed |
| Synthetic cybersecurity data | Completed |
| Leakage-safe splitting and standardization | Completed |
| Normal-only PCA fitting and component selection | Completed |
| Reconstruction-error detector | Completed |
| Synthetic evaluation and reporting | Next phase |
| UNSW-NB15 experiment | To be implemented |
| Agent reasoning evaluation | To be implemented |
| Continuous integration | To be implemented |
