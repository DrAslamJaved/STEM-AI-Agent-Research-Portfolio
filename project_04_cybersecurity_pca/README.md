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

## Phase 6 verification evidence

Phase 6 evaluated the frozen synthetic PCA detector using hidden test labels
accessed only after anomaly prediction. It also produced deterministic JSON,
CSV, PNG, and command-line evidence.

Verified evidence:

- 311 passing tests;
- failures, errors, and skipped tests: 0;
- line coverage: 93.27%;
- branch coverage: 83.97%;
- 90.71% combined coverage;
- selected principal components: 5;
- achieved explained variance: `0.95811145295725997`;
- frozen threshold: `0.19016111759041537`;
- true negatives: 797;
- false positives: 3;
- false negatives: 0;
- true positives: 1,000;
- confusion matrix: `((797, 3), (0, 1000))`;
- precision: `0.9970089730807578`;
- recall: `1.0`;
- F1: `0.9985022466300548`;
- accuracy: `0.9983333333333333`;
- false-positive rate: `0.00375`;
- false-negative rate: `0.0`;
- all four synthetic attack scenarios detected at 250 of 250;
- reporting module coverage: 100%;
- deterministic module and console CLI execution;
- dry-run created no files;
- eight regenerated artifacts matched permanent evidence by SHA-256.

Evidence files:

- `docs/evaluation_reporting_contract.md`;
- `tests/test_evaluation.py`;
- `tests/test_evaluation_validation.py`;
- `tests/test_evaluation_integration.py`;
- `tests/test_reporting.py`;
- `tests/test_reporting_validation.py`;
- `tests/test_cli_evaluation.py`;
- `tests/test_cli_evaluation_inprocess.py`;
- `results/synthetic_evaluation.json`;
- `results/synthetic_predictions.csv`;
- `reports/tables/synthetic_metrics.csv`;
- `reports/tables/synthetic_scenario_metrics.csv`;
- `reports/figures/synthetic_confusion_matrix.png`;
- `reports/figures/synthetic_reconstruction_errors.png`;
- `reports/figures/synthetic_scree_plot.png`;
- `reports/figures/synthetic_scenario_rates.png`;
- `reports/validation/phase_06_pytest.xml`;
- `reports/validation/phase_06_coverage.xml`;
- `agent_trace/phase_06.md`.

The fitted scaler, PCA model, retained components, calibrated threshold,
reconstruction errors, predictions, and raw test observations remained
unchanged during evaluation.

These synthetic results do not represent real-world cybersecurity performance.

## Phase 7 verification evidence

Phase 7 acquired and validated the official curated UNSW-NB15 files, recorded
immutable provenance, and produced deterministic leakage-safe real-data model
matrices without fitting PCA or evaluating anomaly-detection performance.

Verified evidence:

- 175,341 training observations;
- 82,332 testing observations;
- 45 identical curated columns in both partitions;
- 49 feature-description rows;
- training labels: 56,000 normal and 119,341 attack;
- testing labels: 37,000 normal and 45,332 attack;
- duplicate rows, duplicate partition-local IDs, and missing values: 0;
- 42,000 normal fitting observations;
- 14,000 normal calibration observations;
- training attack observations used for development: 0;
- numeric model inputs: 39;
- normal-fit one-hot columns: 25;
- standardized model features: 64;
- maximum absolute standardized fitting mean:
  `2.6744790509220754e-13`;
- maximum population-standard-deviation error:
  `1.0685896612017132e-12`;
- 400 passing tests;
- failures, errors, and skipped tests: 0;
- line coverage: 94.73%;
- branch coverage: 87.41%;
- 92.71% combined coverage;
- `unsw_data.py` coverage: 100%;
- `unsw_preprocessing.py` coverage: 100%.

Raw-file SHA-256 values:

- training:
  `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`;
- testing:
  `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559`;
- feature descriptions:
  `c55f19cceebb6360dc50f44f8a5f246ccefbcf8a6c604ac1ad46e643869cafce`.

The raw files remain immutable and Git-ignored. The encoder and scaler use only
normal fitting traffic. Training attacks are excluded, unknown categorical
values are ignored safely, and official test labels do not influence fitting.

Evidence files:

- `docs/unsw_nb15_data_contract.md`;
- `prompts/phase_07_unsw_nb15.md`;
- `agent_trace/phase_07.md`;
- `reports/validation/phase_07_unsw_nb15_manifest.json`;
- `reports/validation/phase_07_unsw_nb15_preprocessing.json`;
- `reports/validation/phase_07_pytest.xml`;
- `reports/validation/phase_07_coverage.xml`;
- `tests/test_unsw_data.py`;
- `tests/test_unsw_data_validation.py`;
- `tests/test_unsw_preprocessing.py`;
- `tests/test_unsw_preprocessing_validation.py`;
- `tests/test_unsw_integration.py`.

These results validate acquisition, schema, provenance, and preprocessing for
the official curated dataset. They do not represent UNSW-NB15 anomaly-detection performance.

## Phase 8 verification evidence

Phase 8 evaluated the frozen PCA reconstruction-error detector on the official
UNSW-NB15 testing partition. Predictions were created before hidden labels or
attack categories were accessed, and post-evaluation tuning performed: 0.

Verified evidence:

- normal fitting observations: 42,000;
- normal calibration observations: 14,000;
- official test observations: 82,332;
- standardized model features: 64;
- selected principal components: 34;
- explained-variance target: 0.95;
- achieved explained variance: `0.9521414327676875`;
- frozen threshold: `0.4923769885740442`;
- predicted normal: 78,951;
- predicted anomaly: 3,381;
- true negatives: 35,974;
- false positives: 1,026;
- false negatives: 42,977;
- true positives: 2,355;
- confusion matrix: `((35974, 1026), (42977, 2355))`;
- precision: `0.6965394853593612`;
- recall: `0.05195005735462808`;
- F1: `0.09668876891178946`;
- accuracy: `0.4655419520963902`;
- false-positive rate: `0.02772972972972973`;
- false-negative rate: `0.9480499426453719`;
- 531 passing tests;
- failures, errors, and skipped tests: 0;
- line coverage: 96.15%;
- branch coverage: 91.16%;
- 94.74% combined coverage;
- `unsw_experiment.py` coverage: 100%;
- `unsw_evaluation.py` coverage: 100%;
- `unsw_reporting.py` coverage: 100%.

The complete `evaluate-unsw` CLI regenerated eight artifacts with byte counts
and SHA-256 values identical to the permanent evidence. The temporary
regeneration directory was removed only after all comparisons passed.

Evidence files:

- `docs/unsw_nb15_evaluation_contract.md`;
- `prompts/phase_08_unsw_evaluation.md`;
- `agent_trace/phase_08.md`;
- `results/unsw_nb15_evaluation.json`;
- `results/unsw_nb15_predictions.csv`;
- `reports/tables/unsw_nb15_metrics.csv`;
- `reports/tables/unsw_nb15_attack_category_metrics.csv`;
- `reports/figures/unsw_nb15_confusion_matrix.png`;
- `reports/figures/unsw_nb15_reconstruction_errors.png`;
- `reports/figures/unsw_nb15_scree_plot.png`;
- `reports/figures/unsw_nb15_attack_category_rates.png`;
- `reports/validation/phase_08_pytest.xml`;
- `reports/validation/phase_08_coverage.xml`.

The low attack recall and high false-negative rate are retained as
untuned observed baseline performance. These results are not evidence of an optimized
or operational cybersecurity detector.

## Phase 9 verification evidence

Phase 9 evaluated the agent's critical reasoning about the frozen UNSW-NB15
PCA baseline. The structured response was reviewed by Dr Aslam Javed and
approved as written. Its conclusion remains unchanged: the PCA baseline is
not recommended for operational deployment because of its low recall and high
false-negative rate.

Verified evidence:

- 96 targeted Phase 9 tests passed;
- the reviewed response SHA-256 is
  a0b1a562e0e3bed5d78050e61ddf4743fe5979638406cca84e1990a401f3a48c;
- human_reviewed is recorded as true;
- the review covers factual accuracy, evidence references, and the
  non-deployment conclusion;
- the operational recommendation remains not recommended for operational
  deployment.

Evidence files:

- results/phase_09_agent_reasoning_response.md;
- results/phase_09_agent_reasoning_human_review.json;
- docs/critical_reasoning.md;
- src/cyber_pca/agent_reasoning.py;
- tests/test_agent_reasoning.py;
- tests/test_agent_reasoning_contract.py;
- tests/test_agent_reasoning_response.py;
- tests/test_agent_reasoning_validation.py.

## Phase 10 verification evidence

Phase 10 completed the reproducibility audit and added a dedicated GitHub
Actions workflow for Project 04. The workflow performs a clean Linux checkout,
installs the package, enforces the 90% branch-coverage gate, and uploads JUnit
and coverage evidence.

Verified evidence:

- Windows Python 3.12.8 validation: 627 passed, 95.10% coverage;
- clean GitHub Actions Ubuntu validation on Python 3.12.14:
  626 passed, 1 expected skip, 95.10% coverage;
- the expected skip requires official ignored UNSW-NB15 raw files that are not
  stored in the repository;
- the CI run used commit
  099d4476dc266d11c2812106728223d614d99949;
- CI artifact: project-04-validation-python-3.12;
- artifact SHA-256:
  078a0ed00d9109d6c9681d9c6af8d32d8bc4030359e35988e89ce584af6f346e;
- the Phase 6 evaluation test now checks alignment contracts rather than
  platform-dependent raw floating-point bytes.

Evidence files:

- .github/workflows/project_04_ci.yml;
- docs/phase_10_reproducibility_audit.md;
- tests/test_evaluation_integration.py.

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
| Synthetic evaluation and reporting | Completed |
| UNSW-NB15 acquisition and preprocessing | Completed |
| UNSW-NB15 experiment | Completed |
| Agent reasoning evaluation | Completed; human reviewed |
| Continuous integration | Completed; GitHub Actions validated |
| Reproducibility audit and project closure | Completed on branch; merge pending |
