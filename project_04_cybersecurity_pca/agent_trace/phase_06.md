# Phase 06 Agent Trace

## Date

2026-08-29

## Scope

Implement hidden-label synthetic evaluation, attack-specific analysis,
deterministic reporting, scientific figures, and command-line execution while
preserving the Phase 5 fitted PCA model and calibrated anomaly threshold.

## Starting state

Starting branch:

`feature/project-04-cyber-pca`

Starting commit:

`9ec4b5a Implement calibration-only PCA anomaly detector`

The local branch and remote feature branch had zero divergence. The sibling
`project_02_drug_target/` directory remained untracked and outside the Phase 6
scope.

## Frozen Phase 5 inputs

Phase 6 retained:

- five principal components;
- achieved explained variance `0.95811145295725997`;
- frozen threshold `0.19016111759041537`;
- 800 normal calibration observations;
- 1,800 hidden-label test observations;
- 797 predicted-normal observations;
- 1,003 predicted-anomaly observations.

Labels were accessed only after PCA fitting, threshold calibration,
reconstruction-error calculation, and anomaly prediction.

## Contract and configuration

Created `docs/evaluation_reporting_contract.md`.

Extended `configs/baseline.yaml` with:

- positive label 1;
- negative label 0;
- zero-division value 0;
- confusion-matrix ordering;
- scenario ordering;
- JSON, CSV, and PNG output paths;
- default figure resolution of 150 DPI.

An intentional configuration red test first failed because `positive_label`
was absent. The configuration passed after the evaluation and reporting
contract was added.

## Evaluation module TDD

Created `src/cyber_pca/evaluation.py`.

Implemented:

- `BinaryEvaluationResult`;
- `align_evaluation_data`;
- `evaluate_binary_predictions`;
- `evaluate_scenarios`.

Red tests first demonstrated missing imports and unimplemented functions.
Green implementations then validated flow-ID alignment, binary formulas,
scikit-learn agreement, zero-division behavior, deterministic scenario
statistics, order invariance, and input immutability.

Focused evaluation validation completed with 39 passing tests. Four additional
integration tests validated the complete frozen synthetic evaluation workflow.

## Verified binary results

The hidden synthetic test labels produced:

- total observations: 1,800;
- normal support: 800;
- anomaly support: 1,000;
- predicted normal: 797;
- predicted anomalous: 1,003;
- true negatives: 797;
- false positives: 3;
- false negatives: 0;
- true positives: 1,000;
- precision: `0.9970089730807578`;
- recall: `1.0`;
- F1: `0.9985022466300548`;
- accuracy: `0.9983333333333333`;
- false-positive rate: `0.00375`;
- false-negative rate: `0.0`;
- confusion matrix: `((797, 3), (0, 1000))`.

## Verified scenario results

All four synthetic attack scenarios produced 250 predictions out of 250 as
anomalous.

Normal traffic produced 3 false-positive predictions among 800 observations.

Mean reconstruction errors were:

- normal: `0.039238`;
- brute force: `10.354994`;
- denial of service: `122.207760`;
- exfiltration: `18.288151`;
- port scan: `80.268211`.

These values describe the controlled synthetic fixture only.

## Reporting module TDD

Created `src/cyber_pca/reporting.py`.

Implemented:

- `SyntheticEvaluationArtifacts`;
- `resolve_synthetic_evaluation_artifacts`;
- `build_synthetic_evaluation_summary`;
- `write_synthetic_evaluation_artifacts`.

The reporting workflow writes deterministic JSON, CSV, and PNG artifacts.
Four core reporting tests validated path resolution, serializable summaries,
complete artifact writing, and input immutability.

The first complete coverage run exposed untested reporting validation guards.
Eighteen focused boundary tests were then added. Reporting coverage increased
to 100%.

## CLI TDD

Extended `src/cyber_pca/cli.py` with the `evaluate-synthetic` subcommand.

Three black-box subprocess tests first failed because the command did not
exist. The green implementation added module and console execution,
`--dry-run`, `--output-root`, and `--dpi`.

Two in-process CLI tests were added because subprocess execution did not
contribute to the parent coverage process.

The manual CLI smoke test verified:

- module help exit code 0;
- console help exit code 0;
- dry-run exit code 0;
- dry-run created no output;
- real execution exit code 0;
- exactly eight temporary artifacts;
- temporary artifacts matched permanent evidence byte for byte;
- safe removal of the temporary execution directory;
- clean repository diff.

## Type-interface repair

`StandardizedDataSplits` was imported for the
`compute_reconstruction_errors` annotation.

Runtime type-hint inspection resolved the type to
`cyber_pca.preprocessing.StandardizedDataSplits`. This repair changed no
detector mathematics or runtime predictions.

## Permanent evidence artifacts

Created:

- `results/synthetic_evaluation.json`;
- `results/synthetic_predictions.csv`;
- `reports/tables/synthetic_metrics.csv`;
- `reports/tables/synthetic_scenario_metrics.csv`;
- `reports/figures/synthetic_confusion_matrix.png`;
- `reports/figures/synthetic_reconstruction_errors.png`;
- `reports/figures/synthetic_scree_plot.png`;
- `reports/figures/synthetic_scenario_rates.png`;
- `reports/validation/phase_06_pytest.xml`;
- `reports/validation/phase_06_coverage.xml`.

The permanent artifacts were independently regenerated in a temporary output
root and matched the committed-working-copy evidence by SHA-256.

## Complete regression and coverage

Final pre-documentation validation produced:

- 310 passing tests;
- failures: 0;
- errors: 0;
- skipped: 0;
- lines covered: 1,025 of 1,099;
- line coverage: 93.27%;
- branches covered: 351 of 418;
- branch coverage: 83.97%;
- combined coverage: 90.71%;
- reporting module coverage: 100%;
- compilation exit code: 0;
- dependency-check exit code: 0;
- diff-check exit code: 0.

## State-integrity validation

The complete Phase 6 evaluation confirmed:

- PCA components unchanged;
- PCA mean unchanged;
- threshold unchanged;
- test reconstruction errors unchanged;
- predictions unchanged;
- raw test observations unchanged;
- labels accessed after prediction.

## Phase outcome

Phase 6 implemented a deterministic, leakage-safe synthetic evaluation and
reporting workflow with binary metrics, scenario results, plots, reproducible
artifacts, and CLI execution.

These synthetic results do not represent real-world cybersecurity
performance. Phase 7 will introduce UNSW-NB15 acquisition, provenance, schema
validation, and leakage-safe preparation.

## Post-documentation closure

After adding the Phase 6 prompt, agent trace, README evidence, artifact
requirements, and implementation-status update, the final verification
produced:

- 311 passing tests;
- failures: 0;
- errors: 0;
- skipped: 0;
- line coverage: 93.27%;
- branch coverage: 83.97%;
- combined coverage: 90.71%;
- compilation exit code: 0;
- dependency-check exit code: 0;
- evidence-validation exit code: 0;
- diff-check exit code: 0.

The final README marks synthetic evaluation and reporting as completed and
identifies UNSW-NB15 acquisition and validation as the next phase.
