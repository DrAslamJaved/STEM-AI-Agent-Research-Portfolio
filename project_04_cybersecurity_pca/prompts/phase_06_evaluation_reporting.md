# Phase 06 Evaluation and Reporting Prompt

## Project

Agentic Cybersecurity Anomaly Detection Using PCA and Eigenvalue Analysis.

## Scope

Evaluate the frozen synthetic PCA anomaly detector using previously hidden test
labels. Produce reproducible binary metrics, attack-specific results, tables,
figures, JSON evidence, prediction records, and command-line execution.

Evaluation must not modify the fitted scaler, fitted PCA model, retained
components, calibrated threshold, reconstruction errors, predictions, or raw
test observations.

## Existing validated inputs

Use the completed Phase 1 through Phase 5 workflow:

- deterministic synthetic network-flow generation;
- leakage-safe normal fitting, calibration, and test partitions;
- standardization fitted only on normal fitting traffic;
- PCA fitted only on standardized normal fitting traffic;
- minimum component selection using the 0.95 variance target;
- standardized-space mean-squared reconstruction errors;
- the 0.99 normal-calibration quantile threshold;
- strict greater-than anomaly prediction;
- hidden synthetic test labels accessed only after prediction.

## Evaluation contract

Align test labels, predictions, scenarios, and reconstruction errors by
`flow_id`, not by incidental row position.

The aligned evaluation table must contain:

- `true_anomaly`;
- `predicted_anomaly`;
- `scenario`;
- `reconstruction_error`.

Use anomaly label 1 as the positive class and normal label 0 as the negative
class.

Calculate:

- true negatives;
- false positives;
- false negatives;
- true positives;
- precision;
- recall;
- F1;
- accuracy;
- false-positive rate;
- false-negative rate;
- confusion matrix.

Use confusion-matrix layout `((TN, FP), (FN, TP))` and zero-division value 0.

## Scenario contract

Report one row for each scenario in this order:

1. normal;
2. brute force;
3. denial of service;
4. exfiltration;
5. port scan.

For every scenario calculate:

- true label;
- observation count;
- predicted-normal count;
- predicted-anomaly count;
- predicted-anomaly rate;
- mean reconstruction error;
- median reconstruction error;
- maximum reconstruction error.

## Reporting contract

Write deterministic artifacts to the configured output root:

- `results/synthetic_evaluation.json`;
- `results/synthetic_predictions.csv`;
- `reports/tables/synthetic_metrics.csv`;
- `reports/tables/synthetic_scenario_metrics.csv`;
- `reports/figures/synthetic_confusion_matrix.png`;
- `reports/figures/synthetic_reconstruction_errors.png`;
- `reports/figures/synthetic_scree_plot.png`;
- `reports/figures/synthetic_scenario_rates.png`.

Figures must use a headless Matplotlib backend and 150 DPI by default.

The reporting workflow must validate types, table schemas, index names,
observation totals, output roots, and DPI values before writing artifacts.

## CLI contract

Extend the existing CLI with `evaluate-synthetic`.

The command must support:

- `--dry-run`;
- `--output-root`;
- `--dpi`.

Dry-run mode must print the ordered workflow and planned artifact paths without
creating directories or files.

Normal execution must run the deterministic synthetic workflow, write all eight
artifacts, and print the selected component count, achieved explained
variance, frozen threshold, confusion matrix, precision, recall, F1,
false-positive rate, false-negative rate, and evaluation-report path.

## Required public API

Expose:

- `BinaryEvaluationResult`;
- `align_evaluation_data`;
- `evaluate_binary_predictions`;
- `evaluate_scenarios`;
- `SyntheticEvaluationArtifacts`;
- `build_synthetic_evaluation_summary`;
- `resolve_synthetic_evaluation_artifacts`;
- `write_synthetic_evaluation_artifacts`.

## Required validation

Use intentional red-green testing for:

- module imports;
- public dataclass contracts;
- flow-ID alignment;
- binary metric formulas;
- agreement with scikit-learn;
- zero-division behavior;
- scenario statistics;
- input immutability;
- invalid types;
- empty tables;
- incorrect columns;
- incorrect index names;
- invalid output roots;
- invalid DPI values;
- artifact creation;
- module CLI execution;
- console-script CLI execution;
- dry-run non-writing behavior;
- deterministic artifact hashes.

## Evidence requirements

The final evidence must include:

- complete pytest XML;
- line and branch coverage XML;
- successful package compilation;
- successful dependency validation;
- a clean `git diff --check`;
- deterministic JSON and CSV evidence;
- verified PNG figures;
- a human-readable Phase 6 agent trace;
- a README evidence section.

Synthetic results must be clearly distinguished from real-world
cybersecurity performance.
