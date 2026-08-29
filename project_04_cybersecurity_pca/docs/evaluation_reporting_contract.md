# Synthetic Evaluation and Reporting Contract

## 1. Objective

Evaluate the frozen Phase 5 synthetic anomaly predictions using previously
hidden test labels.

Phase 6 calculates performance metrics, scenario-specific results, plots,
machine-readable reports, and an end-to-end command-line workflow.

## 2. Frozen model boundary

Before labels are accessed, the workflow must complete:

1. deterministic synthetic-data generation;
2. leakage-safe splitting;
3. normal-fit-only standardization;
4. normal-fit-only PCA;
5. explained-variance component selection;
6. reconstruction-error calculation;
7. normal-calibration-only threshold selection;
8. test anomaly prediction.

Phase 6 must not refit or modify:

- the feature scaler;
- PCA feature means;
- PCA eigenvalues;
- PCA eigenvectors;
- selected component count;
- reconstruction errors;
- anomaly threshold;
- test predictions.

Labels are evaluation inputs only.

## 3. Evaluation alignment

Align evaluation data by `flow_id`.

The aligned evaluation table must contain:

- `flow_id`;
- `true_anomaly`;
- `predicted_anomaly`;
- `scenario`;
- `reconstruction_error`.

The table must contain exactly 1,800 test observations.

It must reject:

- missing identifiers;
- duplicate identifiers;
- missing labels;
- missing predictions;
- missing scenarios;
- missing reconstruction errors;
- nonbinary labels;
- nonbinary predictions;
- misaligned identifier sets.

The input order must not affect aggregate metrics.

## 4. Positive and negative classes

Use:

- anomaly as the positive class;
- normal as the negative class;
- positive label: 1;
- negative label: 0.

## 5. Confusion matrix

Use matrix layout:

\[
\begin{bmatrix}
TN & FP \\
FN & TP
\end{bmatrix}.
\]

Definitions:

- true negative: normal traffic predicted normal;
- false positive: normal traffic predicted anomalous;
- false negative: attack traffic predicted normal;
- true positive: attack traffic predicted anomalous.

The matrix total must equal 1,800.

## 6. Required metrics

Calculate:

\[
\text{Precision}
=
\frac{TP}{TP+FP}.
\]

\[
\text{Recall}
=
\frac{TP}{TP+FN}.
\]

\[
F_1
=
2
\frac{
\text{Precision}\times\text{Recall}
}{
\text{Precision}+\text{Recall}
}.
\]

\[
\text{False Positive Rate}
=
\frac{FP}{FP+TN}.
\]

\[
\text{False Negative Rate}
=
\frac{FN}{FN+TP}.
\]

Use zero-division value 0 where a denominator is zero.

Also record:

- total observations;
- normal support;
- anomaly support;
- predicted-normal count;
- predicted-anomaly count;
- TN;
- FP;
- FN;
- TP;
- accuracy as a supplementary metric.

Scikit-learn results must agree with independently calculated confusion-matrix
formulas.

## 7. Scenario-specific evaluation

Create one row for each scenario:

- normal;
- brute force;
- denial of service;
- exfiltration;
- port scan.

Each row must contain:

- scenario;
- true label;
- observations;
- predicted normal;
- predicted anomaly;
- predicted-anomaly rate;
- mean reconstruction error;
- median reconstruction error;
- maximum reconstruction error.

For attack scenarios, predicted-anomaly rate is attack-specific recall.

For the normal scenario, predicted-anomaly rate is the false-positive rate.

Scenario observation counts must sum to 1,800.

## 8. Required plots

Generate deterministic PNG figures for:

1. confusion matrix;
2. reconstruction-error distributions for normal and anomalous test classes;
3. PCA scree and cumulative explained-variance plot;
4. scenario-specific predicted-anomaly rates.

Figures must:

- use a noninteractive backend;
- have explicit titles and axis labels;
- use deterministic ordering;
- use configured resolution;
- close figure objects after saving;
- be nonempty valid PNG files.

The reconstruction-error figure should use a logarithmic error axis or another
clearly documented treatment because attack errors span a much larger range
than normal errors.

## 9. Required machine-readable artifacts

Create:

- `results/synthetic_evaluation.json`;
- `results/synthetic_predictions.csv`;
- `reports/tables/synthetic_metrics.csv`;
- `reports/tables/synthetic_scenario_metrics.csv`;
- `reports/figures/synthetic_confusion_matrix.png`;
- `reports/figures/synthetic_reconstruction_errors.png`;
- `reports/figures/synthetic_scree_plot.png`;
- `reports/figures/synthetic_scenario_rates.png`.

The JSON report must include:

- project and phase identifiers;
- seed;
- feature count;
- selected component count;
- achieved explained variance;
- threshold;
- threshold quantile;
- quantile method;
- calibration count;
- confusion-matrix counts;
- all required metrics;
- artifact paths;
- explicit leakage safeguards.

## 10. Command-line contract

Add:

- `python -m cyber_pca evaluate-synthetic --dry-run`;
- `python -m cyber_pca evaluate-synthetic`.

The dry run must:

- print the ordered workflow;
- print intended output paths;
- write no result, table, or figure files.

The real command must:

- execute the deterministic baseline;
- write every required artifact;
- print selected components;
- print threshold;
- print TN, FP, FN, and TP;
- print precision, recall, F1, false-positive rate, and false-negative rate;
- return exit code 0 only when validation succeeds.

## 11. Testing requirements

Tests must verify:

- exact flow-identifier alignment;
- binary label validation;
- confusion-matrix orientation;
- manual metric formulas;
- agreement with scikit-learn;
- false-positive and false-negative counts;
- zero-division behaviour;
- scenario counts and rates;
- row-order invariance;
- deterministic repeated evaluation;
- no input mutation;
- valid JSON and CSV schemas;
- valid nonempty PNG outputs;
- dry-run nonmutation;
- CLI success and failure exit codes;
- unchanged Phase 5 threshold and predictions;
- complete regression coverage of at least 90%.

## 12. Interpretation boundary

High precision means most generated alerts correspond to attacks.

High recall means most attacks are detected.

False positives represent legitimate traffic incorrectly flagged as anomalous.

False negatives represent attacks that the PCA detector misses.

Synthetic performance validates the implementation and controlled hypothesis.
It must not be presented as real-world intrusion-detection performance.
