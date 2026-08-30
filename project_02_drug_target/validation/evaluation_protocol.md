# Davis Binary-Classification Evaluation Protocol

## Purpose

This protocol defines the metrics used to evaluate binary DTI classifiers.
It separates threshold-free ranking performance from threshold-dependent
classification performance.

## Positive Class

A positive label is `1`.

For the primary binary experiment, this means:

`interaction_kd_le_1000_nM = 1`

or equivalently:

`Kd <= 1,000 nM`

## Primary Metric

Average precision is the primary ranking metric.

It is calculated with scikit-learn's `average_precision_score` and used as the
project's PR-AUC measure. It summarizes precision-recall performance from
positive-class probabilities without selecting a single decision threshold.

This choice is appropriate because only about 18.5% of the primary-label pairs
are positive. Accuracy alone can appear high when a model mostly predicts the
majority negative class.

## Secondary Metrics

The evaluator reports:

- ROC-AUC;
- accuracy;
- precision;
- recall;
- F1 score;
- confusion-matrix counts.

ROC-AUC is useful as a threshold-free ranking summary but can be less
informative than precision-recall performance when positives are uncommon.

## Fixed Decision Threshold

The decision threshold is fixed at `0.50`.

A pair is classified as positive when:

`predicted_positive_probability >= 0.50`

Accuracy, precision, recall, F1, and the confusion matrix are calculated at
this threshold.

The outer test set must not be used to choose a more favorable threshold.
Any future threshold selection must occur within training data or an
inner validation procedure and be reported separately.

## Confusion-Matrix Convention

The confusion matrix uses labels `[0, 1]` and is interpreted as:

| Row / column | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | true negative | false positive |
| Actual 1 | false negative | true positive |

## Undefined Precision

If a model predicts no positive pairs, precision and F1 are reported as `0.0`
using `zero_division=0`. This is explicit and prevents an undefined metric
from being hidden.

## Reporting by Split Policy

Every model must be evaluated separately for:

- `random_pair`;
- `cold_drug`;
- `cold_target`.

The `cold_drug` result is the headline estimate for unseen-drug
generalization. The random-pair result is an interpolation benchmark and must
not be presented as evidence of novelty generalization.

## Interpretation Boundary

Metric values describe benchmark prediction performance only.

They do not establish statistical significance, biological mechanism,
pharmacological causality, clinical effectiveness, or drug safety.