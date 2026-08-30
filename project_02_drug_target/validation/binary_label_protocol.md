# Davis Binary-Label Protocol

## Purpose

The Davis benchmark provides continuous dissociation constants (Kd). Binary
interaction labels are derived secondary outcomes for classification experiments.

## Pre-specified Label Variants

| Analysis | Positive-label rule | pKd equivalent | Positives | Positive rate |
| --- | --- | --- | ---: | ---: |
| Main binary analysis | Kd <= 1,000 nM | pKd >= 6 | 5,561 | 18.50% |
| Stringent sensitivity analysis | Kd <= 100 nM | pKd >= 7 | 2,502 | 8.32% |

All remaining observed pairs are labelled negative within the corresponding
operational definition.

## Safeguards

- Continuous pKd remains available and is not replaced by binary labels.
- Thresholds were chosen before model training and evaluation.
- The two variants will be trained, cross-validated, and reported separately.
- Class imbalance handling will be fitted using training data only.
- Test-set performance will not be used to alter thresholds, features, or model
  settings.
- Predictive performance does not establish biological mechanism or causality.