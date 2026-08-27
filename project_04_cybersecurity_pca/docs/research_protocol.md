
Save the file.

# 5.3 Write `docs/research_protocol.md`

Paste:

```markdown
# Research Protocol

## 1. Research question

Can PCA reconstruction error reliably distinguish anomalous cybersecurity
observations from normal network traffic, and under what data and modelling
assumptions does this reasoning fail?

## 2. Experimental design

The project will implement a one-class, semi-supervised PCA anomaly detector.

The observations will have the following labels:

- `0`: normal;
- `1`: anomaly or attack.

Labels will be retained for evaluation but will not be supplied to PCA during
model fitting.

## 3. Dataset stages

### Stage A: deterministic synthetic data

Synthetic network-flow observations will be used to verify:

- mathematical correctness;
- software correctness;
- deterministic execution;
- interpretable attack patterns;
- threshold behaviour;
- expected reconstruction behaviour.

The synthetic generator will include at least:

1. port scanning;
2. denial of service;
3. brute-force authentication;
4. data exfiltration.

### Stage B: UNSW-NB15

After the complete workflow passes synthetic validation, it will be evaluated
using UNSW-NB15.

The real-data stage must include:

- source documentation;
- acquisition date;
- file hashes;
- schema validation;
- label validation;
- missing-value checks;
- duplicate checks;
- train/test provenance.

## 4. Data partitions

Three logically separate partitions will be used.

### Normal fitting set

Used to:

- fit feature preprocessing;
- estimate training means and standard deviations;
- fit PCA;
- calculate explained variance;
- select the number of components.

### Normal calibration set

Used only to estimate the anomaly threshold from normal reconstruction errors.

The baseline threshold will be:

\[
\tau = Q_{0.99}(e_1,e_2,\ldots,e_m),
\]

where the errors belong to held-out normal calibration observations.

### Labelled test set

Contains normal and attack observations.

It will be used only after preprocessing, PCA, component selection, and
threshold calibration are complete.

## 5. Leakage controls

The following operations must not use test observations:

- imputation fitting;
- categorical encoder fitting;
- feature scaling;
- feature selection;
- PCA fitting;
- component selection;
- anomaly-threshold calibration.

Training means, standard deviations, encoders, and feature order must be reused
without refitting on calibration or test data.

## 6. PCA configuration

The baseline model will:

- use `float64`;
- calculate sample covariance using denominator \(n-1\);
- use a symmetric eigenvalue solver;
- sort eigenvalues in descending order;
- retain the smallest number of components explaining at least 95% of normal
  training variance;
- calculate reconstruction error in standardized feature space.

## 7. Evaluation metrics

With anomaly as the positive class:

\[
\text{Precision} = \frac{TP}{TP+FP},
\]

\[
\text{Recall} = \frac{TP}{TP+FN},
\]

and

\[
F_1 =
2\frac{\text{Precision}\times\text{Recall}}
{\text{Precision}+\text{Recall}}.
\]

The final report will include the complete confusion matrix:

|  | Predicted normal | Predicted anomaly |
|---|---:|---:|
| Actual normal | TN | FP |
| Actual anomaly | FN | TP |

## 8. Required visual evidence

The completed project will produce:

- scree plot;
- cumulative explained-variance plot;
- normal-versus-attack reconstruction-error distributions;
- anomaly-threshold plot;
- confusion-matrix plot;
- component-loading visualization.

## 9. Reproducibility requirements

Every experiment must record:

- Python version;
- dependency versions;
- random seed;
- configuration file;
- input data hashes;
- executed command;
- output paths;
- test results;
- coverage results;
- failures and corrections.

Unexecuted work must be labelled `TO BE EXECUTED/VERIFIED`.
