# Phase 8 Agent Trace: Official UNSW-NB15 Evaluation

## Starting state

Phase 8 started from commit `5391287`, with the local and remote feature branch
synchronized. Phase 7 had validated the raw curated files and produced
deterministic 64-feature standardized partitions. Raw files remained ignored.

## Compatibility investigation

The existing synthetic `StandardizedDataSplits` and the UNSW standardized
container shared the three model matrices but differed in metadata. The
legacy PCA workflow also enforced the ten-column synthetic feature contract.

A direct adapter was rejected after a red behavior test demonstrated that it
would violate the synthetic schema guard. The agent instead implemented
UNSW-specific mathematical wrappers using the existing validated `ManualPCA`,
component-selection rule, reconstruction-error definition, threshold
calibration, and prediction functions.

## Label-blind detector

The detector fitted PCA only on 42,000 normal fitting observations and selected
34 components. The achieved explained variance was
`0.9521414327676875`. The 0.99 linear calibration quantile produced the frozen
threshold `0.4923769885740442`.

Predictions were generated for 82,332 official test observations before labels
or attack categories entered the evaluation workflow.

## Hidden-label evaluation

The evaluation module constructed testing identifiers as
`unsw_testing:{id}`. It required identical identifier sets for errors,
predictions, and hidden labels before reindexing.

The frozen confusion matrix was:

- true negatives: 35,974;
- false positives: 1,026;
- false negatives: 42,977;
- true positives: 2,355.

Observed precision was `0.6965394853593612`, recall was
`0.05195005735462808`, and F1 was `0.09668876891178946`. The high
false-negative rate of `0.9480499426453719` was retained without tuning.

## Reporting

The reporting module produced deterministic JSON, CSV, and PNG evidence for
binary metrics, predictions, PCA variance, reconstruction-error distributions,
the confusion matrix, and attack-category detection rates.

A complete CLI regeneration reproduced all eight permanent artifacts
byte-for-byte. The temporary verification directory was checked before safe
removal.

## Validation

Focused defensive tests exercised:

- malformed standardized matrices;
- invalid feature names and scaler state;
- overlapping partition identifiers;
- invalid PCA targets and inconsistent refits;
- malformed reconstruction errors;
- invalid official IDs, labels, and categories;
- error, prediction, and hidden-label identifier mismatches;
- invalid attack-category aggregations;
- reporting schema, total, identifier, path, and DPI guards;
- in-process CLI orchestration.

The post-validation suite contained 529 passing tests before adding the two
documentation contract tests. The final Phase 8 suite therefore contains 531
passing tests.

Coverage after implementation and validation:

- line coverage: 96.15%;
- branch coverage: 91.16%;
- combined coverage: 94.74%;
- `unsw_experiment.py`: 100%;
- `unsw_evaluation.py`: 100%;
- `unsw_reporting.py`: 100%.

## Reproducibility hashes

Key artifact SHA-256 values are:

- evaluation JSON:
  `6a18da60eddad611f2b953cab6de6147facced1f6ada392f37ff54604dfd961f`;
- predictions CSV:
  `286cab456827c7429da48c612667c2a2a475e32b07cdc2ef281c620f2ef71e2d`;
- binary metrics CSV:
  `f7af67f6166e371d19b383e470168ceb792836219981c84d9f86e01df707a79c`;
- category metrics CSV:
  `e5d9e06b53cc24b36332f45f24a169bbd723170bd778cc0d507827fd8a55e41b`.

## Machine-readable validation evidence

The final regression and coverage records are stored in:

- `reports/validation/phase_08_pytest.xml`;
- `reports/validation/phase_08_coverage.xml`.

## Closure

Phase 8 records untuned observed baseline performance. Post-evaluation tuning
performed: 0. Raw datasets remain immutable and excluded from version control.
The next phase may evaluate agent reasoning while preserving the frozen Phase
8 evidence.
