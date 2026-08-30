# Phase 8 Prompt: Official UNSW-NB15 Evaluation

Implement and verify the official UNSW-NB15 evaluation without changing the
Phase 7 preprocessing contract or using test outcomes for model selection.

## Required workflow

1. Load the ignored official curated files through the validated Phase 7
   loader.
2. Recreate the deterministic normal fitting, normal calibration, and official
   test partitions.
3. Recreate the normal-fit-only encoder and standardizer.
4. Fit PCA using only the standardized normal fitting matrix.
5. Select the minimum component count reaching the configured 0.95
   explained-variance target.
6. Compute standardized-space reconstruction errors.
7. Calibrate the 0.99 linear quantile threshold using only normal calibration
   errors.
8. Freeze official test predictions before accessing labels or attack
   categories.
9. Align labels, predictions, and errors by `(source_partition, id)`.
10. Calculate binary and official attack-category metrics exactly once.
11. Write deterministic JSON, CSV, and PNG evidence.
12. Regenerate the artifacts through the CLI and compare all eight SHA-256
    values.

## Prohibited actions

- Do not fit on training attacks.
- Do not fit any transformation on calibration or test observations.
- Do not use official test labels for PCA component selection.
- Do not use official test labels or categories for threshold selection.
- Do not tune after observing precision, recall, F1, or category results.
- Do not commit raw UNSW-NB15 files.
- Do not modify or stage files belonging to another project.

## Testing requirements

Use red-green development for:

- the UNSW PCA wrapper;
- label-blind orchestration;
- identifier-safe hidden-label alignment;
- official category aggregation;
- deterministic reporting;
- package exports;
- configuration;
- CLI dry-run and execution;
- defensive validation branches.

The final suite must achieve at least 90% combined coverage. Phase 8 UNSW
experiment, evaluation, and reporting modules must each reach complete line
and branch coverage.

## Reporting requirement

Report weak results faithfully. Describe the output as an untuned observed
baseline. Explicitly state that no post-evaluation tuning was performed and
that the result is not an optimized operational detector.
