# Davis Binary DTI: Final Evidence Synthesis

## Scope

This document consolidates already fixed, versioned evidence. It does not train a new model, tune hyperparameters, choose a probability threshold, or reopen model selection.

Dataset representation: DeepDTA-format Davis benchmark. Upstream DeepDTA commit: `a546a8433a6822e958f36171c4356ad6f414d623`.

## Leakage-aware evaluation design

The primary evaluation is a cold-drug design: 54 training drugs and 14 held-out drugs, with zero drug overlap. Targets overlap by design (442 targets), so the claim is generalisation to unseen drugs rather than unseen drug-target pairs or unseen targets.

Inner validation uses 5 drug-grouped folds on 23868 outer-training pairs. These artefacts explicitly record `outer_test_partition_used=false`.

Random pair splitting reuses related drugs and targets in both partitions and can overestimate unseen-entity generalisation.

**Holdout disclosure:** The final synthesis reads only inner-CV and sensitivity artefacts that record outer_test_partition_used=false. Earlier model-specific outer-holdout results remain part of the development history; therefore, the outer holdout must not be presented as a newly blind confirmatory test for any decision made after it was inspected.

## Primary 1,000 nM task

Positive label: `interaction_kd_le_1000_nM` — Kd less than or equal to 1,000 nM (pKd greater than or equal to 6).

Values are unweighted mean +/- standard deviation across frozen inner cold-drug folds. Average precision is the principal metric.

| Model | AP | ROC-AUC | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dummy_prior | 0.1862 +/- 0.0275 | 0.5000 +/- 0.0000 | 0.8138 +/- 0.0275 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| logistic_regression_balanced | 0.3229 +/- 0.1785 | 0.6333 +/- 0.1096 | 0.6343 +/- 0.1150 | 0.2821 +/- 0.1026 | 0.5516 +/- 0.1616 | 0.3617 +/- 0.1127 |
| random_forest_balanced | 0.3959 +/- 0.0815 | 0.7109 +/- 0.0567 | 0.7937 +/- 0.0371 | 0.4310 +/- 0.0613 | 0.3437 +/- 0.1071 | 0.3780 +/- 0.0767 |
| hist_gradient_boosting_balanced | 0.3897 +/- 0.1255 | 0.6909 +/- 0.0737 | 0.7367 +/- 0.0404 | 0.3506 +/- 0.0679 | 0.5114 +/- 0.1724 | 0.4119 +/- 0.0976 |

Pre-specified selection: `random_forest_balanced` by mean inner-fold AP. The next-ranked fixed candidate was `hist_gradient_boosting_balanced`; the AP margin was 0.0062. This is a descriptive ranking under five frozen inner cold-drug folds, not a formal statistical superiority result.

### Selected-model pooled OOF confusion matrix

The following fixed-threshold (0.5) confusion matrix is pooled across separately fitted inner folds and is descriptive, not a new independent test result.

| True negatives | False positives | False negatives | True positives |
| ---: | ---: | ---: | ---: |
| 17468 | 1983 | 2895 | 1522 |

## Affinity-threshold sensitivity

The 100 nM task reuses the frozen 1,000 nM inner folds and fixed candidate configurations. It is descriptive only and does not replace the primary model-selection decision.

| Variant | Positive rate | Model | AP | ROC-AUC |
| --- | ---: | --- | ---: | ---: |
| 1000 nM (interaction_kd_le_1000_nM) | 0.1851 | dummy_prior | 0.1862 +/- 0.0275 | 0.5000 +/- 0.0000 |
| 1000 nM (interaction_kd_le_1000_nM) | 0.1851 | logistic_regression_balanced | 0.3229 +/- 0.1785 | 0.6333 +/- 0.1096 |
| 1000 nM (interaction_kd_le_1000_nM) | 0.1851 | random_forest_balanced | 0.3959 +/- 0.0815 | 0.7109 +/- 0.0567 |
| 1000 nM (interaction_kd_le_1000_nM) | 0.1851 | hist_gradient_boosting_balanced | 0.3897 +/- 0.1255 | 0.6909 +/- 0.0737 |
| 100 nM (interaction_kd_le_100_nM) | 0.0848 | dummy_prior | 0.0858 +/- 0.0188 | 0.5000 +/- 0.0000 |
| 100 nM (interaction_kd_le_100_nM) | 0.0848 | logistic_regression_balanced | 0.2114 +/- 0.1316 | 0.6829 +/- 0.0788 |
| 100 nM (interaction_kd_le_100_nM) | 0.0848 | random_forest_balanced | 0.2477 +/- 0.0429 | 0.7570 +/- 0.0337 |
| 100 nM (interaction_kd_le_100_nM) | 0.0848 | hist_gradient_boosting_balanced | 0.2915 +/- 0.1232 | 0.7438 +/- 0.0622 |

Raw AP values should not be compared across the two label definitions because the prediction task and class prevalence differ.

## Feature-representation limits

The unsupervised audit found 0 exact drug feature-collision groups and 18 exact target feature-collision groups. The latter are associated with duplicated benchmark sequences; this limits distinguishability for the current representation and does not establish biological equivalence.

## Scientific interpretation boundaries

### Predictive performance

- The reported values estimate ranking and thresholded classification performance for this Davis benchmark under the stated cold-drug inner-CV design.
- Average precision is the principal imbalance-aware comparison metric; ROC-AUC is secondary, while accuracy, precision, recall, F1, and confusion matrices describe the fixed 0.5 operating point.

### Statistical evidence

- Fold means, standard deviations, and pooled OOF summaries are descriptive. Five grouped folds do not by themselves establish statistical superiority or provide a p-value.
- The small primary average-precision difference between the selected random forest and the histogram gradient booster must be interpreted with fold variability in view.

### Biological interpretation

- The benchmark labels are measured affinities, but computational predictions do not validate binding experimentally.
- Target representation collisions and coarse transparent descriptors limit entity-level biological interpretation; identical benchmark sequences do not establish biological equivalence.

### Causal claims

- No association, feature importance, prediction, or threshold-sensitivity result establishes a biological mechanism, therapeutic effect, clinical utility, or causal drug-target relationship.

## Reproducibility record

Evidence source commit: `3523d35b445353e254c90135cf356481f6807914`.

Execution environment: cpython 3.12.8 on Windows-10-10.0.19045-SP0.

The companion JSON records SHA-256 hashes for every input artefact and the complete requirements lock. Re-run the named upstream commands before this synthesis if any source artefact changes.
