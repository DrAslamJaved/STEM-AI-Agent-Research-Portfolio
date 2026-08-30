# Phase 9 Agent Reasoning Response: UNSW-NB15 PCA Baseline

Status: AI-generated reasoning response awaiting structured human review.

## Decision

The frozen UNSW-NB15 PCA reconstruction-error baseline is not recommended for operational deployment. Its observed precision is high enough to show
that some flagged flows are anomalous, but its recall is too low and its
false-negative rate too high for an operational cybersecurity decision.
This is an untuned observed baseline, with post-evaluation tuning performed: 0.

## 1. Geometric subspace argument

PCA learns a linear subspace that captures the dominant covariance structure
of representative normal traffic. A flow that has a large component
perpendicular to that learned subspace cannot be reconstructed accurately and
is therefore an anomaly candidate. In the frozen experiment, PCA used 42,000 normal fitting observations, retained 34 principal components, and achieved
explained variance of `0.9521414327676875`.

## 2. Anomaly evidence is not proof

A large reconstruction error is statistical evidence of unusual behaviour; it
is not proof that the flow is malicious. Legitimate maintenance, backups,
software updates, research activity, or rare operational conditions can also
differ from common traffic. Conversely, an attack that lies mainly within the
retained high-variance directions can have a small reconstruction error.

## 3. Normal-training assumption

The reasoning assumes that the 42,000 normal fitting observations are
representative of the normal traffic structure that the detector should learn.
If those observations are contaminated by attacks, unrepresentative, or
affected by later concept drift, the learned subspace can make both
false-positive and false-negative decisions. The separate 14,000 normal calibration observations were reserved for threshold calibration rather than
PCA fitting.

## 4. Leakage and contamination controls

No test observation was used to fit imputation, categorical encoding, feature
scaling, PCA, component selection, or threshold calibration. The encoder and
scaler were fitted only on normal fitting traffic; PCA was fitted only on the
normal fitting partition; and the threshold was calibrated only from normal
calibration reconstruction errors. Official test predictions for 82,332 official test observations were frozen before hidden labels and attack
categories entered evaluation.

## 5. Scaling and threshold selection

The model used 64 standardized model features. Standardization matters because
unscaled high-magnitude variables can dominate covariance and reconstruction
error without representing the most meaningful cybersecurity variation. The
frozen anomaly threshold was the linear 0.99 quantile of normal calibration
errors: `0.4923769885740442`. An unsuitable threshold can create excessive
false positives when too low or excessive false negatives when too high.

## 6. False-positive and false-negative consequences

The frozen evaluation produced 1,026 false positives and 42,977 false negatives. Its recall was `0.05195005735462808`, while its false-negative rate
was `0.9480499426453719`. The false positives create analyst workload and may
reduce trust in alerts. More seriously, the false negatives mean that many
labelled attacks were predicted as normal, so the detector must not be treated
as a sufficient operational control.

## 7. Empirical validation and next action

Empirical validation, rather than intuition alone, is required. The frozen
Phase 8 evaluation provides that evidence for this particular PCA baseline:
it reports the confusion matrix, binary metrics, attack-category results,
artifact hashes, and byte-for-byte CLI regeneration. The observed low recall
and high false-negative rate must remain visible. Future work may compare
alternative models or feature representations under a newly specified,
leakage-safe protocol, but it must not retune or reinterpret the frozen Phase
8 evidence.

## Evidence sources

- `results/unsw_nb15_evaluation.json`;
- `reports/tables/unsw_nb15_metrics.csv`;
- `reports/tables/unsw_nb15_attack_category_metrics.csv`;
- `agent_trace/phase_08.md`;
- `docs/critical_reasoning.md`.
