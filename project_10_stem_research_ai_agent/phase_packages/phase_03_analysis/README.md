# Phase 03 — Reproducible Analysis Baselines

## Goal
Implement transparent baselines rather than complex models first.

## Deliverables
- DTI baseline: prevalence classifier and logistic regression with group-aware split option.
- Metrics: precision, recall, F1, ROC-AUC, PR-AUC; saved confusion matrix.
- Fuzzy baseline: Jaccard, Dice, cosine, and approved cardinality-based similarity.
- Seeded experiment runner and JSON result contract.

## Acceptance criteria
1. Identical seed and inputs reproduce metrics.
2. Random-split and cold-drug/cold-target leakage risks are explicitly compared.
3. Similarity outputs are bounded and symmetric under tested conditions.

## Git checkpoint
`feat(project07): implement reproducible DTI and fuzzy baselines`
