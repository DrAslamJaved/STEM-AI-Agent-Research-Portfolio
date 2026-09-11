# Phase 3 Analysis Protocol

This phase supplies reproducible reference baselines, not a production DTI
predictor. A seeded random split is retained only as a diagnostic: shared drugs
and targets are reported because overlap can inflate apparent performance.
Cold-drug evaluation is a stricter alternative.

The prevalence baseline predicts the positive-label prevalence measured on the
training partition and reports accuracy, precision, recall, F1, ROC-AUC, and
PR-AUC. Reference fuzzy similarities include Jaccard, Dice, cosine, and a
cardinality-balance comparison. The latter compares sigma-count cardinalities;
equal cardinality does not imply identical fuzzy sets.
