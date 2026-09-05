# Phase 06 — Cross-validated citation-audit policy

## Decision

Add a transparent post-verification audit before introducing a more powerful
NLI model. The audit filters a frozen candidate trace by assertion confidence,
sentence-evidence confidence, and citation length; it cannot access SciFact
gold fields.

## Critical split finding

The supplied SciFact folds include ordinary-development claim IDs. Treating
their `claims_train_i.jsonl` files as training data would invalidate a final
ordinary-development comparison. Phase 06 instead retains only supplied fold
assignments that overlap `claims_train.jsonl`, then uses the complement within
that file as each fold's training split.

## Pre-specified selection rule

- Grid: 6 assertion thresholds × 5 sentence thresholds × 2 citation lengths.
- Constraint: pooled cross-validation coverage must be at least 0.20.
- Objective: minimize pooled unsupported-assertion rate.
- Tie-breakers: coverage, faithfulness, strict citation F1, stricter thresholds,
  and fewer sentences.
- All fold candidate traces freeze before any fold evaluation labels load.

## Outcome

The selected policy is assertion threshold 0.65, sentence threshold 0.60, and
two sentences per citation. On the untouched 300-claim development split, it
reduced unsupported-assertion rate from 0.9419 to 0.9310, while coverage fell
from 0.8600 to 0.4833. This small risk reduction is retained as evidence of an
abstention trade-off, not as proof of a reliable evidence-verification agent.

## Evidence required before commit

1. Unit tests for policy application, fold filtering, coverage-constrained
   selection, artifact writing, and the two-command CLI workflow.
2. Fold models and full traces stored only in ignored local artifacts.
3. Compact cross-validation and held-out development reports committed to Git.
4. Passing tests, coverage, compilation, whitespace validation, and a fresh
   SciFact run with the selected policy recorded.
