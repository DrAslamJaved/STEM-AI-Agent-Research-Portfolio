# v0.2 Leakage-Aware DTI Split Protocol

## Motivation

Pair-random DTI evaluation permits the same drug and target identities to occur
in training and test data. It is a useful reference but does not measure the
ability to generalize to new drugs or new targets. v0.2 therefore requires all
three deterministic split conditions below.

| Split | Train/test identity overlap permitted | Primary interpretation |
|---|---|---|
| Pair-random | Drugs and targets may overlap; pairs may not | Reference condition only |
| Cold-drug | Targets may overlap; drugs may not | New-drug generalization |
| Cold-target | Drugs may overlap; targets may not | New-target generalization |

## Required diagnostics

For every split, record record counts, class prevalence, unique drug/target
counts, and shared drug, target, and pair counts. A shared pair is always an
error. A shared drug in a cold-drug split or a shared target in a cold-target
split is an error.

## Determinism

The initial v0.2 experiments use a 20% held-out fraction and seed `20260915`.
Any later change to fraction, seed, or threshold creates a distinct evaluation
condition and must be reported as such.
