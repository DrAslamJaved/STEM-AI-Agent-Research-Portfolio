# Phase 09 interpretation decision

Date: 2026-09-17

## Observed outcome

The approved Phase 09 runner completed six conditions with 100 replications
each and wrote `results/phase_09_robustness.json`.

## Decision

Apply the pre-registered decision rule.  Record `no_supported_advantage` for
theta1 against Jaccard, Dice, and cosine.  In particular, do not promote the
small theta1-cosine difference at epsilon 0.20 to a practical or general
superiority claim because its mean ARI effect (0.001599) is below 0.02.

## Next action

Freeze the synthetic result and prepare the independent Phase 10 external
Davis drug-target-interaction protocol.
