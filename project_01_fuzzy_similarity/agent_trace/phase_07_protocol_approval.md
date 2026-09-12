# Phase 07 — Experiment Protocol Approval

## Human decision

Dr. Aslam Javed approved the Phase 07 numerical-experiment protocol.

## Approval scope

The approved protocol authorizes implementation and testing of the stated
comparison baselines and reproducible experiment runner. It does not authorize
unlabelled post-hoc changes to scenarios, parameters, seeds, or interpretation
rules.

## Evidence

- `docs/phase_07_experiment_protocol.md`
- user decision: `approved`

## Claim boundary

Future numerical outputs may be labelled `EXPERIMENTALLY_SUPPORTED` only.
They cannot establish a theorem, novelty, or universal superiority claim.

## Next gate

Implement fuzzy Dice and cosine baselines, test their stated definitions and
boundary conventions, then review the resulting code before executing Phase
07 scenarios.
