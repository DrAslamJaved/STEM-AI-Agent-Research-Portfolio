# Phase 07 — Numerical-Experiment Protocol

## Status

`HUMAN_APPROVED` to prepare this protocol. No numerical experiment has been executed and no empirical conclusion is reported here.

## Aim

Characterize the approved rational similarity subfamily on controlled fuzzy vectors relative to standard comparison measures. This tests implementation behaviour and patterns; it does not prove Phase 05 theorems.

## Measures

1. Approved rational subfamily, including its min-max fuzzy Jaccard control setting.
2. Fuzzy Jaccard: min-max intersection over min-max union, with explicit zero-union handling.
3. Fuzzy Dice: twice min-max intersection over the sum of sigma-counts, with explicit zero-denominator handling.
4. Fuzzy cosine: dot product divided by Euclidean norms, with explicit zero-norm handling.

Dice and cosine are comparison baselines only; their definitions and boundary rules must be preserved in the implementation.

## Pre-specified scenarios

| ID | Construction | Purpose |
|---|---|---|
| E01 | identical vectors | reflexivity sanity check |
| E02 | graded perturbations | sensitivity to membership change |
| E03 | disjoint/near-disjoint supports | low-overlap behaviour |
| E04 | nested vectors | inclusion-related behaviour only |
| E05 | complements | self-complementarity is not assumed |
| E06 | zero/near-zero vectors | boundary and stability behaviour |
| E07 | random vectors in [0,1]^n | numerical sanity distribution |

## Design

- dimensions: \(n\in\{1,2,5,10,50\}\);
- random seed: `20260905`;
- E07: 1,000 ordered pairs per dimension;
- IEEE 754 double precision;
- record settings separately, with configuration, source hash, command, Python version, and outputs.

## Outcomes and interpretation

Record each similarity value and, for repeated samples, mean, median, standard deviation, minimum, maximum, and quantiles. Report paired baseline differences without calling one measure superior unless a criterion and application evidence are pre-specified.

- A range, symmetry, or reflexivity failure is an implementation defect requiring review.
- Numerical agreement with Phase 05 is `EXPERIMENTALLY_SUPPORTED` only.
- A conflict with Phase 05 requires an implementation/formulation review.
- No claim of novelty, universal superiority, identity, monotonicity, or transitivity follows from these experiments.

## Completion gate

Before execution, create baseline implementations/tests, a versioned configuration, and an experiment script that writes machine-readable results.
