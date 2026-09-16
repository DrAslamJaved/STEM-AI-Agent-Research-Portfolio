# Phase 08 — Synthetic Clustering Application Protocol

## Status

Pre-execution protocol only. No clustering result is reported.

## Objective

Assess whether the approved \(\theta_1\) similarity separates known synthetic fuzzy clusters differently from Jaccard, Dice, and cosine.

## Data generation

- three latent clusters; 100 vectors per cluster; dimension \(n=10\);
- fixed seed `20260908`;
- cluster centres: ten-dimensional membership prototypes with predominantly low, medium, or high memberships;
- each observation: centre plus independent bounded noise, clipped to `[0,1]`;
- true cluster labels are retained only for evaluation, never supplied to clustering.

## Methods

For each similarity, form the full pairwise similarity matrix and apply the
same deterministic agglomerative clustering procedure with three clusters.

## Outcomes

- Adjusted Rand Index against known labels;
- within-cluster mean similarity;
- between-cluster mean similarity;
- separation gap: within minus between;
- cluster-size vector and any invalid numerical value.

## Rules

- all methods use identical generated vectors and the same target cluster count;
- no tuning after inspecting results;
- a larger score is not called superiority beyond this synthetic generator;
- output is `EXPERIMENTALLY_SUPPORTED` only, never a theorem or novelty claim.

## Next gate

Human approval before implementation and execution.
