# Phase 08R — Completion Record and Interpretation

## Status

**COMPLETE — EXPERIMENTALLY_SUPPORTED; no comparative advantage established.**

This record closes the revised synthetic clustering application defined in
`phase_08_revised_application_protocol.md`.  It does not replace or erase the
original Phase 08 result, whose magnitude-based generator was recorded as
unsuitable for comparative inference.

## Executed design

- Three latent fuzzy clusters; 100 observations per cluster; dimension 10.
- Thirty independent replications with seeds 20260909–20260938.
- Equal sigma-count prototype cardinality (5.0) and non-collinear membership
  patterns.
- Identical generated observations for the approved rational setting
  \(\theta_1\), min–max Jaccard, min–max Dice, and cosine.
- Deterministic average-linkage agglomerative clustering into three clusters.
- True labels were retained solely to calculate adjusted Rand index (ARI).

## Recorded result

All four methods recovered the three latent clusters perfectly in every
replication:

| Method | Mean ARI | Standard deviation | Range |
|---|---:|---:|---|
| \(\theta_1\) | 1.000000 | 0.000000 | [1.000000, 1.000000] |
| Jaccard | 1.000000 | 0.000000 | [1.000000, 1.000000] |
| Dice | 1.000000 | 0.000000 | [1.000000, 1.000000] |
| Cosine | 1.000000 | 0.000000 | [1.000000, 1.000000] |

## Interpretation and permitted claim

The revised generator is cleanly recoverable, but is not difficult enough to
differentiate the methods.  Therefore, this study supports only the following
statement:

> On the pre-specified Phase 08R synthetic generator, all four evaluated
> similarities achieved perfect average-linkage cluster recovery; no
> comparative advantage for \(\theta_1\) was observed.

The result does **not** support claims of superiority, robustness in general,
or practical advantage.  The absence of a difference here must not be treated
as evidence of equivalence.

## Handoff

The next study is Phase 09, a separately pre-registered robustness study with
increasing, cardinality-preserving perturbation.  Its protocol must be approved
before its runner is implemented or executed.
