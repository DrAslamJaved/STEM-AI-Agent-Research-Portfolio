# Phase 09 — Pre-registered Robustness Study Protocol

## Status

**APPROVED 2026-09-17 — implementation authorized; execution remains gated by
the runner's explicit `--execute` flag.**

This is a new study.  It follows the completed Phase 08R application and does
not tune, repeat, or alter that study after its perfect-recovery outcome.

## Research question

When fuzzy clusters become increasingly overlapping through controlled,
cardinality-preserving perturbation, how do the approved rational setting
\(\theta_1\), min–max Jaccard, min–max Dice, and cosine compare for recovery of
the known latent clusters?

The study tests robustness on this generator only.  It is not designed to
establish a universal ranking of similarities.

## Fixed synthetic generator

### Prototypes

Use three non-collinear ten-dimensional prototypes, each with sigma-count
cardinality 5.0:

\[
\begin{aligned}
c_1&=(.70,.70,.30,.30,.70,.70,.30,.30,.50,.50),\\
c_2&=(.70,.30,.70,.30,.70,.30,.70,.30,.50,.50),\\
c_3&=(.70,.30,.30,.70,.30,.70,.70,.30,.50,.50).
\end{aligned}
\]

### Samples and seeds

- Three latent clusters and 100 observations in each cluster (300 per
  replication).
- 100 independent replications per noise condition.
- For noise condition index \(q\in\{0,1,2,3,4,5\}\) and replicate index
  \(r\in\{0,\ldots,99\}\), use seed
  \(20261000+100q+r\).  Thus all methods receive exactly the same observations
  within a replication.

### Perturbation conditions

For every prototype vector, generate ten i.i.d. \(z_i\sim\mathrm{Uniform}(-1,1)\),
replace \(z\) by \(z-\bar z\mathbf{1}\), and rescale it so its largest absolute
component equals \(\varepsilon\).  Add the resulting vector to the prototype.

Evaluate the six pre-specified amplitudes:

\[
\varepsilon\in\{0.00,0.04,0.08,0.12,0.16,0.20\}.
\]

The rescaled perturbation has sum zero, so every generated vector preserves
the prototype's sigma-count cardinality exactly.  Since all prototype entries
lie in \([.30,.70]\) and \(\varepsilon\le .20\), no clipping is required and
all memberships remain in \([0,1]\).

## Compared methods

1. Approved rational setting
   \(\theta_1=(1,0,0,0,0;1,1,1,0,1)\).
2. Min–max fuzzy Jaccard.
3. Min–max fuzzy Dice.
4. Cosine similarity.

No parameter tuning, metric selection, linkage selection, prototype changes,
or seed changes are permitted after results are inspected.

## Clustering and outcome measures

For each noise level, replicate, and method:

1. construct the complete pairwise similarity matrix;
2. run deterministic average-linkage agglomerative clustering with exactly
   three clusters;
3. calculate ARI against labels not exposed to the clustering algorithm;
4. calculate within-cluster mean similarity, between-cluster mean similarity,
   their separation gap, and predicted cluster sizes.

Store all replicate-level values and seeds in a machine-readable JSON result.

## Primary analysis and decision rule

The primary outcome is ARI.  For each noise condition and comparator, calculate
the paired difference

\[
\Delta_{r,m}=\operatorname{ARI}_{r,\theta_1}-\operatorname{ARI}_{r,m}.
\]

Report the mean difference, its 95% percentile bootstrap confidence interval
(10,000 resamples; seed 20261099), and a two-sided paired permutation p-value
(10,000 sign-flips; seed 20261100).  Adjust the three p-values within each
noise condition by Holm's method.

Call \(\theta_1\) *better on one pre-specified condition* only when all three
criteria hold against a comparator:

- mean paired ARI difference is at least 0.02;
- the 95% bootstrap confidence interval has lower bound greater than 0;
- Holm-adjusted permutation \(p<0.05\).

Otherwise report **no supported advantage** for that comparison.  A performance
claim across all noise levels requires the criterion to hold at two or more
non-zero noise conditions; otherwise report condition-specific findings only.

## Validity checks

Before interpretation, verify:

- exactly 600 replications (6 conditions \(\times\) 100);
- 300 samples and three true classes per replication;
- every generated vector has membership values in \([0,1]\) and sigma-count
  cardinality 5.0 within numerical tolerance \(10^{-12}\);
- each method uses the same generated sample matrix per replication;
- each clustering result has exactly three non-empty clusters;
- JSON is valid and the runner refuses execution without `--execute`.

## Interpretation boundaries

The findings will be labelled `EXPERIMENTALLY_SUPPORTED`.  They do not prove
theorems, establish equivalence, or demonstrate general superiority.  A
publication-strength practical claim also requires a separate external-data
application and a reproducibility release after this study.
