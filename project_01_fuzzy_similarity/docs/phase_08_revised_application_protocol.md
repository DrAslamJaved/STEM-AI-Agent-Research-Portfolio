# Phase 08R — Revised Synthetic Clustering Application Protocol

## Status

Pre-execution protocol.  It replaces the unsuitable magnitude-based generator
used in Phase 08; it does not alter the recorded Phase 08 result.

## Objective

Compare the clustering behaviour of the approved \(\theta_1\) setting,
Jaccard, Dice, and cosine on fuzzy clusters distinguished by *membership
pattern*, rather than overall membership magnitude.

## Fixed design

- 3 latent clusters, 100 observations per cluster, and dimension \(n=10\);
- 30 independent replications with seeds \(20260909,\ldots,20260938\);
- each replication uses the identical generated observations for every method;
- cluster prototypes have equal sigma-count cardinality 5.0:

\[
\begin{aligned}
c_1&=(.85,.85,.15,.15,.85,.85,.15,.15,.50,.50),\\
c_2&=(.85,.15,.85,.15,.85,.15,.85,.15,.50,.50),\\
c_3&=(.85,.15,.15,.85,.15,.85,.85,.15,.50,.50).
\end{aligned}
\]

- independently add \(\mathrm{Uniform}[-.08,.08]\) noise to every component,
  then clip values to \([0,1]\);
- retain true labels solely for post-clustering evaluation.

## Methods and evaluation

For each replication and method, construct the full pairwise similarity matrix,
run deterministic average-linkage agglomerative clustering with exactly three
clusters, then calculate:

- adjusted Rand index (ARI);
- within- and between-cluster mean similarity and their gap;
- predicted cluster sizes.

Report each metric across the 30 replications as mean, standard deviation,
minimum, and maximum, plus all per-replication values in JSON.

## Interpretation rules

- Do not tune prototypes, noise, linkage, or parameters after reviewing results.
- A method is called *better on this generator* only if its mean ARI is higher
  and its 30 paired ARI differences versus the comparator are all non-negative,
  with at least one strictly positive difference.
- Otherwise report no comparative advantage.
- Results remain `EXPERIMENTALLY_SUPPORTED` only; they establish neither a
  theorem nor general superiority.

## Execution gate

Implementation and execution require human approval of this revised protocol.
