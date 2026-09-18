# Phase 09 — Completion Record and Interpretation

## Status

**COMPLETE — EXPERIMENTALLY_SUPPORTED; no supported comparative advantage.**

## Executed design

Phase 09 used the approved cardinality-preserving synthetic generator with:

- six fixed noise amplitudes \(\varepsilon\in\{0,.04,.08,.12,.16,.20\}\);
- 100 replications per condition (600 total);
- equal-cardinality, non-collinear fuzzy prototypes;
- average-linkage clustering and ARI as the primary outcome;
- the approved \(\theta_1\) setting, Jaccard, Dice, and cosine;
- paired bootstrap intervals, paired sign-flip permutation tests, and Holm
  adjustment as specified before execution.

## Recorded result

All four methods attained ARI 1.0 in every replication at
\(\varepsilon\le .16\).  At \(\varepsilon=.20\), mean ARIs were:

| Method | Mean ARI | Standard deviation |
|---|---:|---:|
| \(\theta_1\) | 0.997297 | 0.004870 |
| Jaccard | 0.997297 | 0.004870 |
| Dice | 0.997297 | 0.004870 |
| Cosine | 0.995698 | 0.006513 |

The \(\theta_1\)-cosine mean ARI difference was 0.001599, with 95% bootstrap
interval [0.000401, 0.002891] and Holm-adjusted permutation \(p=0.024598\).

## Decision

The pre-registered practical-effect threshold was a mean ARI difference of at
least 0.02.  No comparison met that threshold.  Therefore every comparison is
recorded as **no supported advantage**, including the statistically non-zero
but practically small difference from cosine at \(\varepsilon=.20\).

This study must not be changed or rerun with post-hoc generator settings.  The
next study is a separately pre-registered external-data application.
