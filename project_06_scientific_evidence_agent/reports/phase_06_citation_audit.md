# Phase 06 — Cross-validated citation audit

## Selection boundary

SciFact's supplied five folds include the ordinary 300-claim development split.
To preserve a genuine final development evaluation, the fold assignment was
filtered to the 809 ordinary training claims before any model or threshold was
selected. The resulting fold sizes were:

| Fold | Train claims | Validation claims | Ordinary-dev claims excluded |
| --- | ---: | ---: | ---: |
| 1 | 647 | 162 | 60 |
| 2 | 645 | 164 | 58 |
| 3 | 655 | 154 | 68 |
| 4 | 642 | 167 | 55 |
| 5 | 647 | 162 | 59 |

Each fold model and its complete gold-free candidate trace were written before
the corresponding validation labels were loaded. The audit searched 60 fixed
policies and required at least 20% pooled assertion coverage.

## Selected policy

The pooled out-of-fold selector chose:

| Parameter | Value |
| --- | ---: |
| Assertion threshold | 0.65 |
| Sentence-evidence threshold | 0.60 |
| Maximum sentences per citation | 2 |

Its pooled out-of-fold unsupported-assertion rate was 0.9146 at 0.4054
coverage. This is a selection result, not the final development result.

## Held-out development comparison

Both policies were applied to the exact same frozen 300-claim development
trace from a verifier trained on all 809 ordinary training claims.

| Measure | Fixed Phase 05 policy | Selected audit policy | Difference |
| --- | ---: | ---: | ---: |
| Claim macro-F1 | 0.4005 | 0.4177 | +0.0172 |
| Sentence evidence F1 | 0.0531 | 0.0428 | −0.0103 |
| Strict citation correctness F1 | 0.0067 | 0.0083 | +0.0016 |
| Faithfulness | 0.0581 | 0.0690 | +0.0108 |
| Coverage | 0.8600 | 0.4833 | −0.3767 |
| Unsupported-assertion rate | 0.9419 | 0.9310 | −0.0108 |

The audit reduces unsupported assertions by 1.08 percentage points, but it
does so by withholding assertions on substantially more claims. Its citation
and evidence F1 values remain far too low for a reliable scientific-verification
system. This phase therefore provides a valid, coverage-aware baseline for a
future learned NLI or cross-encoder improvement; it does not establish a useful
production citation auditor.
