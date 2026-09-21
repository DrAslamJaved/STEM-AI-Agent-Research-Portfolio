# Benchmark data and causal evaluation protocol

Phase 11 evaluates causal estimates without confusing an experimental reference
with unit-level counterfactual truth. Benchmark files remain local and are never
downloaded implicitly or committed by this phase.

## IHDP

The loader accepts a single-replication CSV or an NPZ containing `x`, `t`, `yf`,
`mu0`, and `mu1`. Canonical multi-replication NPZ arrays are selected by an
explicit zero-based replication index. The individual truth is `mu1 - mu0`, the
reference ATE is its mean, and PEHE is the root mean squared error between the
estimated and reference individual effects.

## Lalonde/NSW

The loader accepts a local CSV with binary treatment, observed earnings outcome,
and numeric covariates. The data do not reveal both potential outcomes for any
individual. Therefore PEHE is unavailable and must not be synthesized. ATE error
and confidence-interval coverage are calculated only when the caller explicitly
supplies an experimental average-effect reference. Such a value is recorded as
an average reference, not individual ground truth.

## Metrics

- ATE bias: estimated ATE minus reference ATE.
- ATE error: absolute ATE bias.
- ATE RMSE: root mean squared ATE bias across replications.
- PEHE: root mean squared individual-effect error when individual truth exists.
- CI coverage: fraction of supplied intervals containing the corresponding
  reference ATE.

Metrics are counted independently during aggregation so missing PEHE or coverage
cannot silently change the denominator of ATE metrics. Invalid predictions,
incoherent ATE/ITE inputs, malformed intervals, truth leakage, missing values,
and non-binary treatment data are rejected. Replication summaries reject invalid
evaluation results and cannot mix different benchmark datasets.

## Provenance and limitations

Every loaded file records its absolute local path, SHA-256 digest, declared
source URI, version, and IHDP replication index. Users must verify acquisition
rights and dataset-specific licensing before use. Benchmark performance does not
establish identification, external validity, or correctness on other datasets.

## Primary references

- Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference.
  *Journal of Computational and Graphical Statistics*, 20(1), 217–240.
  https://doi.org/10.1198/jcgs.2010.08162
- LaLonde, R. J. (1986). Evaluating the econometric evaluations of training
  programs with experimental data. *American Economic Review*, 76(4), 604–620.
