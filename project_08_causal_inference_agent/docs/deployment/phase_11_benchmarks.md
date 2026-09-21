# Phase 11: Benchmark datasets and causal evaluation metrics

## Objective

Add provenance-aware IHDP and Lalonde adapters plus ATE error, PEHE, and
confidence-interval coverage evaluation without committing benchmark data.

## Deployment order

Phase 10 must already be merged. Extract this overlay into the repository root,
verify the exact file set, and run the complete test and coverage suites.

## Acceptance criteria

- [ ] IHDP CSV and canonical NPZ replications are supported.
- [ ] IHDP individual effects and reference ATE are derived from `mu0` and `mu1`.
- [ ] Lalonde PEHE is unavailable by design.
- [ ] Lalonde ATE metrics require an explicit experimental reference.
- [ ] ATE error, PEHE, CI coverage, and replication summaries are deterministic.
- [ ] File SHA-256 and source metadata are retained.
- [ ] Malformed, leaked, or incompatible benchmark inputs are rejected.
- [ ] The benchmarking module has complete line and branch coverage.

## Files in this overlay

- `project_08_causal_inference_agent/configs/benchmarks/default.yaml`
- `project_08_causal_inference_agent/docs/benchmark_protocol.md`
- `project_08_causal_inference_agent/docs/deployment/phase_11_benchmarks.md`
- `project_08_causal_inference_agent/src/causal_audit_agent/benchmarking.py`
- `project_08_causal_inference_agent/tests/test_benchmarking.py`

## Recommended branch

`feature/project-08-phase-11-benchmark-evaluation`

## Recommended commit

`feat(project08): complete phase 11 benchmark evaluation`

## Scope boundary

This phase does not redistribute datasets, download mutable remote files, train
estimators, or claim that Lalonde provides individual counterfactual truth.
