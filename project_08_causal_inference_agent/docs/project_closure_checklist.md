# Project 08 closure checklist

## Repository integrity

- [ ] All twelve phase pull requests are merged in order.
- [ ] Local `main` and `origin/main` are synchronized.
- [ ] The worktree is clean and obsolete phase branches are removed.
- [ ] No generated datasets, credentials, caches, or local environments are tracked.

## Scientific integrity

- [ ] The mathematical and causal-question contracts remain explicit.
- [ ] Synthetic truth-recovery experiments are reproducible across seeds.
- [ ] Invalid and non-identifiable scenarios emit no numerical causal conclusion.
- [ ] Identification, estimability, robustness, and human approval remain distinct.
- [ ] Diagnostics include overlap, standardized mean differences, weights, and effective sample size.
- [ ] Refutation and hidden-confounding sensitivity results are reported with limitations.

## Benchmark integrity

- [ ] IHDP provenance, version, replication, and SHA-256 evidence are recorded when used.
- [ ] Lalonde provenance and any external reference ATE are explicitly declared when used.
- [ ] ATE error and confidence-interval coverage are reported only when reference truth exists.
- [ ] PEHE is reported only for benchmarks with individual treatment-effect truth.
- [ ] Lalonde analyses never report PEHE without individual-effect truth.

## Engineering and release

- [ ] The complete test suite passes on Python 3.11 and 3.12.
- [ ] Branch coverage is at least 98% on Windows and Linux CI.
- [ ] JSON output is machine-readable and Markdown output is human-auditable.
- [ ] Reports separate question, DAG, identification, estimation, diagnostics, refutation, sensitivity, and benchmarking.
- [ ] The model card and orchestration protocol match the released implementation.
- [ ] Human approval of the DAG and final interpretation is recorded outside the automated result.
