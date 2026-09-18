# Phase 10R Numerical-boundary Correction

- **Date:** 2026-09-18
- **Observed execution stop:** `brier_score_loss` rejected a weighted-kNN
  probability marginally above one before any valid Phase 10R result was
  written.
- **Correction:** The runner now accepts only finite probabilities within
  \(10^{-12}\) of \([0,1]\), clips those round-off boundary values, and fails
  for a larger violation. It writes CSV/JSON outputs to temporary files and
  publishes them only after a complete successful run.
- **No design change:** raw source, labels, split seed, methods, k grid,
  selection, bootstrap, and decision rule are unchanged. Any prior partial
  output is invalid and must not be interpreted.
