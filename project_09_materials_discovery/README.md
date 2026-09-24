# Project 09: Uncertainty-Aware Materials Discovery Agent

## Phase 01: reproducible learning-curve experiment

**Question:** At a predeclared prediction-error threshold, does uncertainty sampling need fewer experimentally labelled compositions than random sampling?

The first task is `matbench_expt_gap` (Matbench v0.1), a composition-input regression task with 4,604 observations and an experimental gap target in eV. We use the official Matbench outer folds. This is a retrospective simulation: all targets already exist, but pool targets are hidden from the selection algorithm until acquired. It does not demonstrate actual laboratory discovery. The comparator is a property-mean baseline, a random forest using element-fraction and simple elemental summary descriptors, and a bootstrap forest ensemble with split-conformal prediction intervals. A crystal graph neural network is deferred to a structure-input task; this dataset supplies no crystal structures.

## Install and run (PowerShell, from this directory)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\p09-experiment.exe --folds 0 --seeds 17 23 --budgets 200 400 800 1600 --target-mae 0.60 --output results\pilot.json
```

On some Python environments Matbench's older dependencies may need a compatible environment. The first Matbench run downloads its dataset. No benchmark result is bundled here. For the complete prespecified outer-fold analysis use `--folds 0 1 2 3 4`; retain the other settings. Inspect the JSON for per-fold/per-seed metrics and `threshold_summary`. An absent threshold crossing is reported as `null`, never imputed.

### Design locks

* Official five-fold Matbench split defines the outer test set. The inner calibration set is selected from training records by reduced-composition groups; no calibration record is queried during acquisition. The pool and test labels cannot influence selection.
* At each seed/fold, both policies start with **the same** 200 randomly drawn labels and then each acquire to exactly the stated budgets. Selection uses highest bootstrap ensemble prediction spread; ties use a seeded randomized ordering. Random draws without replacement from the remaining pool.
* Each checkpoint fits a property-mean model, one random forest, and an independently bootstrapped forest ensemble. It records MAE, RMSE, 90% interval coverage, interval mean width, and test count. Conformal residual quantiles come only from the fixed calibration set. Coverage is marginal under exchangeability, not a per-material guarantee.
* The target `0.60 eV` is a **planning threshold**, not a claimed result or a literature benchmark. The first crossing is measured at the scheduled budgets only. Compare paired fold/seed crossings and report failures; do not select the target after seeing test curves.
* The implementation audits overlapping reduced compositions between outer training and outer test, plus repeated reduced compositions inside training. Official splits remain untouched. Overlap can make a composition-only model look optimistic; disclose it and follow with a separate group-exclusive robustness analysis before claiming chemical generalization.
* Fixed RF hyperparameters avoid tuning against the outer test. Comparisons are internal learning curves; they are **not** a Matbench leaderboard submission. Report 5 folds and multiple seeds before any scientific claim. Acquisition uses predictive disagreement, which may be weak or miscalibrated in unfamiliar chemical families.

## Files

`src/p09/experiment.py` loads the official fold and writes the complete reproducibility record. `src/p09/core.py` implements fixed-budget evaluation without knowledge of test targets in the selection code. `tests/` checks split disjointness, budget parity, interval quantiles and selection behavior with synthetic data.

## Next research phases

1. Run all five folds and review overlap, variance, interval coverage, and threshold failures.
2. Add a group-exclusive robustness split and alternative acquisition (uncertainty plus diversity) under the same budget.
3. If justified, move to one structure-input Matbench task for a crystal graph model, with an explicit compute budget and matched classical features.

Source: Dunn et al., *npj Computational Materials* **6**, 138 (2020), DOI: [10.1038/s41524-020-00406-3](https://doi.org/10.1038/s41524-020-00406-3). Dataset and task metadata: [Materials Project Matbench](https://docs.materialsproject.org/services/ml-and-ai-applications/matbench), [Matbench source](https://github.com/materialsproject/matbench).
