# Project 09: Uncertainty-Aware Materials Discovery Agent

## Phases 01–04: learning curves, composition diversity and evidence audit

**Question:** At a predeclared prediction-error threshold, does uncertainty sampling need fewer experimentally labelled compositions than random sampling?

The first task is `matbench_expt_gap` (Matbench v0.1), a composition-input regression task with 4,604 observations and an experimental gap target in eV. We use the official Matbench outer folds. This is a retrospective simulation: all targets already exist, but pool targets are hidden from the selection algorithm until acquired. It does not demonstrate actual laboratory discovery. The comparator is a property-mean baseline, a random forest using element-fraction and simple elemental summary descriptors, and a bootstrap forest ensemble with split-conformal prediction intervals. A crystal graph neural network is deferred to a structure-input task; this dataset supplies no crystal structures.

## Install and run (PowerShell, from this directory)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\p09-experiment.exe --folds 0 --seeds 17 23 --budgets 200 400 800 1600 --target-mae 0.60 --output results\pilot.json
.\.venv\Scripts\p09-report.exe results\pilot.json --output results\pilot_report.md
```

The PyPI `matbench==0.6` release requires SciPy 1.7.3, which is incompatible with Python 3.12. This project installs the official Matbench repository at the pinned source commit in `pyproject.toml`; Git must be installed and GitHub reachable during installation. The first Matbench run downloads its dataset. No benchmark result is bundled here. For the complete prespecified outer-fold analysis run:

```powershell
.\.venv\Scripts\p09-experiment.exe --folds 0 1 2 3 4 --seeds 17 23 --budgets 200 400 800 1600 --target-mae 0.60 --output results\official.json
.\.venv\Scripts\p09-report.exe results\official.json --output results\official_report.md
.\.venv\Scripts\p09-experiment.exe --folds 0 1 2 3 4 --seeds 17 23 --budgets 200 400 800 1600 --target-mae 0.60 --split-mode group_exclusive --output results\group_exclusive.json
.\.venv\Scripts\p09-report.exe results\group_exclusive.json --output results\group_exclusive_report.md
```

The `group_exclusive` robustness mode removes training records with a reduced composition also in that fold's fixed official test set; it does not look at test labels. It is **not an official Matbench benchmark score** because the training data have changed. If filtering leaves too few records for a budget, the run fails explicitly; lower the budgets for a new separately labelled robustness protocol and disclose the change. Inspect JSON for per-fold/per-seed metrics and `threshold_summary`. An absent threshold crossing is reported as `null`, never imputed.

### Phase 03: predeclared diversity comparison

Use `--include-diversity` to run a third acquisition policy with identical folds, seeds, starting labels, calibration set and scheduled budgets. It assigns equal weight to percentile ranks of ensemble disagreement and Euclidean distance from each unlabelled composition to the nearest **currently labelled** composition, using only the 118 element-fraction features. Ties use the seeded ordering. The policy is batch based: distances are recomputed at each scheduled budget, not between acquisitions in one batch. These choices are fixed before viewing the test curves. The existing `p09-report` still reports the original two-policy comparison; the new `p09-diversity-report` requires matched checkpoints for all three policies.

```powershell
.\.venv\Scripts\p09-experiment.exe --folds 0 --seeds 17 23 --budgets 200 400 800 1600 --target-mae 0.60 --include-diversity --output results\phase03_pilot.json
.\.venv\Scripts\p09-diversity-report.exe results\phase03_pilot.json --output results\phase03_pilot_report.md
.\.venv\Scripts\p09-experiment.exe --folds 0 1 2 3 4 --seeds 17 23 --budgets 200 400 800 1600 --target-mae 0.60 --include-diversity --output results\phase03_official.json
.\.venv\Scripts\p09-diversity-report.exe results\phase03_official.json --output results\phase03_official_report.md
.\.venv\Scripts\p09-experiment.exe --folds 0 1 2 3 4 --seeds 17 23 --budgets 200 400 800 1600 --target-mae 0.60 --include-diversity --split-mode group_exclusive --output results\phase03_group_exclusive.json
.\.venv\Scripts\p09-diversity-report.exe results\phase03_group_exclusive.json --output results\phase03_group_exclusive_report.md
```

The report averages repeated seeds within each fold, shows paired hybrid MAE gains against random and uncertainty sampling, and gives descriptive fold-bootstrap intervals only when at least three folds are present. It reports how often each policy meets the locked threshold; label savings are defined only where **both** paired policies reach it. Check all failed crossings and conditional coverage before claiming improved label efficiency. Pilot results alone do not support that claim.

### Phase 04: evaluate the completed runs without retraining

The completed five-fold Phase 03 experiment did **not** show label savings for the hybrid under the predeclared ensemble target. The [Phase 04 findings](docs/phase04_findings.md) retain this result and its input archive checksum. This diagnostic CLI checks checkpoint completeness, identical policy starts, cumulative acquisitions, calibration exclusion and saved crossings; it then compares the forest and ensemble while marking the forest threshold analysis exploratory. It detects when group-exclusive filtering removed zero records. Run it against the original JSON files:

```powershell
.\.venv\Scripts\python.exe -m p09.diagnostics results\phase03_official.json --robustness results\phase03_group_exclusive.json --output results\phase04_evidence_audit.md --summary-json results\phase04_evidence_audit.json
```

The audit uses only stored checkpoint summaries. It cannot estimate errors or interval coverage within material subgroups; those require a later prediction-level export and a new full run. Keep the original 0.60 eV ensemble target and the negative Phase 03 result visible rather than changing the target after seeing the curves.

### Design locks

* Official five-fold Matbench split defines the outer test set. The inner calibration set is selected from training records by reduced-composition groups; no calibration record is queried during acquisition. The pool and test labels cannot influence selection.
* At each seed/fold, all enabled policies start with **the same** 200 randomly drawn labels and then each acquire to exactly the stated budgets. Uncertainty selection uses highest bootstrap ensemble prediction spread; ties use a seeded randomized ordering. Random draws without replacement from the remaining pool.
* Each checkpoint fits a property-mean model, one random forest, and an independently bootstrapped forest ensemble. It records MAE, RMSE, 90% interval coverage, interval mean width, and test count. Conformal residual quantiles come only from the fixed calibration set. Coverage is marginal under exchangeability, not a per-material guarantee.
* The target `0.60 eV` is a **planning threshold**, not a claimed result or a literature benchmark. The first crossing is measured at the scheduled budgets only. Compare paired fold/seed crossings and report failures; do not select the target after seeing test curves.
* The implementation audits overlapping reduced compositions between outer training and outer test, plus repeated reduced compositions inside training. Official splits remain untouched. Overlap can make a composition-only model look optimistic; disclose it and follow with a separate group-exclusive robustness analysis before claiming chemical generalization.
* Fixed RF hyperparameters avoid tuning against the outer test. Comparisons are internal learning curves; they are **not** a Matbench leaderboard submission. Report 5 folds and multiple seeds before any scientific claim. Acquisition uses predictive disagreement, which may be weak or miscalibrated in unfamiliar chemical families.

## Files

`src/p09/experiment.py` loads the official fold and writes the complete reproducibility record. `src/p09/core.py` implements fixed-budget evaluation without knowledge of test targets in the selection code. `src/p09/report.py` reports the original paired comparison, while `src/p09/diversity_report.py` reports the three-policy comparison with fold-level intervals (seeds averaged within fold), coverage and failed crossings. `src/p09/diagnostics.py` audits saved runs and reports model-specific exploratory checks. `tests/` checks split disjointness, budget parity, interval quantiles, label independence and reporting behavior with synthetic data.

## Next research phases

1. Preserve the Phase 03 negative result and inspect the Phase 04 evidence audit, including exploratory single-forest outcomes.
2. Add prediction-level diagnostics for per-material errors and conditional coverage in a later phase, with an explicit repeat-run compute budget.
3. If justified, move to one structure-input Matbench task for a crystal graph model, with an explicit compute budget and matched classical features.

Source: Dunn et al., *npj Computational Materials* **6**, 138 (2020), DOI: [10.1038/s41524-020-00406-3](https://doi.org/10.1038/s41524-020-00406-3). Dataset and task metadata: [Materials Project Matbench](https://docs.materialsproject.org/services/ml-and-ai-applications/matbench), [Matbench source](https://github.com/materialsproject/matbench).
