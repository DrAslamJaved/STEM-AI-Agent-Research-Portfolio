# Project 09: Uncertainty-Aware Materials Discovery Agent

## Phases 01–07: learning curves, uncertainty calibration and structure graphs

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

### Phase 05: optional per-material diagnostics

Re-run the **official** five-fold comparison with `--predictions-output` to export per-material ensemble evaluation records in a separate JSON file. This takes approximately one additional full experiment run; the new file contains a row for every test material, policy, budget, fold and seed. It records absolute error, model disagreement, nominal 90% interval inclusion and the number of constituent elements. It excludes raw test targets and formula strings. The usual checkpoint JSON and its predeclared primary conclusion remain separate. The report refuses prediction records whose stored checkpoint SHA-256 differs from the actual checkpoint JSON, and verifies that their aggregated MAE and coverage agree with each saved checkpoint.

```powershell
.\.venv\Scripts\python.exe -m p09.experiment --folds 0 1 2 3 4 --seeds 17 23 --include-diversity --output results\phase05_official.json --predictions-output results\phase05_predictions.json
.\.venv\Scripts\python.exe -m p09.conditional_report results\phase05_predictions.json --checkpoints results\phase05_official.json --output results\phase05_conditional_report.md
```

The conditional report compares coverage in the lowest and highest thirds of predicted ensemble disagreement, computes a descriptive rank correlation between disagreement and absolute error, and reports coverage by fixed element-count strata (one, two, three or more) at the final budget. It withholds an element-count stratum unless every fold/seed has at least 20 test rows there. These subgroup checks are **exploratory**, because Phase 03 results were already seen. Coverage close to 0.90 overall does not imply coverage close to 0.90 in each subgroup. No subgroup analysis changes which labels are acquired.

### Phase 06: normalized conformal intervals

Phase 05 showed that ensemble disagreement tracks absolute error (descriptive Spearman correlations 0.60–0.79), yet fixed-width split-conformal intervals covered only 0.75–0.80 of the highest-disagreement third across the Phase 03 budgets. Phase 06 evaluates an additional, fixed uncertainty interval that scales its half-width by ensemble disagreement. It uses calibration scores

\[
\frac{|y_i-\hat y_i|}{\max(s_i,\; q_{0.10}(s_{\mathrm{cal}}))},
\]

where \(s_i\) is ensemble disagreement and the 10th-percentile floor and finite-sample conformal score quantile use the fixed calibration partition only. The original constant-width interval remains in the output for direct comparison. Acquisition, folds, labels, models, and error threshold are unchanged.

```powershell
.\.venv\Scripts\python.exe -m p09.experiment --folds 0 1 2 3 4 --seeds 17 23 --include-diversity --include-normalized-conformal --output results\phase06_official.json --predictions-output results\phase06_predictions.json
.\.venv\Scripts\python.exe -m p09.normalized_report results\phase06_predictions.json --checkpoints results\phase06_official.json --output results\phase06_normalized_report.md
```

Compare scaled coverage and interval width together, particularly in the high-disagreement third. The Phase 06 comparison is an uncertainty-calibration study after inspecting Phase 05; it does not alter the locked Phase 03 label-efficiency conclusion.

### Phase 07: structure task and crystal-graph foundation

`matbench_expt_gap` provides compositions but no crystal structures, so a graph neural network was not justified there. Phase 07 moves the graph comparison to `matbench_dielectric`, a structure-input regression task with 4,764 records and a unitless dielectric target. It defines two matched inputs for the later model comparison:

* A transparent 129-column structure descriptor: 118 atomic-number fractions, site count, density, volume per atom, lattice lengths and angles, and atomic-number mean and standard deviation.
* A periodic crystal graph with atomic-number nodes and directed neighbour edges. Edges use a fixed 5.0 Å radius, retain at most 12 nearest neighbours per site, and carry intersite distance.

Run the audit before training a graph model. It samples structures from the official fold without recording targets and checks graph and descriptor construction on both training and test partitions.

```powershell
.\.venv\Scripts\python.exe -m p09.structure_task --fold 0 --limit 64 --output results\phase07_structure_audit.json
```

The graph construction is deterministic: periodic neighbours are sorted by distance, site index and periodic image before the 12-neighbour cap. The audit is a data-contract check, not a model score. Phase 08 will define the CPU compute budget and compare the descriptor forest with a graph network on the same official folds.

### Design locks

* Official five-fold Matbench split defines the outer test set. The inner calibration set is selected from training records by reduced-composition groups; no calibration record is queried during acquisition. The pool and test labels cannot influence selection.
* At each seed/fold, all enabled policies start with **the same** 200 randomly drawn labels and then each acquire to exactly the stated budgets. Uncertainty selection uses highest bootstrap ensemble prediction spread; ties use a seeded randomized ordering. Random draws without replacement from the remaining pool.
* Each checkpoint fits a property-mean model, one random forest, and an independently bootstrapped forest ensemble. It records MAE, RMSE, 90% interval coverage, interval mean width, and test count. Conformal residual quantiles come only from the fixed calibration set. Coverage is marginal under exchangeability, not a per-material guarantee.
* The target `0.60 eV` is a **planning threshold**, not a claimed result or a literature benchmark. The first crossing is measured at the scheduled budgets only. Compare paired fold/seed crossings and report failures; do not select the target after seeing test curves.
* The implementation audits overlapping reduced compositions between outer training and outer test, plus repeated reduced compositions inside training. Official splits remain untouched. Overlap can make a composition-only model look optimistic; disclose it and follow with a separate group-exclusive robustness analysis before claiming chemical generalization.
* Fixed RF hyperparameters avoid tuning against the outer test. Comparisons are internal learning curves; they are **not** a Matbench leaderboard submission. Report 5 folds and multiple seeds before any scientific claim. Acquisition uses predictive disagreement, which may be weak or miscalibrated in unfamiliar chemical families.

## Files

`src/p09/experiment.py` loads the official fold and writes the complete reproducibility record. `src/p09/core.py` implements fixed-budget evaluation without knowledge of test targets in the selection code. `src/p09/report.py` reports the original paired comparison, while `src/p09/diversity_report.py` reports the three-policy comparison with fold-level intervals (seeds averaged within fold), coverage and failed crossings. `src/p09/diagnostics.py` audits saved runs and reports model-specific exploratory checks. `src/p09/conditional_report.py` checks optional prediction-level records, and `src/p09/normalized_report.py` compares fixed and disagreement-scaled conformal intervals. `src/p09/structure_graph.py` creates graph and descriptor inputs from periodic structures; `src/p09/structure_task.py` audits their Matbench adapter. `tests/` checks split disjointness, budget parity, interval quantiles, label independence and reporting behavior with synthetic data.

## Next research phases

1. Preserve the Phase 03 negative result and inspect the Phase 04 evidence audit, including exploratory single-forest outcomes.
2. Use the Phase 07 data contract to prepare the fixed-budget descriptor-versus-graph model comparison.
3. Train and assess the graph model on the official `matbench_dielectric` folds, including uncertainty calibration on the selected model.

Source: Dunn et al., *npj Computational Materials* **6**, 138 (2020), DOI: [10.1038/s41524-020-00406-3](https://doi.org/10.1038/s41524-020-00406-3). Dataset and task metadata: [Materials Project Matbench](https://docs.materialsproject.org/services/ml-and-ai-applications/matbench), [Matbench source](https://github.com/materialsproject/matbench).
