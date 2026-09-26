# Phase 04: evidence audit of the five-fold Phase 03 experiment

## Provenance and scope

- Source: `phase03_results_review.zip`, SHA-256 `2c302543330223329bce308e1553a7136b49c326565af135f272645a78f5b2c2`.
- Files: `phase03_official.json`, `phase03_group_exclusive.json`, and their corresponding Markdown reports.
- Task: `matbench_expt_gap`; 5 outer folds, seeds 17 and 23, budgets 200/400/800/1600; target ensemble MAE at most 0.60 eV.
- All three policies begin with the same 200 labelled records within each fold/seed. Split-conformal calibration remains disjoint from acquisition.
- These are retrospective selection runs; no new materials were synthesized.

## Primary ensemble analysis

| Labels | Random MAE (eV) | Uncertainty MAE (eV) | Hybrid MAE (eV) |
| ---: | ---: | ---: | ---: |
| 200 | 0.784 | 0.784 | 0.784 |
| 400 | 0.752 | 0.771 | 0.766 |
| 800 | 0.681 | 0.687 | 0.715 |
| 1600 | 0.608 | 0.611 | 0.614 |

At 800 labels, the hybrid is worse than both comparators in every fold after averaging the two predeclared seeds. Its hybrid MAE gain versus random is -0.034 eV overall (five-fold bootstrap 95% descriptive interval -0.047 to -0.023 eV). At the locked threshold, 5/10 random runs, 4/10 uncertainty runs, and 1/10 hybrid runs cross at a scheduled budget. The one paired hybrid crossing occurs at 1600 labels and saves zero labels. The experiment does not support a label-efficiency gain for the hybrid.

Mean ensemble interval coverage by policy stays approximately 0.89–0.90 across the scheduled budgets; this alone does not demonstrate calibrated coverage for particular materials or composition families. Five-fold bootstrap intervals are descriptive and imprecise.

## Composition overlap and robustness

The outer-fold audits find zero identical reduced compositions shared between training and test in every fold. Group-exclusive filtering therefore removes zero training records. Official and group-exclusive results have identical selections, identical crossing counts, and numerically equal checkpoint metrics to within 1.8e-15 (floating-point rounding). The second run is redundant for this dataset; it cannot independently establish generalization to chemically dissimilar materials.

## Exploratory forest analysis

At 1600 labels, the single-forest MAE is 0.588 eV under random selection, 0.578 eV under uncertainty selection, and 0.592 eV under the hybrid. Calculating forest crossings after inspecting the ensemble outcome yields 6/10, 9/10, and 8/10 respectively. These figures motivate a later model-aware study; they are **post hoc exploratory results**, not evidence that the locked primary target was met by a superior active-learning policy. Do not change the 0.60 eV target or rewrite the primary conclusion after seeing these values.

## Next research decision

Preserve this negative acquisition result. A subsequent phase can export per-material predictions and intervals to examine uncertainty-error alignment and conditional coverage, with its rerun cost and analysis groups specified first. A crystal graph model belongs with a structure-input task, not the current composition-only inputs.
