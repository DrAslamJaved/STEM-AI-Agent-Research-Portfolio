# Phase 07 protocol: structure input and crystal graph

## Motivation and task choice

The prior `matbench_expt_gap` task is composition input. A crystal graph model would require structures that are not supplied by that task, so its inclusion there would not have been justified. Phase 07 uses `matbench_dielectric`, a Matbench v0.1 structure-input regression task with 4,764 samples, target `n`, and unit `unitless`. The Matbench task metadata is the authoritative source for these fields: <https://github.com/materialsproject/matbench/blob/main/matbench/matbench_v0.1_dataset_metadata.json>.

This phase constructs and audits inputs only. It does not train a graph model or claim a benchmark result.

## Descriptor baseline

The future classical baseline receives 129 deterministic structure features:

1. Fractions of atomic numbers 1 through 118.
2. Site count, density, and volume per atom.
3. Three lattice lengths and three lattice angles.
4. Mean and standard deviation of atomic number across sites.

These fields are intentionally transparent and are not tuned against an outer test fold.

## Crystal graph

Each atom is a node whose stored feature is atomic number. For each central site, all periodic neighbours within 5.0 Å are sorted by distance, neighbour site index, and periodic-image coordinates. At most the first 12 are retained. An edge stores the source site, neighbour site, and Cartesian intersite distance. The graph is directed: a physical pair may appear in each direction. A positive-distance periodic image of the same site index is retained, because it represents a physical periodic neighbour.

The radius and neighbour cap are locked for Phase 08. A later graph model must use the exact graph protocol for every official fold and must disclose any graph that has no retained edge.

## Audit

`p09-structure-audit` loads one official fold, constructs descriptors and graphs for a requested prefix of both train and test structure inputs, and writes only counts and graph summaries. It calls the Matbench test API with `include_target=False`. Target values, test metrics, and trained-model choices do not enter this audit.

## Next phase

Phase 08 will set a fixed CPU training budget and compare a random forest on the 129 descriptors against a graph network on these graphs. It will use the same official folds and report wall-clock time, MAE, RMSE, and failure status. Any uncertainty evaluation follows model selection and retains the Phase 06 calibration distinction.
