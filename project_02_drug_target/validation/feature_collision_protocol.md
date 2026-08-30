Davis Transparent-Feature Collision Audit Protocol

Purpose

This audit tests whether the transparent, dependency-free feature representation
can map distinct Davis raw drug or target inputs to the same descriptor vector.
It was added after descriptive error-analysis patterns suggested that some
entity-level summaries could be difficult to interpret under a coarse
representation.

The audit is a feature-quality diagnostic. It is not a model-selection step,
biological validation, or causal analysis.

Frozen Inputs and Scope

Raw drug input: data/raw/davis/ligands_can.txt.

Raw target input: data/raw/davis/proteins.txt.

Descriptors: the existing functions and frozen columns in
src.features.representations.

The audit uses no affinity values, binary labels, split assignments, model
predictions, model coefficients, feature importances, or outer-holdout
results.

The audit is therefore unsupervised with respect to the DTI outcome.

The raw representation mappings cover the full benchmark, including entities in
the frozen outer cold-drug holdout. No holdout outcome values or predictions are
read. Therefore this is post hoc representation documentation only: it must not
be used to alter the already evaluated models, select a new feature set, tune a
threshold, or claim a revised outer-holdout result.

Pre-Specified Checks

For drugs and targets separately, the script records:

Raw-representation duplicates: different entity IDs with exactly identical
raw SMILES or protein-sequence strings.

Exact transparent-feature collisions: different entity IDs with exactly
equal feature vectors.

Distinct-raw feature collisions: exact feature collisions whose raw strings
differ. These identify information lost by the transparent representation.

The ten nearest pairs with distinct raw strings under the fixed descriptor
distance below.

The committed JSON report contains entity IDs and aggregate counts, but does
not store raw SMILES strings or protein sequences. The detailed ID-only pair
table is written locally under data/interim/ and is ignored by Git.

Distance Definition

For each entity type, descriptor distances are computed only from feature
columns with non-zero range across that entity table. Each active feature
difference is divided by that feature's entity-level range. The report records
the mean and maximum absolute normalized difference for each retained nearest
pair. Zero-range columns are listed and excluded only because they cannot
distinguish entities.

This is a transparent descriptor diagnostic. It is not a chemical-similarity
metric, molecular fingerprint similarity, protein-alignment score, sequence
homology measure, or biological-distance measure.

Interpretation Guardrails

An identical raw input can be an intentional property of the benchmark
mapping; it is not automatically a data error.

An exact feature collision shows loss of information in this simple feature
representation, not chemical, protein, pharmacological, or biological
equivalence.

A nearest descriptor pair is descriptive only. It does not establish shared
mechanism, biological similarity, target family membership, or causality.

Results must not be used to retrospectively tune the existing models or the
frozen outer holdout. Any richer representation is a separately specified
future experiment, chosen from training data and/or external knowledge before
a new leakage-safe train/validation/test evaluation.

Reproduction Command

From project_02_drug_target:

& .\.venv\Scripts\python.exe -m src.features.collision_audit `
  --data-dir .\data\raw\davis `
  --top-n 10 `
  --summary-output .\reports\davis_feature_collision_audit.json `
  --pairs-output .\data\interim\davis_feature_collision_pairs.csv

The expected version-controlled output is
reports/davis_feature_collision_audit.json. The detailed CSV is expected to
remain ignored because it is a derived local diagnostic.