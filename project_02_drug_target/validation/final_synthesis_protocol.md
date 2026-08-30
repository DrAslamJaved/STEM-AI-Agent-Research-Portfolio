Davis DTI Final Evidence-Synthesis Protocol

Purpose

This closing step combines already generated, version-controlled Davis artefacts
into one reproducibility record and one readable final report. It does not
train a model, tune hyperparameters, choose a probability threshold, or reopen
model selection.

Frozen Inputs

The program reads only these existing files:

reports/davis_inner_cold_drug_cv.json

reports/davis_threshold_sensitivity.json

reports/davis_split_audit.json

reports/davis_feature_collision_audit.json

validation/dataset_provenance.md

requirements.txt

It records a SHA-256 hash for every input in the JSON output. The caller must
provide the exact Git commit that contains the evaluated input artefacts.

Required Checks

The final synthesis rejects the run unless all of the following hold:

the primary evaluation is cold_drug with zero training/test drug overlap;

the inner CV used only the frozen outer-training partition and records
outer_test_partition_used = false;

every inner-CV fold has zero drug overlap;

the primary label is interaction_kd_le_1000_nM;

all four fixed candidates are present in both affinity-label variants;

the sensitivity report kept model selection closed and performed no
hyperparameter tuning;

the sensitivity report did not use the outer test partition or select its
outcomes;

the pinned DeepDTA commit is present in the dataset-provenance record;

the collision audit used neither predictions nor outcome values.

Reported Results

For the primary 1,000 nM task, the report includes fold mean, standard
deviation, minimum, and maximum for average precision, ROC-AUC, accuracy,
precision, recall, and F1. It also includes the selected model's pooled OOF
confusion matrix at the fixed probability threshold of 0.5.

Average precision is the principal metric because the positive class is a
minority. ROC-AUC is a secondary ranking measure. Accuracy, precision, recall,
F1, and the confusion matrix are operating-point summaries, not a substitute
for PR-AUC in this imbalanced task.

Model Selection and Threshold Sensitivity

The primary selected model remains random_forest_balanced, selected using
unweighted mean inner-fold average precision for the pre-specified 1,000 nM
label. The 100 nM result is a descriptive label-definition sensitivity
analysis. It uses the same frozen folds and fixed configurations and cannot
replace that primary decision.

Raw metric values must not be compared across the two labels as if they were
measurements of the same task: the affinity definition and positive prevalence
change.

Interpretation Boundaries

The final report distinguishes four levels of statement:

Predictive performance: estimates of benchmark performance under the
stated cold-drug CV design.

Statistical evidence: descriptive fold summaries only; five folds do not
demonstrate statistical superiority or yield a causal conclusion.

Biological interpretation: predictions do not experimentally validate
binding, and coarse input features and representation collisions limit
biological interpretation.

Causal claims: none are supported by this workflow.

The cold-drug outer holdout is not read by this final synthesis. Because it was
previously inspected during project development, it must not be portrayed as a
fresh blind confirmatory test for later choices.

Outputs

reports/davis_final_evidence_summary.json — machine-readable evidence,
input hashes, environment, and claim boundaries.

reports/davis_final_evidence_summary.md — concise final scientific report.

Reproduction Command

Run from project_02_drug_target after copying the source and test files:

& .\.venv\Scripts\python.exe -m src.reporting.final_synthesis `
  --source-git-commit 3523d35b445353e254c90135cf356481f6807914 `
  --summary-output .\reports\davis_final_evidence_summary.json `
  --markdown-output .\reports\davis_final_evidence_summary.md

The source commit above is the committed threshold-sensitivity evidence. The
subsequent closure commit should contain this module, its test, this protocol,
and both generated outputs.