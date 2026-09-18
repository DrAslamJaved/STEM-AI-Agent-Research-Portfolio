# Phase 10R — Re-frozen External Davis DTI Application Protocol

## Status

**APPROVED 2026-09-18 — implementation authorized; experimental execution remains gated by an explicit `--execute` command.**

Phase 10R is a new study, created because Phase 10 closed without execution when its frozen local CSV inputs could not be recovered. It is not a repair, continuation, or reanalysis of Phase 10. No Phase 10 or Project 10 v0.2 model prediction, split, label, or result may enter this study.

## Research question

Using a transparent fuzzy representation built from the retained raw Davis source, does the approved rational similarity \(\theta_1\) provide a practically meaningful improvement over min--max Jaccard, min--max Dice, or cosine for unseen-drug and unseen-target DTI prediction with the same weighted kNN classifier?

The claim is limited to this re-frozen data construction, representation, classifier, and evaluation design.

## Newly frozen source and construction gate

The only permitted raw inputs are located under the validated Project 10 v0.2 worktree:

```text
G:\Research\STEM\Project10_v02_clean\project_10_stem_research_ai_agent\data\raw\davis\
  ligands_can.txt
  proteins.txt
  Y
```

They must match DeepDTA commit `a546a8433a6822e958f36171c4356ad6f414d623`. Before construction, the runner must record SHA-256 hashes for all three files and compare them with the retained `v0_2_davis_manifest.json` when that manifest supplies hashes. Any mismatch, unreadable file, non-finite affinity, dimensional inconsistency, or unexpected missing value is a fail-closed stop.

`ligands_can.txt` and `proteins.txt` must be decoded as UTF-8 JSON mappings, preserving their source iteration order. `Y` is a trusted, binary legacy Python pickle and must be read only after its hash check with `pickle.load(handle, encoding="latin1")`, then converted with `numpy.asarray`. The resulting affinity matrix must be a finite, strictly positive, rectangular numeric array with shape `(68, 442)`: rows align to the ligand mapping and columns align to the protein mapping. No transpose, reorder, imputation, or missing-value filtering is permitted.

The runner must construct a new table directly from those raw inputs and record its hash. It must require exactly 68 canonical drugs, 442 targets, and 30,056 finite observed affinities. For each pair, preserve the source row and column indices, canonical drug identifier, SMILES, target identifier, sequence, and \(K_d\) in nM. Define the primary label before inspection as

\[
y=\mathbb{1}\{K_d\le 1{,}000\ \mathrm{nM}\}.
\]

The \(K_d\le100\) nM label is excluded from Phase 10R. It requires a separate future protocol. Neither pKd \(\ge7\) nor Project 10 v0.2's label is permitted.

## Fixed fuzzy pair representation

For every pair, construct the same label-free vector used in the Phase 10 design:

1. relative frequencies over the printable-SMILES alphabet `#%()+-.0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz`, plus `OTHER`;
2. relative frequencies over amino acids in the order `ACDEFGHIKLMNPQRSTVWY`, plus `OTHER`;
3. concatenate the two components without scaling, feature selection, learned embedding, or label-derived transformation.

All coordinates must be in \([0,1]\), and each component block must sum to one within \(10^{-12}\). An empty source string stops the study.

## Newly frozen splits

After table construction and before model fitting, generate and persist one new split-assignment file using random state **20260918**. This is a new design, not a replacement for the unavailable Phase 10 assignments.

- **Primary cold-drug:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260918)`, group = drug identifier, outer fold 0 as test.
- **Co-primary cold-target:** the same procedure, group = target identifier, outer fold 0 as test.
- **Reference pair-random:** `StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=20260918)`.

The runner must verify each test set contains both classes, every pair has one assignment, pair overlap is zero in all splits, and the relevant held-out entity overlap is zero in each cold split. It must report counts and entity-overlap diagnostics. A failed check stops the study; no alternative fold, seed, resampling, or replacement split is permitted.

## Methods, selection, and outcomes

Compare exactly the approved \(\theta_1=(1,0,0,0,0;1,1,1,0,1)\), min--max Jaccard, min--max Dice, and cosine similarities. Each uses weighted kNN with \(k\in\{3,5,9,15,25\}\). The weight and zero-denominator rule are exactly:

\[
\hat p(x)=
\frac{\sum_{i\in N_k(x)} S(x,x_i)y_i}
     {\sum_{i\in N_k(x)} S(x,x_i)},
\]

with training-fold positive prevalence used only when the denominator is zero. At a neighbour boundary, order ties by ascending constructed-pair index.

Every weighted prediction must be finite and in \([0,1]\). Values within
\(10^{-12}\) of a boundary are clipped to that boundary solely to remove
floating-point round-off; a larger violation is a fail-closed error.

For each outer split and similarity, choose \(k\) using only the outer-training partition: run five deterministic inner folds grouped by the outer split's entity type for cold splits, or stratified for the pair-random split; maximize mean validation AP and choose the smaller \(k\) on a tie. Test labels are never used for representation, selection, or fitting.

Primary outcome: outer-test average precision (AP). Secondary outcomes: ROC-AUC, Brier score, precision, recall, and F1 using a threshold selected only within outer training data. Store all test predictions, selected \(k\), training and test prevalence, and provenance in machine-readable outputs.

## Inference and decision rule

For every outer split and comparator, calculate \(\Delta=\mathrm{AP}_{\theta_1}-\mathrm{AP}_{m}\). Estimate a 95% paired bootstrap interval using 10,000 accepted resamples and seed **20260919** plus a fixed split/comparator offset. Resample held-out drugs for cold-drug, held-out targets for cold-target, and stratified pairs for the reference split. Discard resamples lacking either class and record their count.

Call \(\theta_1\) *better on an external split* only when both conditions hold:

- \(\Delta\ge0.02\); and
- the bootstrap interval lower bound is greater than zero.

Call an advantage *replicated* only when it holds against the same comparator in both cold splits. Otherwise report **no supported advantage**. The pair-random split is reference-only and cannot establish generalization.

## Execution and interpretation boundaries

Implementation must refuse execution without `--execute`. Before any result, the runner must write the constructed-table hash, raw-input hashes, split-file hash, and validation diagnostics. It must not commit raw data.

Results are labelled `EXPERIMENTALLY_SUPPORTED`. They describe only Phase 10R; they do not prove mathematical superiority or validate all DTI models and must report null findings faithfully.
