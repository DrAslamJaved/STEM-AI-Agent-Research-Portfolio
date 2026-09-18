# Phase 10 — Pre-registered External Davis DTI Application Protocol

## Status

**APPROVED 2026-09-17 — implementation authorized; experimental execution
remains gated by an explicit `--execute` command.**

This is an external-data application of the approved fuzzy similarities.  It
follows Phase 09 and does not alter, replace, or tune its completed synthetic
study.

## Research question

On a leakage-audited real drug–target interaction (DTI) benchmark, does the
approved similarity \(\theta_1\) provide a practically meaningful improvement
over min–max Jaccard, min–max Dice, or cosine when all methods use the same
fuzzy pair representation and weighted \(k\)-nearest-neighbour (kNN)
classifier?

The study tests this explicitly defined representation, classifier, data
threshold, and splits only.  It does not claim that one similarity is superior
for all DTI models or all datasets.

## Frozen data source and provenance gate

Use the validated Davis kinase benchmark already acquired for Project 02:

- source: `hkmztrk/DeepDTA`, pinned commit
  `a546a8433a6822e958f36171c4356ad6f414d623`;
- original scale: 72 kinase inhibitors and 442 kinases; processed representation:
  68 canonical drugs and 442 targets;
- task label: `interaction_kd_le_1000_nM` (\(K_d\le1{,}000\) nM), the prior
  Project 02 primary label;
- data and split provenance must be recorded with SHA-256 hashes before any
  model result is generated.

Before execution, the runner must locate the existing Project 02 validated
table and its split-assignment file, verify the expected columns and report
their hashes.  If files, labels, or split identifiers are absent or incompatible,
the runner must stop rather than reconstructing, downloading, or silently
changing data.

The \(K_d\le100\) nM label is excluded from the primary analysis.  It may only
be used later as a separately labelled sensitivity analysis under a new protocol.

## Fuzzy pair representation

Construct one fixed feature vector for each drug–target pair using only its
available molecular string and target sequence:

1. **Drug component:** relative frequencies over the fixed printable-SMILES
   character alphabet
   `#%()+-.0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz`.
   Characters outside this alphabet are assigned to one additional `OTHER`
   component.  Divide every character count by the SMILES length.
2. **Target component:** relative frequencies of the 20 standard amino acids in
   the fixed order `ACDEFGHIKLMNPQRSTVWY`; non-standard residues are assigned
   to one additional `OTHER` component.  Divide every count by sequence length.
3. Concatenate the two components.  Every coordinate is therefore a membership
   in \([0,1]\).  No label-derived transformation, global scaling, feature
   selection, or learned embedding is permitted.

The same exact vector is used by all four methods.  This deliberately simple,
transparent representation tests similarity behaviour rather than maximizing
DTI prediction performance.

## Compared similarities and classifier

Evaluate exactly:

1. approved \(\theta_1=(1,0,0,0,0;1,1,1,0,1)\);
2. min–max fuzzy Jaccard;
3. min–max fuzzy Dice;
4. cosine similarity.

For each method use weighted kNN.  For a test pair \(x\), obtain its \(k\)
most similar training pairs and calculate

\[
\hat p(x)=
\frac{\sum_{i\in N_k(x)}S(x,x_i)y_i}
     {\sum_{i\in N_k(x)}S(x,x_i)}.
\]

If the denominator is zero, use the training-fold positive prevalence.  Ties
at the neighbour boundary are resolved by ascending training-row index.

Select \(k\in\{3,5,9,15,25\}\) independently for every similarity and outer
split using training data only: maximize mean validation average precision
(AP) in the existing split-compatible inner folds, then break ties by choosing
the smaller \(k\).  The outer test data are never used for representation,
selection, or tuning.

## Frozen evaluation splits

Reuse Project 02's existing fixed split assignments (random state 20260830):

- **Primary:** cold-drug holdout, with no held-out drug entity in training.
- **Co-primary:** cold-target holdout, with no held-out target entity in
  training.
- **Reference only:** pair-random holdout.

The runner must verify zero exact-pair overlap for every split and zero held-out
entity overlap for each cold split.  No resampling, re-stratification, or split
replacement is permitted.

## Outcomes and comparisons

Primary outcome: test-set AP.  Secondary outcomes: ROC-AUC, Brier score,
precision, recall, and F1 at a decision threshold selected on training data
only.  Report selected \(k\), training prevalence, test prevalence, and
prediction count for every method and split.

For each outer split and comparator, compute the paired difference
\(\Delta=\mathrm{AP}_{\theta_1}-\mathrm{AP}_{m}\).  Estimate a 95% paired
bootstrap confidence interval with 10,000 resamples:

- resample held-out drugs for the cold-drug split;
- resample held-out targets for the cold-target split;
- stratified pair-resample for the pair-random reference split.

Resamples lacking either class are discarded and the accepted-resample count is
recorded.  Use seed 20261020 plus a fixed split/comparator offset.  No
post-hoc resampling scheme is permitted.

## Decision rule

Call \(\theta_1\) *better on one external split* only if both conditions hold
against a comparator:

- mean AP difference is at least 0.02; and
- the paired 95% bootstrap interval lower bound is greater than zero.

Call an external advantage *replicated* only if the condition holds for the
same comparator in **both** cold-drug and cold-target splits.  Otherwise report
the relevant comparison as **no supported advantage**.  The pair-random split
cannot establish a generalization claim by itself.

## Execution, reporting, and interpretation boundaries

The runner must refuse execution without `--execute`, write a JSON result,
and store test predictions and machine-readable provenance.  It must not
commit raw data.

Results will be labelled `EXPERIMENTALLY_SUPPORTED`.  They may support a
bounded statement about this transparent DTI representation and these audited
splits; they do not prove mathematical properties, causal mechanisms, or
general superiority.  A paper must report the negative result if no comparison
meets the decision rule.
