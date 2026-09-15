# Evidence-Constrained Baseline Evaluation on the Davis DTI Benchmark

## Abstract

This researcher-controlled draft reports fixed-feature baseline evaluation on the
`davis_kinase_deepdta_layout` dataset. The workflow preserves dataset provenance, evaluates
pair-random and cold-start conditions separately, and constrains contextual
statements to Crossref-verified, human-approved evidence. It does not claim
clinical validity, experimental binding confirmation, or state-of-the-art DTI
performance.

## Introduction

The Davis study profiled kinase-inhibitor interactions across a broad kinase
panel [@C01]. DeepDTA describes binding-affinity prediction from compound and
protein sequence representations [@C02]. These sources provide benchmark and
method context; they do not validate the present models' conclusions.

## Methods

The recorded source commit was `a546a8433a6822e958f36171c4356ad6f414d623`. Fixed SMILES and protein
composition features were evaluated using a training-prevalence reference,
class-balanced logistic regression, and a class-balanced random forest. The
primary ranking metric reported here is PR-AUC because the positive class is
uncommon. Pair-random, cold-drug, and cold-target splits are reported
separately; pair-random identities overlap across partitions and therefore do
not establish cold-start generalization.

## Results

The table reports recorded PR-AUC values from the committed model report.

| Split | Random-forest PR-AUC | Prevalence PR-AUC | Difference |
| --- | ---: | ---: | ---: |
| pair random | 0.613 | 0.083 | 0.529 |
| cold drug | 0.402 | 0.094 | 0.308 |
| cold target | 0.578 | 0.095 | 0.483 |

Across the recorded conditions, the random forest had a higher PR-AUC than the
training-prevalence reference. The cold-drug result should be interpreted as a
test of unseen-compound generalization, while the cold-target result addresses
unseen-target generalization.

## Discussion

These results establish reproducible baseline behavior for this specific
dataset, representation, split design, and seed. They do not identify causal
binding mechanisms, demonstrate prospective performance, or establish utility
for drug discovery decisions. Any comparison with other studies requires a
matched task definition, data split, and evaluation protocol.

## Evidence and review limitations

The contextual claims above are traceable to the approved claim IDs `C01` and
`C02`. Crossref verification confirms bibliographic identity only; a researcher
must still assess source entailment, quality, completeness, and manuscript
suitability before release.

## References

1. Comprehensive analysis of kinase inhibitor selectivity (2011). DOI: 10.1038/nbt.1990.
2. DeepDTA: deep drug–target binding affinity prediction (2018). DOI: 10.1093/bioinformatics/bty593.
