# v0.2 Constrained Davis Manuscript Drafting Protocol

## Inputs

The generator accepts only two committed, machine-readable artifacts:

1. the real Davis baseline-model report; and
2. the Crossref-verified literature ledger.

It refuses to draft if required context claims (`C01`, `C02`) are not
finalizable, if their supporting sources are not verified, or if any of the
pair-random, cold-drug, and cold-target model outcomes is missing or invalid.

## Output

The Markdown draft contains a title, abstract, introduction, methods, results,
discussion, limitations, and references. It includes the exact recorded PR-AUC
values and cites contextual evidence with claim IDs such as `[@C01]`.

## Boundary

This is template-based constrained drafting, not autonomous authorship. It does
not retrieve full text, infer scientific mechanisms, claim clinical relevance,
or assess publication readiness. The researcher must review and revise every
section before human approval and release.
