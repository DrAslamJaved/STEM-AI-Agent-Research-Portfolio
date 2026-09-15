# Project 10 v0.2 — Real DTI-to-Manuscript Research Workflow

## Objective

Turn the v0.1 control prototype into one transparent, reproducible case study:
an approved real Davis kinase dataset is converted into a leak-aware DTI
evaluation, linked to verified scholarly evidence, and used to produce a
researcher-reviewed manuscript package.  The system remains an assistant; it
does not autonomously establish novelty, authorship, factual correctness, or
publication readiness.

## v0.2 acceptance criteria

1. A pinned, locally stored Davis source is validated against a recorded
   manifest and SHA-256 checksums.
2. Every long-form DTI record retains drug, target, raw Kd (nM), derived pKd,
   label definition, and source version.
3. Pair-random, cold-drug, and cold-target splits are generated deterministically
   and report identity overlap explicitly.
4. A simple reproducible baseline and at least one stronger model are evaluated
   with ROC-AUC, PR-AUC, F1, calibration, and uncertainty intervals where valid.
5. External literature records have their DOI and bibliographic metadata checked
   before their claims are eligible for drafting.
6. A constrained manuscript package contains methods, results, limitations,
   references, figure/table provenance, and the human review record.
7. A release report records the exact source hashes, environment, commands,
   test result, evaluation artifacts, audit result, and human approval status.

## Delivery increments

| Increment | Outcome | Gate |
|---|---|---|
| 1 | Davis acquisition and data contract | source hashes approved |
| 2 | Leakage-aware splits and modelling | held-out evaluation reproduced |
| 3 | Verified literature ledger | DOI/metadata checks pass |
| 4 | Manuscript package | every reported result traces to an artifact |
| 5 | Human review and tagged release | approval, lock, and reproducibility pass |

## Non-goals for v0.2

- Autonomous article submission or autonomous authorship.
- Claims of plagiarism-free output or exhaustive literature coverage.
- Use of unapproved credentials or subscription-only sources.
- Inflated performance claims based solely on random pair splits.
