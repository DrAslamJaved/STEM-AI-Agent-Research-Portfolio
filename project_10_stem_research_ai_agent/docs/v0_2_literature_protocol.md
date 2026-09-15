# v0.2 Literature and Evidence Protocol

## Purpose

This increment turns a small, researcher-declared list of DOI sources into a
claim-to-evidence ledger for the real Davis DTI workflow. It does not generate
claims from search snippets or retrieve article full text.

## Verification boundary

For every planned source, the workflow queries Crossref by DOI and verifies the
returned DOI, a substantial title match, and (when declared) the publication
year. A source is marked `VERIFIED` only after those checks pass. This confirms
bibliographic identity, not the truth of every scientific statement in the
article.

## Claim gate

A claim is finalizable only when all three conditions hold:

1. the researcher has explicitly set its status to `HUMAN_APPROVED` or
   `EXPERIMENTALLY_SUPPORTED`;
2. it has a `supports` link with a rationale; and
3. the linked source is Crossref verified.

Human review remains responsible for source suitability, claim entailment,
interpretation, and any manuscript wording.

## Reproduction

Run `run_v0_2_literature_demo.py` with an optional `--mailto` address. The
resulting ledger is a review artifact and should be committed only after the
researcher checks the resolved source metadata and claims.
