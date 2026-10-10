# Phase 00 — Foundation and Protocol Definition

## Human decision

Dr. Aslam Javed approved Phase 00 as the foundation of Project 1: the research
protocol, the claim-classification protocol, and the project's governance
model.

## Scope of approval

This approval establishes *how* the project will work. It authorizes the
governance framework only. It does **not**:

- define, propose, or approve any candidate similarity formula;
- establish any theorem or mathematical result;
- assert novelty;
- report or authorize any numerical or experimental result;
- classify any literature claim as established.

All subsequent mathematical, computational, and empirical work is gated behind
the later phases defined by this protocol.

## Approved foundation

- **Research question.** Whether a human-supervised AI-agent workflow can
  support rigorous formulation, analysis, implementation, testing, and
  empirical evaluation of a cardinality-based fuzzy similarity measure
  *without* confusing generated conjectures or numerical observations with
  proven results.
- **Claim taxonomy.** Every substantive claim must carry exactly one status
  from the ledger: `ESTABLISHED`, `AGENT_PROPOSED`, `HUMAN_APPROVED`,
  `PROOF_DRAFTED`, `PROVED_VERIFIED`, `EXPERIMENTALLY_SUPPORTED`, `REFUTED`, or
  `INCONCLUSIVE`.
- **The firewall.** `EXPERIMENTALLY_SUPPORTED` may never be promoted to
  `PROVED_VERIFIED`. A numerical test, symbolic calculation, or randomized
  simulation can refute or support a conjecture but cannot by itself establish
  a theorem.
- **Evidence and provenance policy.** Reproducible literature search, verified
  bibliographic metadata, recorded seeds, configuration and dependency
  versions, executed commands, and input hashes are required for all future
  experimental work.
- **Human-intervention requirement.** At least five meaningful human
  interventions must be documented over the life of the project (for example:
  challenging an unsupported theorem claim, identifying a hidden assumption,
  rejecting an invalid proof, requiring a counterexample, or correcting an
  implementation error).

## Claim status

`HUMAN_APPROVED` governance foundation for Project 1.

## Human intervention

At foundation, the agent was required to:

1. state the research question in terms of epistemic discipline, not in terms
   of producing a "novel" measure;
2. commit to the claim taxonomy and the
   `EXPERIMENTALLY_SUPPORTED ⇏ PROVED_VERIFIED` firewall before any work began;
3. defer every candidate formula, proof, and numerical result to its own gated
   phase.

## Evidence

- `docs/research_protocol.md`
- `validation/claim_classification_protocol.md`
- user approval: `Approve Phase 00`

## Next gate

Phase 01 — fix the mathematical contract: domain, operations, cardinality
convention, boundary rules, and the axiom vocabulary, reviewed and accepted as
the project's operational mathematical contract.
