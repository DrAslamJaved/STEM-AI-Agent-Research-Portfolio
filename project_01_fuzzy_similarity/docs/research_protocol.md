# Research Protocol

## Project title

AI-Assisted Development and Validation of a Novel Fuzzy Similarity Measure

## Status

Phase 00 — foundation and protocol definition.

No candidate similarity measure, theorem, numerical result, or application result is claimed in this document. All future experimental work is **TO BE EXECUTED/VERIFIED**.

## Research question

Can a human-supervised AI-agent workflow support the rigorous formulation, mathematical analysis, implementation, testing, and empirical evaluation of a cardinality-based fuzzy similarity measure without confusing generated conjectures or numerical observations with proven mathematical results?

## Objectives

1. Identify and verify established fuzzy similarity theory and relevant cardinality-based measures.
2. Define a clearly labelled agent-generated candidate measure only after literature review.
3. Obtain human approval for any formulation selected for mathematical analysis.
4. Establish which properties are formally proven, conditionally valid, disproven, or empirically observed.
5. Implement the approved definition and classical baselines in Python.
6. validate the implementation using unit, boundary, negative, randomized property-based, and independent mathematical tests.
7. Compare the proposed measure with fuzzy Jaccard, Dice, Cosine, and suitable established cardinality-based baselines.
8. Evaluate its behaviour in controlled clustering and classification experiments.
9. Document agent contributions, errors, human interventions, and reproducibility evidence.

## Mathematical scope

The initial scope is finite fuzzy sets on a non-empty universe:

\[
X = \{x_1, x_2, \ldots, x_n\}, \qquad n \geq 1.
\]

A fuzzy set \(A\) is represented by its membership function:

\[
\mu_A : X \rightarrow [0,1].
\]

Working cardinality notation will use the sigma-count:

\[
|A| = \sum_{i=1}^{n} \mu_A(x_i).
\]

Any alternative cardinality, intersection, union, complement, normalization, or parameter convention must be defined explicitly and justified with verified literature.

## Required similarity properties

For an approved similarity function \(S(A,B)\), the project will examine, where justified:

\[
0 \leq S(A,B) \leq 1,
\]

\[
S(A,A)=1,
\]

\[
S(A,B)=S(B,A).
\]

Identity, self-complementarity, monotonicity, continuity, and other properties will be stated precisely before analysis. Reflexivity does not by itself establish identity:

\[
S(A,A)=1 \not\Rightarrow [S(A,B)=1 \Leftrightarrow A=B].
\]

## Evidence and claim policy

Every substantive claim must be classified as one of:

- **Established theory:** supported by verified primary literature.
- **Agent-generated proposal:** a candidate produced by an AI agent.
- **Human-approved formulation:** accepted after mathematical review.
- **Formally proven result:** supported by a complete human-reviewed proof.
- **Disproven claim:** refuted by a valid counterexample.
- **Experimentally supported observation:** numerical evidence only.

Numerical experiments, randomized tests, and symbolic checks may support or refute conjectures, but cannot alone prove a theorem.

## Methodology

1. Conduct and document a reproducible literature search.
2. Verify bibliographic metadata and mathematical claims against primary sources.
3. Define candidate measures and state all domain assumptions.
4. Request agent-generated proof attempts and inspect each logical step.
5. Record human interventions, including rejected proofs, hidden assumptions, and counterexamples.
6. Implement approved measures and classical baselines.
7. Run unit, boundary, negative, property-based, and independent-verification tests.
8. Conduct controlled numerical experiments and application studies.
9. Produce a final scientific interpretation and agent-evaluation report.

## Human-intervention requirements

The project must document at least five meaningful interventions, including:

1. challenging an unsupported theorem claim;
2. identifying a hidden assumption or undefined boundary case;
3. rejecting an invalid proof;
4. requiring a counterexample for an overgeneralized claim;
5. identifying and correcting an implementation error;
6. restricting or revising an unsupported conclusion.

## Reproducibility requirements

Each future experiment must record:

- source data and provenance;
- random seed;
- configuration and parameter values;
- Python and dependency versions;
- executed command;
- input hashes where applicable;
- generated outputs;
- test results, coverage, failures, and corrections.

## Completion criteria

The project is complete only when it includes verified literature, a human-approved formulation, formal proofs or counterexamples, tested Python code, reproducible experiments, application evidence, documented interventions, and an agent-evaluation report.
