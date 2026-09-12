# Mathematical Foundation and Axiom Contract

## Status and scope

**Phase 01 — pre-formulation mathematical specification**

This document fixes the working domain, notation, operations, boundary cases, and axiom vocabulary for Project 1.

It does not introduce or approve a novel similarity formula. It does not establish any theorem. Literature verification, candidate formulation, proof, counterexample construction, and numerical validation remain **TO BE EXECUTED/VERIFIED**.

## Universe and fuzzy sets

Let

\[
X = \{x_1, x_2, \ldots, x_n\}, \qquad n \geq 1,
\]

be a finite non-empty universe.

A fuzzy set \(A\) on \(X\) is represented by a membership function

\[
\mu_A : X \rightarrow [0,1].
\]

Equivalently,

\[
A = \big(\mu_A(x_1), \mu_A(x_2), \ldots, \mu_A(x_n)\big) \in [0,1]^n.
\]

The fuzzy-set family on \(X\) is denoted by \(\mathcal{F}(X)\).

## Working operations

Unless a later, human-approved formulation states otherwise, the project uses the standard pointwise operations:

\[
\mu_{A \cap B}(x_i) = \min\{\mu_A(x_i), \mu_B(x_i)\},
\]

\[
\mu_{A \cup B}(x_i) = \max\{\mu_A(x_i), \mu_B(x_i)\},
\]

and standard complement:

\[
\mu_{A^c}(x_i) = 1-\mu_A(x_i).
\]

The coordinatewise partial order is

\[
A \preceq B
\quad \Longleftrightarrow \quad
\mu_A(x_i) \leq \mu_B(x_i)
\text{ for every } i.
\]

## Cardinality convention

The initial cardinality convention is the sigma-count:

\[
|A| = \sum_{i=1}^{n} \mu_A(x_i).
\]

Therefore,

\[
0 \leq |A| \leq n.
\]

Any alternative fuzzy cardinality requires a separate definition, literature verification, and explicit approval before use.

## Similarity-function domain

A candidate similarity function will eventually be specified as either

\[
S : \mathcal{F}(X) \times \mathcal{F}(X) \rightarrow [0,1],
\]

or as a function on an explicitly stated restricted domain.

A definition containing a denominator must state its value or admissibility rule whenever that denominator is zero. No silent convention for empty or zero-cardinality fuzzy sets is permitted.

## Required properties

For each human-approved candidate \(S(A,B)\), the following properties will be examined separately.

### Boundedness

\[
0 \leq S(A,B) \leq 1.
\]

### Reflexivity

\[
S(A,A)=1.
\]

### Symmetry

\[
S(A,B)=S(B,A).
\]

### Identity

The strong identity condition is:

\[
S(A,B)=1 \Longrightarrow A=B.
\]

Identity is stronger than reflexivity. A proof of \(S(A,A)=1\) alone does not establish identity.

### Monotonicity

No generic monotonicity claim is assumed.

Any monotonicity proposition must state:

1. the ordering relation;
2. which fuzzy sets are being varied;
3. the exact inequality expected of \(S\);
4. all parameter and denominator assumptions.

### Additional properties

Self-complementarity, continuity, invariance, triangle-type inequalities, or relationships to Jaccard, Dice, Cosine, or Tversky measures will be examined only when they are meaningful for the selected definition.

## Boundary and implementation contract

The Python implementation must reject:

- unequal-length membership vectors;
- values outside \([0,1]\);
- NaN or infinite values;
- invalid parameters.

All-zero fuzzy sets are valid inputs unless an approved measure restricts its domain. Any resulting undefined expression must be handled explicitly and tested.

## Proof and experiment separation

For every property:

1. state the exact proposition and assumptions;
2. attempt an analytic proof;
3. conduct adversarial review;
4. search for analytic counterexamples;
5. perform exhaustive small-domain or numerical checks where useful;
6. obtain independent mathematical verification;
7. assign the appropriate claim status.

Numerical experiments can produce `EXPERIMENTALLY_SUPPORTED` evidence only. They cannot produce `PROVED_VERIFIED` status.

## Human review gates

Human approval is required before:

- accepting a candidate formula;
- declaring a property proved;
- adopting a denominator-zero convention;
- asserting novelty;
- generalizing a result beyond its stated assumptions.

## Phase 01 completion condition

Phase 01 is complete when the domain, operations, cardinality convention, boundary rules, and axiom definitions are reviewed and accepted as the project’s mathematical contract.
