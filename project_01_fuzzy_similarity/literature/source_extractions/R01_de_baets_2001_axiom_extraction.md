# R01 — Primary-source axiom extraction

## Source

De Baets, B., De Meyer, H., and Naessens, H. (2001). *A class of rational cardinality-based similarity measures*. Journal of Computational and Applied Mathematics, 132(1), 51–69. DOI: 10.1016/S0377-0427(00)00596-3.

## Extraction status

`ESTABLISHED` for the page-specific literature statements below, based on direct inspection of the supplied primary article.

`NOT A PROJECT-1 CANDIDATE`: this paper does not by itself approve a new fuzzy-set formulation for Project 1.

## Scope and terminology

| Item | Source-supported statement | Location |
|---|---|---|
| Domain | The study restricts itself to crisp subsets of a finite universe \(X\), hence \(A,B\in\mathcal P(X)\). | p. 53 |
| Similarity terminology | A similarity measure is a locally reflexive and symmetric binary fuzzy relation on \(\mathcal P(X)\). | p. 53 |
| Relevance limit | The paper compares binary vectors/crisp sets. It does not, on this page, establish the same results for membership-vector fuzzy sets \([0,1]^n\). | p. 53; scope inference |

## Rational cardinality-based family

Equation (1) gives a rational family whose numerator and denominator are linear combinations of four cardinality terms:

\[
\alpha_{A,B}=\min\{\#(A\setminus B),\#(B\setminus A)\},\quad
\omega_{A,B}=\max\{\#(A\setminus B),\#(B\setminus A)\},
\]

\[
\delta_{A,B}=\#(A\cap B),\quad
\nu_{A,B}=\#(A\cup B)^c,
\]

with binary coefficients. The paper states that the reflexive members of this class are obtained by setting \(c'=c\) and \(d'=d\). When the resulting expression is \(0/0\), it assigns similarity value 1 to maintain reflexivity.

**Important Project 1 treatment:** this is established literature, not an agent-generated proposal and not a human-approved candidate measure for fuzzy sets.

## Extracted property framework

| Category | Exact object examined in the paper | Project 1 treatment |
|---|---|---|
| Local reflexivity and symmetry | Baseline definition of a similarity measure on \(\mathcal P(X)\). | Compare with, but do not automatically transfer to, fuzzy sets. |
| Reflexivity | The paper distinguishes its reflexive subclass \(R\). | Required core property for any future approved candidate. |
| Boundary conditions \(B_1,B_2,B_3\) | Similarity to \(\varnothing\), similarity to \(X\), and \(S(A,A^c)=0\), respectively. | Definition-specific; inspect only if meaningful for the selected fuzzy formulation. |
| C-type monotonicity | Two-set monotonicity, including \(S(A\cup B,A\cap B)\) compared with \(S(A,B)\). | Optional; state order direction and domain before analysis. |
| M-type monotonicity | Three-set movement conditions between \(A\cap B\), \(A\triangle B\), and \((A\cup B)^c\). | Optional and crisp-set-specific until separately reformulated. |
| D-type monotonicity | Stronger three-set conditions derived from M-type behaviour. | Optional and crisp-set-specific until separately reformulated. |
| F-type monotonicity | Four-set monotonicity. | Optional; no generic claim for Project 1. |
| \(T\)-transitivity | Transitivity assessed for selected measures and triangular norms. | Optional; not a required similarity axiom for Project 1. |
| Self-complementarity | Defined through \(S^c(A,B)=S(A^c,B^c)\). | Optional; test only if a complement-compatible candidate is selected. |

## Page-level evidence map

| Pages | Verified content |
|---|---|
| 53 | Scope, terminology, rational family, component definitions, coefficient restrictions, reflexive subclass, explicit \(0/0\mapsto1\) convention, complement/self-complementarity. |
| 55–56 | Boundary conditions \(B_1,B_2,B_3\). |
| 56–60 | C-, M-, D-, and F-type monotonicity framework. |
| 61–68 | \(T\)-transitivity analysis and classification for selected members of the class. |
| 68 | Conclusion and classification summary. |
| 69 | References. |

## Project 1 implications and review gate

1. The uploaded paper is a verified source for a crisp-set cardinality-based family and its property vocabulary.
2. It does not prove that the same family, coefficients, zero-denominator rule, or axioms hold for finite fuzzy sets using sigma-count cardinality.
3. Any adaptation to fuzzy memberships must be recorded first as `AGENT_PROPOSED`, then receive human approval before proof attempts, code, or experiments.
4. Numerical checks can support only `EXPERIMENTALLY_SUPPORTED` claims; they cannot promote a property to `PROVED_VERIFIED`.

## Next verification target

Extract the paper's exact statement and proof conditions for the particular property family most relevant to the eventual human-approved fuzzy formulation. No candidate formula is selected at this stage.
