# Phase 09A2 Property Testing Protocol

## Execution environment

- Operating system: Windows
- Python: 3.12.8
- pytest: 9.1.1
- Hypothesis: 6.168.0
- pytest-cov: 7.1.0
- coverage: 7.16.0

## Test design

The property suite uses generated, finite, non-empty, equal-length fuzzy-membership
vectors with values in [0, 1]. It also generates admissible `Parameters` values
with finite, non-negative coefficients satisfying `a == a_prime`,
`d == d_prime`, `b <= b_prime`, `c <= c_prime`, and `e <= e_prime`.
Simultaneous random coordinate permutations are generated for both vectors.

Each principal Hypothesis property runs with 200 generated examples. The tested
properties are:

- non-negativity of similarity components;
- decomposition of the components into the universe size;
- component symmetry;
- boundedness, reflexivity, symmetry, and simultaneous-permutation invariance
  of rational similarity;
- boundedness, symmetry, and reflexivity of fuzzy Jaccard;
- boundedness and symmetry of fuzzy Dice and fuzzy cosine;
- the documented zero-vector conventions;
- boundary and negative testing for empty, non-iterable, out-of-range, invalid
  parameter, and unequal-length inputs.

The component decomposition is checked as
`alpha + beta + delta + gamma` approximately equal to the universe size,
using a numerically justified absolute tolerance of `1e-10`.

## Results

- 40 tests and 4 subtests passed.
- Statement coverage: 100%.
- Branch coverage: 100%.
- No counterexample was found in the tested domain.

These results are computational evidence about the implementation and generated
domain. Numerical/property testing does not constitute theorem proof.
Identity of indiscernibles was not claimed. Monotonicity remains unresolved
unless separately proven with its assumptions.

The final mathematical interpretation remains subject to human approval; human
approval status is pending.
