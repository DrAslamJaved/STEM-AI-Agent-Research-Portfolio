# Phase 09 Property Testing Agent Trace

## Original agent prompt

```text
Implement Phase 09A1 in:

G:\Research\STEM\Micro1_STEM_Agent_Portfolio\
project_01_fuzzy_similarity

Create only:
tests/test_properties.py

Read core.py and test_core.py before writing tests.

Use pytest and genuine Hypothesis strategies with at least 200 generated
examples per principal property. Generate:

- finite, non-empty, equal-length membership vectors in [0,1];
- admissible Parameters satisfying:
  a == a_prime,
  d == d_prime,
  b <= b_prime,
  c <= c_prime,
  e <= e_prime,
  with all coefficients finite and non-negative;
- simultaneous random coordinate permutations.

Test:

1. Similarity-component non-negativity.
2. alpha + beta + delta + gamma approximately equals universe size.
3. Component symmetry.
4. Rational-similarity boundedness.
5. Reflexivity.
6. Symmetry.
7. Simultaneous-permutation invariance.
8. Jaccard boundedness, symmetry and reflexivity.
9. Dice and cosine boundedness and symmetry.
10. Their documented zero-vector conventions.
11. Rejection of empty and non-iterable membership inputs.
12. Rejection of invalid parameter types.
13. Unequal-length rejection by similarity_components, fuzzy_dice,
    fuzzy_cosine and rational_similarity.

Use numerically justified pytest.approx tolerances.

Add targeted tests for currently uncovered core.py lines:
21-22, 25, 63, 106, 128, 168 and 184.

Do not assert identity of indiscernibles.
Do not assert monotonicity without an approved theorem and assumptions.
Do not modify core.py to make a generated test pass.
If Hypothesis finds a counterexample, preserve and report its minimized
example instead of suppressing it.

Do not stage, commit or push.
```

## Execution record

1. The initial prompt was accidentally saved as Python code in the test artifact.
2. Pytest correctly rejected that artifact with `SyntaxError` during collection.
3. Human intervention identified and rejected the invalid artifact.
4. Valid Hypothesis tests were then supplied.
5. Collection increased from 17 to 40 tests.
6. Coverage increased from 88.80% to 100%.
7. No implementation change was made merely to satisfy tests.
8. Final mathematical interpretation remains subject to human approval.

## Phase 09A2 result

The property-testing protocol records Windows/Python 3.12.8, the pinned test
tool versions, generated domains, 200 examples per principal property, the
tested properties and boundaries, and the result of 40 tests with 4 subtests
passed and 100% statement and branch coverage. No counterexample was found in
the tested domain. Identity of indiscernibles was not claimed, and monotonicity
remains unresolved unless separately proven. Numerical/property testing does
not constitute theorem proof. Human approval status remains pending.