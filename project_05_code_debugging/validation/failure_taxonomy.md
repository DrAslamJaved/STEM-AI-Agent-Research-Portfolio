# Failure Taxonomy

| Failure Type | Example | Detection Method | Resolution |
|---|---|---|---|
| Mathematical | Incorrect formula | Mathematical test | Correct specification |
| Implementation | Unnecessary rounding | Unit test | Modify implementation |
| Numerical | Floating-point discrepancy | Numerical tolerance test | Assess magnitude/source |
| Boundary | Empty/single input | Boundary tests | Explicit handling |
| Specification mismatch | Code adds behavior not requested | Specification review | Remove unintended behavior |
| Regression | Previous behavior breaks after fix | Full test suite | Revise implementation |

