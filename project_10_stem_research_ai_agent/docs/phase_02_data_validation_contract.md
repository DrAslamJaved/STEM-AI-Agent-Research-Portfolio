# Phase 2 Data Validation Contract

DTI records require non-empty drug_id, target_id, and binary label fields.
Duplicate pairs are retained and reported rather than silently removed. A
warning is raised when cold-start evaluation is impossible.

Fuzzy membership vectors must be finite, non-empty, numeric, and contained in
[0, 1]. Sigma-count cardinality is recorded as a descriptive metric only.

Every acquired input file receives a SHA-256, byte count, filename, and path
before use. Validation never modifies raw input.
