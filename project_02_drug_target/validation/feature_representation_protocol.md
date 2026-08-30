# Davis Feature-Representation Protocol

## Decision

The first reproducible baseline uses transparent, dependency-free
representations. No RDKit fingerprints, pretrained embeddings, external
biological annotations, or deep-learning representations are used in this
phase.

## Drug Representation

Canonical SMILES are transformed into fixed, interpretable counts and
fractions:

- SMILES length and lightweight atom count;
- carbon, nitrogen, oxygen, sulfur, phosphorus, halogen, and other-atom
  fractions;
- aromatic-atom fraction;
- ring-marker, branch, double-bond, triple-bond, and charge-marker counts.

These are heuristic string-derived descriptors, not a complete chemical graph
representation or evidence of molecular mechanism.

## Target Representation

Protein sequences are transformed into:

- sequence length;
- fractions for 19 canonical amino acids;
- an explicit unknown-residue (`X`) fraction.

Tyrosine (Y) is omitted as a fixed reference residue so composition features
do not create an exact redundant sum-to-one relationship.

## Unknown-Residue Decision

TESK1 contains two `X` residues in a 244-residue sequence
(unknown-residue fraction 0.0081967). TESK1 and its 68 interaction pairs are
retained. `X` is not imputed as a named amino acid; it is represented only by
`target_unknown_residue_fraction`.

Any non-canonical target symbol other than `X` causes feature construction to
fail.

## Leakage and Feature-Quality Safeguards

- Features are deterministic functions of SMILES or protein sequence only.
- Drug/target IDs, matrix indices, affinity values, pKd, and binary labels are
  excluded from the model feature matrix.
- Every interaction must join to exactly one drug and one target feature row.
- Missing or non-finite feature values cause the pipeline to fail.
- Any later learned vectorizer, scaler, selector, or resampler must fit only
  on the training partition.
- Join identifiers are normalized to trimmed strings before merging to prevent
  CSV type inference from changing Davis mapping keys.

## Interpretation Boundary

Feature importance may describe associations within this benchmark. It does not
establish pharmacological mechanism, biological causality, or clinical utility.