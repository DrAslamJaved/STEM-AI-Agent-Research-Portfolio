# v0.2 Davis Data and Provenance Contract

## Approved source layout

Use an exact local copy of the canonical DeepDTA-style Davis source directory:

```
data/raw/davis/
  ligands_can.txt
  proteins.txt
  Y
```

`ligands_can.txt` and `proteins.txt` must be non-empty JSON objects. In the
pinned DeepDTA source, `Y` is a binary Python-pickle matrix of strictly
positive Kd values in nanomolar units. It must be loaded only after its
SHA-256 value matches the researcher-approved manifest; untrusted pickle files
must never be loaded.
The matrix row order must follow the ligand JSON object order; its column order
must follow the protein JSON object order.

## Dataset identity and integrity

- Record the upstream repository URL and immutable commit in
  `config/v0_2_davis_manifest.json`.
- Record acquisition time, access/licence review, and SHA-256 for all three
  source files. The validation command must reject placeholders or a mismatch
  before the binary `Y` file is unpickled.
- Do not overwrite raw data. Corrected or alternative sources require a new
  manifest and a new release run.
- The expected canonical shape is 68 compounds by 442 targets (30,056 pairs).
  A different shape is an explicit exception requiring researcher approval.

## Derived outcomes

The continuous response is `pKd = 9 - log10(Kd_nM)`.  The default binary
research label is `pKd >= 7.0`, equivalent to `Kd <= 100 nM`. This threshold is
a modelling decision, not a biological truth; sensitivity analyses must be
reported before claims depend on it.

## Required exclusions and checks

- Reject a source whose hash is not approved, and reject zero, negative,
  non-finite, or non-rectangular affinity data.
- Preserve every observed pair; do not create negatives by random pairing.
- Retain raw Kd and pKd simultaneously in analytical outputs.
- Never use a held-out drug or target for fitting in cold-start evaluation.
- Report class prevalence and split overlap for every evaluation condition.
