# Phase 10R Protocol Clarification 01

- **Date:** 2026-09-18
- **Reason:** The proposed protocol did not specify the raw Davis file formats
  and matrix orientation, leaving an implementation degree of freedom.
- **Clarification:** `ligands_can.txt` and `proteins.txt` are UTF-8 JSON
  mappings; `Y` is a trusted legacy pickle loaded with `encoding="latin1"`.
  Its required orientation is 68 ligand rows by 442 target columns.
- **No design change:** Source, label, representation, split seed, methods,
  inference, and interpretation boundaries are unchanged. No input was read
  and no result was generated during this clarification.
