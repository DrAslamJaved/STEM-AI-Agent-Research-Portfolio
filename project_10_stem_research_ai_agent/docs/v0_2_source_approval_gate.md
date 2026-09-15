# v0.2 Davis Source Approval Gate

The canonical DeepDTA Davis `Y` file is a Python pickle. A pickle can execute
code while being deserialized, so it must not be loaded from an unapproved
location or before integrity verification.

## Required sequence

1. Acquire the three raw files only from the pinned repository and commit.
2. Calculate SHA-256 hashes without opening `Y` in Python.
3. Record the values in `config/v0_2_davis_manifest.json` and record the
   researcher's review of source access/licence terms.
4. Run the data-contract demonstration. It verifies every expected hash before
   calling the matrix loader.
5. Commit the completed manifest and report, but never the raw source files.

## Runtime dependency

The canonical `Y` pickle contains a NumPy array and may contain legacy string
representations. NumPy is therefore a required runtime dependency for real
Davis loading, and deserialization uses the `latin1` compatibility encoding
specified by the pinned DeepDTA source.
