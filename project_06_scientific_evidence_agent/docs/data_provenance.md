# SciFact data provenance

## Canonical source

Acquire the release archive from:

```text
https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
```

The acquisition command writes `validation/scifact_acquisition.json`, recording
the exact source URL, local archive path, byte size, SHA-256 digest, extraction
root, timestamp, and whether the archive was downloaded during that run.

## Raw-data policy

The archive and extracted SciFact JSONL files live under `data/raw/scifact/` and
are deliberately excluded from Git. The provenance manifest and validation
report are retained in Git because they make the acquisition auditable without
redistributing the data.

## Reproduction command

```powershell
& .\.venv\Scripts\python.exe -m evidence_agent acquire-data
```

Never edit the raw JSONL files. Derived data belong in `data/interim/` or
`data/processed/`, with a documented transformation step.
