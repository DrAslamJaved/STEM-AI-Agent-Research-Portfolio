# v0.2 Davis Release-Gate Protocol

## Inputs and consistency checks

The gate receives the committed model report, verified literature ledger,
constrained Markdown draft, and draft trace. It rejects mismatched dataset
provenance, missing finalizable claims, unsupported citation IDs, omitted split
conditions, altered result values, and invalid reviewer input.

## Human decision

The command requires a named reviewer and a specific review reason. Those inputs
create append-only `approve` and `lock` events in the release report. The script
cannot supply those values autonomously.

## Reproducibility evidence

The researcher must run tests, source compilation, and the applicable Git diff
check before invoking the gate, then declare each completed check with its
corresponding command-line flag. The gate records this evidence; it does not
silently execute or fabricate it.

## Boundary

`release_ready: true` means the committed v0.2 artifacts are internally
consistent, reviewer approved, locked, and accompanied by the stated checks. It
does not mean that the draft is peer reviewed, clinically validated, complete,
or ready for journal submission.
