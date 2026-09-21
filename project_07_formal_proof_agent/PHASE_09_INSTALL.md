# Phase 09 — Real Study Completion

Phase 09 turns the completed Project 07 MVP into a genuine controlled study.
It must be installed only in the canonical repository:

`G:\Research\STEM\Micro1_STEM_Agent_Portfolio\project_07_formal_proof_agent`

It never copies source files from the old sibling worktree.  The old Phase 02
worktree is retained unchanged until the canonical miniF2F runtime has passed
its own build and pre-flight checks.

## 1. Install and verify the implementation

From the canonical repository root on the dedicated feature branch, extract
the Phase 09 bundle, then run:

```powershell
$Project = 'project_07_formal_proof_agent'

python -m pip install -r "$Project/requirements-phase05.txt"
python -m pytest -q "$Project/tests"
python "$Project/scripts/run_phase_08_synthetic_release.py" --project-root $Project
python "$Project/scripts/validate_phase_08_evaluation_release.py"
python "$Project/scripts/validate_phase_09_live_study.py"
python -m compileall -q "$Project/src" "$Project/scripts"
git diff --check
git status --short
```

The Phase 08 fixture must remain synthetic and `release_ready: false`.
Phase 07 source provenance uses canonical LF-normalized text hashing, so the
same formalization digest is valid in both Windows CRLF and LF checkouts.

## 2. Provision the canonical runtime

Run the PowerShell script from PowerShell 7.  It creates or verifies the
ignored checkout at `benchmarks\minif2f\upstream` within the canonical Project
07 directory, pins it to miniF2F commit
`f0dcc8b59e630fba00ba9569ca6714700e0a8801`, and builds it with Lean 3.42.1.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File `
  "$Project\scripts\provision_phase_09_canonical_runtime.ps1"
```

Do not delete the legacy worktree in this stage.

## 3. Run a small development pilot first

The live runner makes paid model calls only after a plan has been written.  It
expects `OPENAI_API_KEY` in the process environment; never place a key in a
repository file, task manifest, or command history.

Use current, model-specific token prices from the provider's pricing page when
creating the plan.  The following uses placeholders intentionally:

```powershell
$Project = 'project_07_formal_proof_agent'
$Upstream = "$Project\benchmarks\minif2f\upstream"
$Run = 'p07_dev_001'
$RunDir = "$Project\runs\phase_09\$Run"
$Elan = Join-Path $env:USERPROFILE '.elan\bin\elan.exe'

python "$Project/scripts/prepare_phase_09_tasks.py" `
  --project-root $Project --mini-f2f-root $Upstream --split valid --limit 10 `
  --output "$RunDir\tasks.jsonl"

python "$Project/scripts/preregister_phase_09_study.py" `
  --tasks "$RunDir\tasks.jsonl" --output "$RunDir\plan.json" `
  --run-id $Run --model '<REGISTERED_MODEL_ID>' --samples-per-theorem 1 `
  --base-seed 20260921 --input-usd-per-million-tokens <CURRENT_INPUT_PRICE> `
  --output-usd-per-million-tokens <CURRENT_OUTPUT_PRICE> `
  --maximum-estimated-cost-usd <APPROVED_CAP>

# Set this only in the current shell, from a secure source.
$env:OPENAI_API_KEY = '<your-key>'

python "$Project/scripts/run_phase_09_live_study.py" `
  --tasks "$RunDir\tasks.jsonl" --plan "$RunDir\plan.json" `
  --elan $Elan --lean-cwd $Upstream --output-dir "$RunDir\evidence"
```

The development report is not a final result and never uses the test split.

## 4. Locked final evaluation

After reviewing the development pilot and fixing its model, pricing, seed,
sample count, and cost cap, create a new immutable test task manifest and plan
with `--final-evaluation`.  The final run additionally requires
`--confirm-final-evaluation`.

Do not open or generate against the test split before that point.  The runner
creates a Phase 08-compatible evidence bundle but deliberately leaves the
human-review gate unapproved.  A human must inspect the raw attempts,
provenance hashes, static rejections, and limitations before any release claim.

After that review—not before—record the decision without manually editing
hashes or evidence files:

```powershell
python "$Project/scripts/record_phase_09_human_review.py" `
  --run-dir "$RunDir\evidence" --reviewer '<NAME>' `
  --approval-note '<what was reviewed and why the result is acceptable>' --approve
```
