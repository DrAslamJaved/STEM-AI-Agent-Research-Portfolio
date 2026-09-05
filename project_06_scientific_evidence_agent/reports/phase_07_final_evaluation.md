# Phase 07 — Final evaluation (held-out development split)

**This is a held-out development evaluation, not an independent test.** It
reuses `claims_dev.jsonl` (300 claims), the same split already used to report
Phase 06's numbers. No new labelled data was introduced in Phase 07.

Command: `evidence-agent evaluate --config configs/final.yaml`
Result JSON: `results/final_evaluation_dev.json`
Result JSON SHA-256: `b8f997142a49c3cf497ae48727f5378d91288c237562c5cce1a1b861060e03fd`
Raw runtime trace (local, ignored): `artifacts/final_evaluation_dev_trace.json`
Trace SHA-256: `3a86c9fd96d99fe64c4c4e2b0fbdd5a2155ab02b8515cfd59d859e1cf821b5ae`

The result JSON's own hash is not run-to-run stable: it embeds
`runtime_timing` (wall-clock seconds), which varies by run even though every
metric, policy, and artifact hash inside it is identical across runs. The
trace SHA-256 above is fully reproducible run-to-run given the same frozen
inputs declared in `configs/final.yaml`.

All paths recorded inside the result JSON (artifact provenance, the frozen
trace, `config_path`, and `output.*`) are project-relative POSIX paths
(e.g. `data/raw/...`, `artifacts/...`), derived from `configs/final.yaml`'s
own location rather than the process working directory -- never an absolute,
machine- or OS-specific path.

## What was compared

Both policies below were applied to the exact same frozen, zero-threshold
runtime trace over `claims_dev.jsonl`; gold labels were loaded only after that
trace was written to disk.

| Policy | Assertion threshold | Sentence threshold | Max sentences/citation |
| --- | --- | --- | --- |
| Phase 05 (fixed baseline) | 0.45 | 0.50 | 2 |
| Phase 06 selected | 0.65 | 0.60 | 2 |

## Result: a coverage-for-quality trade-off, not a clean improvement

| Metric | Phase 05 | Selected | Delta | 95% paired-bootstrap CI |
| --- | --- | --- | --- | --- |
| Coverage | 0.8567 | 0.4833 | **-0.3733** | [-0.4300, -0.3200] |
| Unsupported-assertion rate | 0.9455 | 0.9310 | -0.0145 | [-0.0405, +0.0089] |
| Faithfulness | 0.0545 | 0.0690 | +0.0145 | [-0.0089, +0.0405] |
| Claim macro-F1 | 0.3957 | 0.4128 | +0.0171 | [-0.0358, +0.0718] |
| Citation-correctness F1 | 0.0067 | 0.0083 | +0.0016 | [-0.0092, +0.0138] |
| Evidence-sentence F1 | 0.0480 | 0.0428 | -0.0052 | [-0.0222, +0.0106] |

(2,000 paired bootstrap resamples, seed `20260904`, claim-level resampling
with replacement, each resample's metrics recomputed from pooled counts.)

**The only effect this evaluation can actually distinguish from resampling
noise is the coverage drop.** The selected policy abstains on roughly
37 percentage points more of the development claims than the Phase 05
policy (coverage falls from 0.857 to 0.483), and that interval is entirely
negative -- it does not cross zero.

Every metric the audit policy was meant to improve -- unsupported-assertion
rate, faithfulness, citation-correctness F1, claim macro-F1, evidence-sentence
F1 -- shows a small point-estimate shift in the intended direction (except
evidence-sentence F1, which moved slightly the wrong way), but **each of
their 95% confidence intervals contains zero**. At 300 development claims and
a citation-correctness F1 near the floor for both policies (0.007-0.008), we
cannot claim the selected policy is a more reliable evidence-verification
agent than the fixed Phase 05 policy. What we can claim is that it abstains
much more often, in exchange for a quality change too small to separate from
noise at this sample size.

This matches, and quantifies with an interval, what Phase 06's own report
already flagged qualitatively: "this small risk reduction is retained as
evidence of an abstention trade-off, not as proof of a reliable
evidence-verification agent." Phase 07 adds the confidence interval that
shows the risk reduction is not even distinguishable from zero at this claim
count, while the abstention cost is large and clear.

## Reproducing this result

```powershell
& .\.venv\Scripts\python.exe -m evidence_agent build-index
& .\.venv\Scripts\python.exe -m evidence_agent train-verifier
& .\.venv\Scripts\python.exe -m evidence_agent calibrate-citation-audit
& .\.venv\Scripts\python.exe -m evidence_agent evaluate --config configs/final.yaml
```

See `docs/final_evaluation_protocol.md` for the configuration contract,
artifact-hash validation, and the ordering guarantee that freezes the runtime
trace before any gold label is read.
