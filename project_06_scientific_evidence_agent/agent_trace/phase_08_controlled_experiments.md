# Phase 08 controlled-experiments execution trace

1. Validated every fixed input artifact against its declared SHA-256.
2. Loaded only corpus text, claims, index, verifier bundle, and frozen calibration policy.
3. Wrote and hashed one raw gold-free runtime trace.
4. Produced direct-RAG and audited-agent official-format predictions from that same trace.
5. Only then loaded development gold annotations for scoring and paired bootstrap analysis.
6. Ran deterministic adversarial evaluator checks for document, stance, and rationale failures.

Result JSON SHA-256: `32fe3b1d7c6886db580b08ad9ffb694f85961bd7b19cafb980e1507a2c081b5d`
Raw trace SHA-256: `43679c7ec8b69d4b21666434039e477901a9cbe64964772fafb146919bdccdc9`
