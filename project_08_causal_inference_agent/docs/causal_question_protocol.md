# Causal-question protocol

## Purpose

The parser is a conservative lexical triage component, not a scientific
authority. It distinguishes causal, predictive, associational, and ambiguous
wording so the agent does not silently answer an associational or predictive
question as though it were causal.

## Decision policy

- A question with cues from exactly one intent family receives that label.
- A question with no recognized cues is `AMBIGUOUS`.
- A question with cues from multiple intent families is `AMBIGUOUS`, even when
  one family has more matching cues than another.
- Matched cue codes are retained for auditability.
- Every causal or ambiguous classification requires human review before causal
  modeling proceeds.

The reported `confidence` value is a deterministic routing score. It is not a
calibrated probability, evidence for a causal claim, or evidence that an
estimand is identifiable.

## Required causal specification

A causal analysis still requires an explicit intervention, treatment levels,
outcome, target population, estimand, temporal order, and an approved causal
graph. Classification as causal only selects the next workflow gate; it does
not authorize estimation.

## Limitations

This phase uses transparent regular-expression cues. It does not resolve
negation, sarcasm, domain-specific language, or semantic context. Unrecognized
or mixed wording must remain `HUMAN_REVIEW_REQUIRED`; it is never silently
rewritten into a causal question.
