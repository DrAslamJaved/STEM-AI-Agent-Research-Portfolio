from __future__ import annotations

import re
from dataclasses import dataclass

from causal_audit_agent.contracts import QuestionType


@dataclass(frozen=True)
class Classification:
    """Conservative lexical triage result for a research question."""

    label: QuestionType
    confidence: float
    matched_cues: tuple[str, ...]
    requires_human_review: bool


CUES: dict[QuestionType, tuple[tuple[str, str], ...]] = {
    QuestionType.CAUSAL: (
        ("counterfactual_if", r"\bwhat (?:would|will) happen if\b"),
        ("causal_effect", r"\b(?:causal\s+)?effect of\b"),
        ("intervention", r"\binterven(?:e|es|ed|ing|tion|tions)\b"),
        ("do_operator", r"\bdo\s*\("),
        ("cause_language", r"\b(?:cause|causes|caused|causal|causality)\b"),
    ),
    QuestionType.PREDICTIVE: (
        ("prediction", r"\bpredict(?:s|ed|ing|ion|ions|ive)?\b"),
        ("forecast", r"\bforecast(?:s|ed|ing)?\b"),
        ("future_value", r"\bwhat will .+ be\b"),
        ("risk_probability", r"\b(?:risk|probability|likelihood) of\b"),
    ),
    QuestionType.ASSOCIATIONAL: (
        ("association", r"\bassociat(?:e|es|ed|ing|ion|ions|ional)\b"),
        ("correlation", r"\bcorrelat(?:e|es|ed|ing|ion|ions)\b"),
        ("relationship", r"\brelationship between\b"),
        ("linked", r"\blinked to\b"),
    ),
}


def classify_question(text: str) -> Classification:
    """Classify question intent without converting association into causation.

    ``confidence`` is a deterministic routing score, not a calibrated
    probability and not evidence that a causal effect is identifiable.
    Questions with no recognized cue or cues from multiple intent families are
    labeled ambiguous and require human review.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = " ".join(text.casefold().split())
    if not normalized:
        return Classification(QuestionType.AMBIGUOUS, 0.0, (), True)

    matches = {
        label: tuple(code for code, pattern in cues if re.search(pattern, normalized))
        for label, cues in CUES.items()
    }
    active_labels = tuple(label for label, cues in matches.items() if cues)
    all_cues = tuple(code for cues in matches.values() for code in cues)

    if len(active_labels) != 1:
        return Classification(QuestionType.AMBIGUOUS, 0.0, all_cues, True)

    label = active_labels[0]
    routing_score = min(0.95, 0.65 + 0.10 * (len(matches[label]) - 1))
    requires_human_review = label is QuestionType.CAUSAL
    return Classification(label, routing_score, matches[label], requires_human_review)
