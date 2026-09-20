import pytest

from causal_audit_agent.contracts import QuestionType
from causal_audit_agent.query_parser import classify_question


@pytest.mark.parametrize(
    "text",
    [
        "What would happen if treatment T were assigned?",
        "What is the effect of treatment T on outcome Y?",
        "Estimate the response under intervention T.",
        "Identify E[Y | do(T=1)].",
        "Does exposure T cause outcome Y?",
    ],
)
def test_causal_language_is_classified_as_causal(text):
    result = classify_question(text)
    assert result.label is QuestionType.CAUSAL
    assert result.requires_human_review
    assert result.matched_cues


@pytest.mark.parametrize(
    "text",
    [
        "Predict next year's outcome.",
        "Forecast hospital admissions for next month.",
        "What will the biomarker value be tomorrow?",
        "Estimate the risk of readmission.",
    ],
)
def test_predictive_language_is_not_causal(text):
    result = classify_question(text)
    assert result.label is QuestionType.PREDICTIVE
    assert not result.requires_human_review


@pytest.mark.parametrize(
    "text",
    [
        "Is treatment associated with the outcome?",
        "Measure the correlation between X and Y.",
        "Describe the relationship between exposure and disease.",
        "Is biomarker X linked to disease Y?",
    ],
)
def test_associational_language_is_not_promoted_to_causal(text):
    result = classify_question(text)
    assert result.label is QuestionType.ASSOCIATIONAL
    assert not result.requires_human_review


def test_normalization_is_case_and_whitespace_insensitive():
    result = classify_question("  WHAT   IS THE CAUSAL EFFECT OF T ON Y?  ")
    assert result.label is QuestionType.CAUSAL
    assert "causal_effect" in result.matched_cues
    assert "cause_language" in result.matched_cues
    assert result.confidence == pytest.approx(0.75)


@pytest.mark.parametrize("text", ["Tell me about T and Y", "", "   \n\t "])
def test_underspecified_question_abstains(text):
    result = classify_question(text)
    assert result.label is QuestionType.AMBIGUOUS
    assert result.confidence == 0.0
    assert result.requires_human_review


@pytest.mark.parametrize(
    "text",
    [
        "Predict the causal effect of treatment T on Y.",
        "Forecast whether X is associated with Y.",
    ],
)
def test_mixed_intent_abstains_and_preserves_diagnostics(text):
    result = classify_question(text)
    assert result.label is QuestionType.AMBIGUOUS
    assert result.confidence == 0.0
    assert result.requires_human_review
    assert len(result.matched_cues) >= 2


@pytest.mark.parametrize("text", [None, 42])
def test_non_string_input_fails_explicitly(text):
    with pytest.raises(TypeError, match="text must be a string"):
        classify_question(text)


def test_routing_score_is_bounded_and_not_full_certainty():
    result = classify_question("What is the causal effect under intervention do(T=1)?")
    assert 0.0 < result.confidence <= 0.95
    assert result.confidence < 1.0
