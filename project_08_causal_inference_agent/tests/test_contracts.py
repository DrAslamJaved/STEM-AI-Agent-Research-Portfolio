import pytest

from causal_audit_agent.contracts import CausalQuestion


def test_valid_contract_serializes():
    q = CausalQuestion(
        "Effect of T on Y?",
        "T",
        "Y",
        treatment_time="baseline",
        outcome_time="follow_up",
    )
    payload = q.to_dict()
    assert payload["estimand"] == "ATE"
    assert payload["question_type"] == "causal"
    assert payload["treatment_time"] == "baseline"
    assert payload["outcome_time"] == "follow_up"


@pytest.mark.parametrize("field", ["question", "treatment", "outcome", "estimand", "population"])
def test_required_core_fields_reject_blank_values(field):
    values = {
        "question": "Effect of T on Y?",
        "treatment": "T",
        "outcome": "Y",
        "estimand": "ATE",
        "population": "study population",
    }
    values[field] = ""
    with pytest.raises(ValueError, match="Missing required causal fields"):
        CausalQuestion(**values).validate()


def test_treatment_and_outcome_must_differ():
    with pytest.raises(ValueError, match="distinct"):
        CausalQuestion("Effect?", "Y", "Y").validate()


def test_estimand_is_controlled():
    with pytest.raises(ValueError, match="Unsupported"):
        CausalQuestion("Effect?", "T", "Y", estimand="ITE").validate()
