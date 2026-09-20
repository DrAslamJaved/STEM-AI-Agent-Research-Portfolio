import pytest

from causal_audit_agent.contracts import CausalQuestion


def test_valid_contract_serializes():
    q = CausalQuestion("Effect of T on Y?", "T", "Y")
    assert q.to_dict()["estimand"] == "ATE"


def test_treatment_and_outcome_must_differ():
    with pytest.raises(ValueError, match="distinct"):
        CausalQuestion("Effect?", "Y", "Y").validate()


def test_estimand_is_controlled():
    with pytest.raises(ValueError, match="Unsupported"):
        CausalQuestion("Effect?", "T", "Y", estimand="ITE").validate()
