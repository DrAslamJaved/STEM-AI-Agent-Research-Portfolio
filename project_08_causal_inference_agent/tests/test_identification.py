from pathlib import Path

import pytest
import yaml

from causal_audit_agent.causal_graph import (
    REQUIRED_ASSUMPTIONS,
    CausalDAG,
    Variable,
    VariableRole,
)
from causal_audit_agent.identification import (
    IdentificationRequest,
    IdentificationStatus,
    identify_estimand,
)


def dag(variables, edges, **changes):
    values = {
        "treatment": "T",
        "outcome": "Y",
        "variables": tuple(variables),
        "edges": tuple(edges),
        "assumptions": tuple(sorted(REQUIRED_ASSUMPTIONS)),
        "proposed_adjustment_set": (),
    }
    values.update(changes)
    return CausalDAG(**values)


def simple_dag(**changes):
    variables = changes.pop(
        "variables",
        (
            Variable("T", VariableRole.TREATMENT, time_order=0),
            Variable("Y", VariableRole.OUTCOME, time_order=1),
        ),
    )
    edges = changes.pop("edges", (("T", "Y"),))
    return dag(
        variables,
        edges,
        **changes,
    )


def confounded_dag(**changes):
    variables = changes.pop(
        "variables",
        (
            Variable("X", VariableRole.CONFOUNDER, time_order=0),
            Variable("T", VariableRole.TREATMENT, time_order=1),
            Variable("Y", VariableRole.OUTCOME, time_order=2),
        ),
    )
    edges = changes.pop(
        "edges", (("X", "T"), ("X", "Y"), ("T", "Y"))
    )
    return dag(
        variables,
        edges,
        **changes,
    )


def test_unconfounded_ate_is_identified_without_adjustment():
    result = identify_estimand(simple_dag())
    assert result.status is IdentificationStatus.IDENTIFIED
    assert result.identified
    assert result.method == "backdoor_adjustment"
    assert result.adjustment_sets == ((),)
    assert result.expression == "E[Y|T=1]-E[Y|T=0]"
    assert result.requires_human_review


def test_observed_confounder_is_returned_as_minimal_adjustment_set():
    result = identify_estimand(confounded_dag())
    assert result.adjustment_sets == (("X",),)
    assert result.expression == "E_{X}[E[Y|T=1,X]-E[Y|T=0,X]]"
    assert result.reasons == (
        "backdoor_criterion_satisfied",
        "identification_conditional_on_declared_dag",
    )


def test_all_joint_confounders_are_required():
    spec = dag(
        (
            Variable("X1", VariableRole.CONFOUNDER),
            Variable("X2", VariableRole.CONFOUNDER),
            Variable("T", VariableRole.TREATMENT),
            Variable("Y", VariableRole.OUTCOME),
        ),
        (("X1", "T"), ("X1", "Y"), ("X2", "T"), ("X2", "Y"), ("T", "Y")),
    )
    assert identify_estimand(spec).adjustment_sets == (("X1", "X2"),)


def test_alternative_minimal_adjustment_sets_are_preserved():
    spec = dag(
        (
            Variable("X", VariableRole.CONFOUNDER),
            Variable("M", VariableRole.COVARIATE),
            Variable("T", VariableRole.TREATMENT),
            Variable("Y", VariableRole.OUTCOME),
        ),
        (("X", "T"), ("X", "M"), ("M", "Y"), ("T", "Y")),
    )
    result = identify_estimand(spec)
    assert result.adjustment_sets == (("M",), ("X",))
    assert ("M", "X") not in result.adjustment_sets


def test_direct_latent_confounding_is_rejected():
    spec = dag(
        (
            Variable("U", VariableRole.LATENT, observed=False),
            Variable("T", VariableRole.TREATMENT),
            Variable("Y", VariableRole.OUTCOME),
        ),
        (("U", "T"), ("U", "Y"), ("T", "Y")),
    )
    result = identify_estimand(spec)
    assert result.status is IdentificationStatus.NON_IDENTIFIABLE
    assert not result.identified
    assert result.adjustment_sets == ()
    assert result.expression is None
    assert result.reasons == (
        "no_observed_backdoor_adjustment_set",
        "latent_common_cause:U",
    )


def test_observed_node_can_block_path_from_latent_ancestor():
    spec = dag(
        (
            Variable("U", VariableRole.LATENT, observed=False),
            Variable("X", VariableRole.CONFOUNDER),
            Variable("T", VariableRole.TREATMENT),
            Variable("Y", VariableRole.OUTCOME),
        ),
        (("U", "X"), ("X", "T"), ("U", "Y"), ("T", "Y")),
    )
    result = identify_estimand(spec)
    assert result.status is IdentificationStatus.IDENTIFIED
    assert result.adjustment_sets == (("X",),)


def test_default_request_is_created_when_omitted():
    assert identify_estimand(simple_dag()).estimand == "ATE"


@pytest.mark.parametrize("value", [None, {}])
def test_invalid_spec_type_fails_explicitly(value):
    with pytest.raises(TypeError, match="spec must be a CausalDAG"):
        identify_estimand(value)


def test_invalid_request_type_fails_explicitly():
    with pytest.raises(TypeError, match="request must be an IdentificationRequest"):
        identify_estimand(simple_dag(), {})


@pytest.mark.parametrize("conditioned_on", [None, "V", ("V", 3)])
def test_malformed_conditioning_collection_is_rejected(conditioned_on):
    request = IdentificationRequest(estimand="CATE", conditioned_on=conditioned_on)
    result = identify_estimand(simple_dag(), request)
    assert result.status is IdentificationStatus.INVALID_QUERY
    assert result.reasons == (
        "conditioning_variables_must_be_a_sequence_of_strings",
    )


def test_invalid_graph_is_rejected_before_identification():
    spec = simple_dag(edges=(("T", "Y"), ("Y", "T")))
    result = identify_estimand(spec)
    assert result.status is IdentificationStatus.INVALID_QUERY
    assert "graph_error:graph_contains_cycle" in result.reasons


@pytest.mark.parametrize("estimand", ["median", "", 42])
def test_unsupported_estimand_is_rejected(estimand):
    result = identify_estimand(simple_dag(), IdentificationRequest(estimand=estimand))
    assert result.status is IdentificationStatus.UNSUPPORTED_QUERY
    assert result.reasons[0].startswith("unsupported_estimand:")


def test_direct_effect_request_is_not_silently_treated_as_total_effect():
    request = IdentificationRequest(effect_type="direct")
    result = identify_estimand(simple_dag(), request)
    assert result.status is IdentificationStatus.UNSUPPORTED_QUERY
    assert result.reasons == ("unsupported_effect_type:direct",)


def test_missing_assumptions_block_identification():
    result = identify_estimand(simple_dag(assumptions=("consistency",)))
    assert result.status is IdentificationStatus.INSUFFICIENT_ASSUMPTIONS
    assert "missing_assumption:exchangeability" in result.reasons
    assert result.assumptions == ("consistency",)


def test_blank_assumptions_are_ignored():
    result = identify_estimand(simple_dag(assumptions=("consistency", "  ")))
    assert "missing_assumption:exchangeability" in result.reasons


def test_att_uses_treated_covariate_distribution():
    result = identify_estimand(
        confounded_dag(), IdentificationRequest(estimand="att")
    )
    assert result.estimand == "ATT"
    assert result.expression == "E_{X|T=1}[E[Y|T=1,X]-E[Y|T=0,X]]"


def test_unconfounded_att_has_plain_contrast():
    result = identify_estimand(simple_dag(), IdentificationRequest(estimand="ATT"))
    assert result.expression == "E[Y|T=1]-E[Y|T=0]"


def test_cate_requires_conditioning_variables():
    result = identify_estimand(simple_dag(), IdentificationRequest(estimand="CATE"))
    assert result.status is IdentificationStatus.INVALID_QUERY
    assert result.reasons == ("cate_requires_conditioning_variables",)


def test_conditioning_variables_are_reserved_for_cate():
    result = identify_estimand(
        simple_dag(), IdentificationRequest(estimand="ATE", conditioned_on=("V",))
    )
    assert result.reasons == ("conditioning_variables_require_cate",)


def test_valid_cate_is_identified_conditionally():
    spec = dag(
        (
            Variable("V", VariableRole.COVARIATE),
            Variable("X", VariableRole.CONFOUNDER),
            Variable("T", VariableRole.TREATMENT),
            Variable("Y", VariableRole.OUTCOME),
        ),
        (("V", "T"), ("V", "Y"), ("X", "T"), ("X", "Y"), ("T", "Y")),
    )
    request = IdentificationRequest(estimand="CATE", conditioned_on=("V", "V"))
    result = identify_estimand(spec, request)
    assert result.conditioning_variables == ("V",)
    assert result.adjustment_sets == (("X",),)
    assert result.expression == "E_{X|V}[E[Y|T=1,X,V]-E[Y|T=0,X,V]]"


def test_unconfounded_cate_has_conditional_contrast():
    variables = simple_dag().variables + (Variable("V", VariableRole.COVARIATE),)
    result = identify_estimand(
        simple_dag(variables=variables),
        IdentificationRequest(estimand="CATE", conditioned_on=("V",)),
    )
    assert result.expression == "E[Y|T=1,V]-E[Y|T=0,V]"


@pytest.mark.parametrize(
    ("variable", "edges", "reason"),
    [
        (None, (("T", "Y"),), "unknown_conditioning_variable:V"),
        (
            Variable("V", VariableRole.LATENT, observed=False),
            (("V", "T"), ("V", "Y"), ("T", "Y")),
            "unobserved_conditioning_variable:V",
        ),
        (
            Variable("V", VariableRole.MEDIATOR),
            (("T", "V"), ("V", "Y"), ("T", "Y")),
            "post_treatment_conditioning:V",
        ),
        (
            Variable("V", VariableRole.COLLIDER),
            (("T", "Y"),),
            "collider_conditioning:V",
        ),
    ],
)
def test_invalid_cate_conditioning_is_rejected(variable, edges, reason):
    variables = list(simple_dag().variables)
    if variable is not None:
        variables.append(variable)
    spec = dag(variables, edges)
    result = identify_estimand(
        spec, IdentificationRequest(estimand="CATE", conditioned_on=("V",))
    )
    assert result.status is IdentificationStatus.INVALID_QUERY
    assert reason in result.reasons


@pytest.mark.parametrize("name", ["T", "Y"])
def test_treatment_or_outcome_cannot_be_cate_condition(name):
    result = identify_estimand(
        simple_dag(), IdentificationRequest(estimand="CATE", conditioned_on=(name,))
    )
    assert result.reasons == (f"invalid_conditioning_target:{name}",)


def test_large_exhaustive_search_is_refused():
    covariates = tuple(
        Variable(f"X{i:02d}", VariableRole.CONFOUNDER) for i in range(13)
    )
    variables = covariates + simple_dag().variables
    edges = tuple((variable.name, "T") for variable in covariates)
    edges += tuple((variable.name, "Y") for variable in covariates)
    edges += (("T", "Y"),)
    result = identify_estimand(dag(variables, edges))
    assert result.status is IdentificationStatus.UNSUPPORTED_QUERY
    assert result.reasons == ("candidate_search_space_exceeded:13>12",)


def test_invalid_proposed_adjustment_blocks_identification():
    spec = confounded_dag(proposed_adjustment_set=("T",))
    result = identify_estimand(spec)
    assert result.status is IdentificationStatus.INVALID_QUERY
    assert "graph_error:invalid_adjustment_target:T" in result.reasons


def test_result_serialization_is_json_compatible():
    data = identify_estimand(confounded_dag()).to_dict()
    assert data["status"] == "IDENTIFIED"
    assert data["adjustment_sets"] == [["X"]]
    assert data["conditioning_variables"] == []
    assert data["requires_human_review"] is True


def test_identification_example_config_is_parseable():
    path = Path(__file__).parents[1] / "configs" / "identification" / "examples.yaml"
    examples = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert examples["identifiable_backdoor"]["expected_status"] == "IDENTIFIED"
    assert examples["unobserved_confounding"]["expected_status"] == "NON_IDENTIFIABLE"
