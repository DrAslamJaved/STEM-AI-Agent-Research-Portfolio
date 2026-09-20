import pytest

from causal_audit_agent.causal_graph import (
    REQUIRED_ASSUMPTIONS,
    CausalDAG,
    Variable,
    VariableRole,
    audit_causal_dag,
)


def valid_spec(**changes):
    values = {
        "treatment": "T",
        "outcome": "Y",
        "variables": (
            Variable("X", VariableRole.CONFOUNDER, time_order=0),
            Variable("T", VariableRole.TREATMENT, time_order=1),
            Variable("Y", VariableRole.OUTCOME, time_order=2),
        ),
        "edges": (("X", "T"), ("X", "Y"), ("T", "Y")),
        "assumptions": tuple(sorted(REQUIRED_ASSUMPTIONS)),
        "proposed_adjustment_set": ("X",),
    }
    values.update(changes)
    return CausalDAG(**values)


def test_valid_dag_is_eligible_but_still_requires_human_review():
    audit = audit_causal_dag(valid_spec())
    assert audit.valid_for_identification_review
    assert not audit.errors
    assert not audit.warnings
    assert not audit.missing_assumptions
    assert audit.requires_human_review


def test_audit_serialization_uses_json_compatible_collections():
    result = audit_causal_dag(valid_spec()).to_dict()
    assert result == {
        "valid_for_identification_review": True,
        "errors": [],
        "warnings": [],
        "missing_assumptions": [],
        "requires_human_review": True,
    }


def test_mapping_constructor_builds_expected_specification():
    spec = CausalDAG.from_mapping(
        {
            "treatment": "T",
            "outcome": "Y",
            "variables": [
                {"name": "T", "role": "treatment", "time_order": 0},
                {"name": "Y", "role": "outcome", "time_order": 1},
            ],
            "edges": [["T", "Y"]],
            "assumptions": sorted(REQUIRED_ASSUMPTIONS),
            "proposed_adjustment_set": [],
        }
    )
    assert spec.variables[0] == Variable("T", VariableRole.TREATMENT, True, 0)
    assert spec.edges == (("T", "Y"),)
    assert audit_causal_dag(spec).valid_for_identification_review


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "data must be a mapping"),
        ({"variables": "T"}, "variables must be a sequence"),
        ({"variables": ["T"]}, "each variable must be a mapping"),
        ({"variables": [], "edges": "T->Y"}, "edges must be a sequence"),
    ],
)
def test_mapping_constructor_rejects_malformed_collections(value, message):
    with pytest.raises(TypeError, match=message):
        CausalDAG.from_mapping(value)


def test_mapping_constructor_rejects_malformed_edge():
    with pytest.raises(ValueError, match="exactly two"):
        CausalDAG.from_mapping({"variables": [], "edges": [["T"]]})


def test_mapping_constructor_rejects_unknown_role():
    with pytest.raises(ValueError, match="not a valid VariableRole"):
        CausalDAG.from_mapping(
            {"variables": [{"name": "T", "role": "exposure"}], "edges": []}
        )


def test_audit_rejects_wrong_input_type():
    with pytest.raises(TypeError, match="spec must be a CausalDAG"):
        audit_causal_dag({})


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"treatment": ""}, "missing_treatment"),
        ({"outcome": ""}, "missing_outcome"),
        ({"outcome": "T"}, "treatment_equals_outcome"),
        (
            {
                "variables": (
                    Variable("", VariableRole.COVARIATE),
                    Variable("T", VariableRole.TREATMENT),
                    Variable("Y", VariableRole.OUTCOME),
                )
            },
            "blank_variable_name",
        ),
        (
            {
                "variables": (
                    Variable("T", VariableRole.TREATMENT),
                    Variable("T", VariableRole.COVARIATE),
                    Variable("Y", VariableRole.OUTCOME),
                )
            },
            "duplicate_variable_name",
        ),
        ({"treatment": "A"}, "treatment_not_declared"),
        ({"outcome": "Z"}, "outcome_not_declared"),
    ],
)
def test_core_contract_errors_are_reported(changes, error):
    assert error in audit_causal_dag(valid_spec(**changes)).errors


@pytest.mark.parametrize(
    ("variables", "error"),
    [
        (
            (
                Variable("X", VariableRole.CONFOUNDER),
                Variable("T", VariableRole.COVARIATE),
                Variable("Y", VariableRole.OUTCOME),
            ),
            "treatment_role_mismatch",
        ),
        (
            (
                Variable("X", VariableRole.CONFOUNDER),
                Variable("T", VariableRole.TREATMENT),
                Variable("Y", VariableRole.COVARIATE),
            ),
            "outcome_role_mismatch",
        ),
        (
            (
                Variable("X", VariableRole.CONFOUNDER),
                Variable("T", VariableRole.TREATMENT, observed=False),
                Variable("Y", VariableRole.OUTCOME),
            ),
            "treatment_must_be_observed",
        ),
        (
            (
                Variable("X", VariableRole.CONFOUNDER),
                Variable("T", VariableRole.TREATMENT),
                Variable("Y", VariableRole.OUTCOME, observed=False),
            ),
            "outcome_must_be_observed",
        ),
    ],
)
def test_treatment_and_outcome_metadata_are_audited(variables, error):
    assert error in audit_causal_dag(valid_spec(variables=variables)).errors


@pytest.mark.parametrize(
    ("edges", "error"),
    [
        ((("X", "T"), ("T", "Z")), "edge_uses_undeclared_variable:T->Z"),
        ((("T", "T"), ("T", "Y")), "self_loop:T"),
        ((("T", "X"), ("X", "T"), ("T", "Y")), "graph_contains_cycle"),
        ((("X", "T"), ("X", "Y")), "no_directed_treatment_outcome_path"),
    ],
)
def test_structural_graph_errors_are_reported(edges, error):
    assert error in audit_causal_dag(valid_spec(edges=edges)).errors


def test_temporal_order_violation_is_reported():
    variables = (
        Variable("T", VariableRole.TREATMENT, time_order=1),
        Variable("Y", VariableRole.OUTCOME, time_order=0),
    )
    audit = audit_causal_dag(
        valid_spec(variables=variables, edges=(("T", "Y"),), proposed_adjustment_set=())
    )
    assert "temporal_order_violation:T->Y" in audit.errors


def test_unknown_time_order_is_not_invented():
    variables = (
        Variable("T", VariableRole.TREATMENT),
        Variable("Y", VariableRole.OUTCOME),
    )
    audit = audit_causal_dag(
        valid_spec(variables=variables, edges=(("T", "Y"),), proposed_adjustment_set=())
    )
    assert not any(item.startswith("temporal_order_violation") for item in audit.errors)


def test_latent_ancestor_of_treatment_and_outcome_is_flagged():
    variables = (
        Variable("U", VariableRole.LATENT, observed=False, time_order=0),
        Variable("M", VariableRole.COVARIATE, time_order=1),
        Variable("T", VariableRole.TREATMENT, time_order=2),
        Variable("Y", VariableRole.OUTCOME, time_order=3),
    )
    audit = audit_causal_dag(
        valid_spec(
            variables=variables,
            edges=(("U", "M"), ("M", "T"), ("U", "Y"), ("T", "Y")),
            proposed_adjustment_set=(),
        )
    )
    assert audit.valid_for_identification_review
    assert "latent_common_cause:U" in audit.warnings


@pytest.mark.parametrize(
    ("adjustment", "error"),
    [
        (("T",), "invalid_adjustment_target:T"),
        (("Y",), "invalid_adjustment_target:Y"),
        (("Z",), "unknown_adjustment_variable:Z"),
    ],
)
def test_invalid_adjustment_targets_are_rejected(adjustment, error):
    audit = audit_causal_dag(valid_spec(proposed_adjustment_set=adjustment))
    assert error in audit.errors


def test_unobserved_adjustment_variable_is_rejected():
    variables = valid_spec().variables + (
        Variable("U", VariableRole.LATENT, observed=False, time_order=0),
    )
    audit = audit_causal_dag(
        valid_spec(variables=variables, proposed_adjustment_set=("U",))
    )
    assert "unobserved_adjustment_variable:U" in audit.errors


def test_post_treatment_adjustment_is_rejected():
    variables = valid_spec().variables + (
        Variable("M", VariableRole.MEDIATOR, time_order=2),
    )
    audit = audit_causal_dag(
        valid_spec(
            variables=variables,
            edges=(("X", "T"), ("X", "Y"), ("T", "M"), ("M", "Y")),
            proposed_adjustment_set=("M",),
        )
    )
    assert "post_treatment_adjustment:M" in audit.errors


def test_declared_collider_adjustment_is_rejected():
    variables = valid_spec().variables + (
        Variable("C", VariableRole.COLLIDER, time_order=1),
    )
    audit = audit_causal_dag(
        valid_spec(
            variables=variables,
            edges=(("X", "T"), ("X", "Y"), ("T", "Y")),
            proposed_adjustment_set=("C",),
        )
    )
    assert "collider_adjustment:C" in audit.errors


def test_missing_assumptions_are_sorted_and_reported_as_warnings():
    audit = audit_causal_dag(valid_spec(assumptions=("consistency", "  ")))
    assert audit.missing_assumptions == tuple(
        sorted(REQUIRED_ASSUMPTIONS - {"consistency"})
    )
    assert "undeclared_assumption:positivity" in audit.warnings
    assert audit.valid_for_identification_review


def test_duplicate_diagnostics_are_collapsed_stably():
    audit = audit_causal_dag(
        valid_spec(
            edges=(("T", "Z"), ("T", "Z"), ("T", "Y")),
            proposed_adjustment_set=("Z", "Z"),
        )
    )
    assert audit.errors.count("edge_uses_undeclared_variable:T->Z") == 1
    assert audit.errors.count("unknown_adjustment_variable:Z") == 1
