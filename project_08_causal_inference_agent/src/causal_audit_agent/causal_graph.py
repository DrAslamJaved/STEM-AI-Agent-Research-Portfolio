from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import networkx as nx


class VariableRole(str, Enum):
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    CONFOUNDER = "confounder"
    MEDIATOR = "mediator"
    COLLIDER = "collider"
    INSTRUMENT = "instrument"
    COVARIATE = "covariate"
    LATENT = "latent"


REQUIRED_ASSUMPTIONS = frozenset(
    {
        "consistency",
        "exchangeability",
        "positivity",
        "no_interference",
        "well_defined_intervention",
        "temporal_order",
    }
)


@dataclass(frozen=True)
class Variable:
    name: str
    role: VariableRole
    observed: bool = True
    time_order: int | None = None


@dataclass(frozen=True)
class CausalDAG:
    treatment: str
    outcome: str
    variables: tuple[Variable, ...]
    edges: tuple[tuple[str, str], ...]
    assumptions: tuple[str, ...] = ()
    proposed_adjustment_set: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CausalDAG":
        """Create a graph specification from a YAML-compatible mapping."""
        if not isinstance(data, Mapping):
            raise TypeError("data must be a mapping")

        raw_variables = data.get("variables", ())
        if not isinstance(raw_variables, (list, tuple)):
            raise TypeError("variables must be a sequence")

        variables: list[Variable] = []
        for raw in raw_variables:
            if not isinstance(raw, Mapping):
                raise TypeError("each variable must be a mapping")
            variables.append(
                Variable(
                    name=str(raw.get("name", "")),
                    role=VariableRole(str(raw.get("role", ""))),
                    observed=bool(raw.get("observed", True)),
                    time_order=raw.get("time_order"),
                )
            )

        raw_edges = data.get("edges", ())
        if not isinstance(raw_edges, (list, tuple)):
            raise TypeError("edges must be a sequence")

        edges: list[tuple[str, str]] = []
        for raw in raw_edges:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError("each edge must contain exactly two node names")
            edges.append((str(raw[0]), str(raw[1])))

        return cls(
            treatment=str(data.get("treatment", "")),
            outcome=str(data.get("outcome", "")),
            variables=tuple(variables),
            edges=tuple(edges),
            assumptions=tuple(str(item) for item in data.get("assumptions", ())),
            proposed_adjustment_set=tuple(
                str(item) for item in data.get("proposed_adjustment_set", ())
            ),
        )


@dataclass(frozen=True)
class GraphAudit:
    valid_for_identification_review: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    missing_assumptions: tuple[str, ...]
    requires_human_review: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_for_identification_review": self.valid_for_identification_review,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "missing_assumptions": list(self.missing_assumptions),
            "requires_human_review": self.requires_human_review,
        }


def audit_causal_dag(spec: CausalDAG) -> GraphAudit:
    """Audit DAG structure and declared assumptions without claiming identification.

    Passing this audit means only that the graph is structurally eligible for the
    later identification stage. It does not prove that its causal arrows or
    untestable assumptions are true.
    """
    if not isinstance(spec, CausalDAG):
        raise TypeError("spec must be a CausalDAG")

    errors: list[str] = []
    warnings: list[str] = []
    names = [variable.name for variable in spec.variables]
    name_set = set(names)

    if not spec.treatment.strip():
        errors.append("missing_treatment")
    if not spec.outcome.strip():
        errors.append("missing_outcome")
    if spec.treatment == spec.outcome and spec.treatment:
        errors.append("treatment_equals_outcome")
    if any(not name.strip() for name in names):
        errors.append("blank_variable_name")
    if len(names) != len(name_set):
        errors.append("duplicate_variable_name")
    if spec.treatment not in name_set:
        errors.append("treatment_not_declared")
    if spec.outcome not in name_set:
        errors.append("outcome_not_declared")

    variables = {variable.name: variable for variable in spec.variables}
    treatment_variable = variables.get(spec.treatment)
    outcome_variable = variables.get(spec.outcome)
    if treatment_variable and treatment_variable.role is not VariableRole.TREATMENT:
        errors.append("treatment_role_mismatch")
    if outcome_variable and outcome_variable.role is not VariableRole.OUTCOME:
        errors.append("outcome_role_mismatch")
    if treatment_variable and not treatment_variable.observed:
        errors.append("treatment_must_be_observed")
    if outcome_variable and not outcome_variable.observed:
        errors.append("outcome_must_be_observed")

    graph = nx.DiGraph()
    graph.add_nodes_from(name_set)
    for source, target in spec.edges:
        if source not in name_set or target not in name_set:
            errors.append(f"edge_uses_undeclared_variable:{source}->{target}")
            continue
        if source == target:
            errors.append(f"self_loop:{source}")
            continue
        graph.add_edge(source, target)

    if nx.is_directed_acyclic_graph(graph):
        if (
            spec.treatment in graph
            and spec.outcome in graph
            and not nx.has_path(graph, spec.treatment, spec.outcome)
        ):
            errors.append("no_directed_treatment_outcome_path")
    else:
        errors.append("graph_contains_cycle")

    for source, target in graph.edges:
        source_time = variables[source].time_order
        target_time = variables[target].time_order
        if source_time is not None and target_time is not None and source_time > target_time:
            errors.append(f"temporal_order_violation:{source}->{target}")

    if nx.is_directed_acyclic_graph(graph) and spec.treatment in graph and spec.outcome in graph:
        for variable in spec.variables:
            if (
                not variable.observed
                and variable.name not in {spec.treatment, spec.outcome}
                and nx.has_path(graph, variable.name, spec.treatment)
                and nx.has_path(graph, variable.name, spec.outcome)
            ):
                warnings.append(f"latent_common_cause:{variable.name}")

        descendants = nx.descendants(graph, spec.treatment)
        for name in spec.proposed_adjustment_set:
            variable = variables.get(name)
            if name in {spec.treatment, spec.outcome}:
                errors.append(f"invalid_adjustment_target:{name}")
            elif variable is None:
                errors.append(f"unknown_adjustment_variable:{name}")
            elif not variable.observed:
                errors.append(f"unobserved_adjustment_variable:{name}")
            elif name in descendants:
                errors.append(f"post_treatment_adjustment:{name}")
            elif variable.role is VariableRole.COLLIDER:
                errors.append(f"collider_adjustment:{name}")

    declared_assumptions = {item.strip() for item in spec.assumptions if item.strip()}
    missing = tuple(sorted(REQUIRED_ASSUMPTIONS - declared_assumptions))
    warnings.extend(f"undeclared_assumption:{item}" for item in missing)

    return GraphAudit(
        valid_for_identification_review=not errors,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        missing_assumptions=missing,
    )
