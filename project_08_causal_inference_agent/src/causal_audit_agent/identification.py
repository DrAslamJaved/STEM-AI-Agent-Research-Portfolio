from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any

import networkx as nx

from causal_audit_agent.causal_graph import (
    REQUIRED_ASSUMPTIONS,
    CausalDAG,
    VariableRole,
    audit_causal_dag,
)


class IdentificationStatus(str, Enum):
    IDENTIFIED = "IDENTIFIED"
    NON_IDENTIFIABLE = "NON_IDENTIFIABLE"
    INSUFFICIENT_ASSUMPTIONS = "INSUFFICIENT_ASSUMPTIONS"
    INVALID_QUERY = "INVALID_QUERY"
    UNSUPPORTED_QUERY = "UNSUPPORTED_QUERY"


SUPPORTED_ESTIMANDS = frozenset({"ATE", "ATT", "CATE"})
MAX_ADJUSTMENT_CANDIDATES = 12


@dataclass(frozen=True)
class IdentificationRequest:
    estimand: str = "ATE"
    effect_type: str = "total"
    conditioned_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class IdentificationResult:
    status: IdentificationStatus
    estimand: str
    method: str | None
    adjustment_sets: tuple[tuple[str, ...], ...]
    conditioning_variables: tuple[str, ...]
    expression: str | None
    reasons: tuple[str, ...]
    assumptions: tuple[str, ...]
    requires_human_review: bool = True

    @property
    def identified(self) -> bool:
        return self.status is IdentificationStatus.IDENTIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "estimand": self.estimand,
            "method": self.method,
            "adjustment_sets": [list(items) for items in self.adjustment_sets],
            "conditioning_variables": list(self.conditioning_variables),
            "expression": self.expression,
            "reasons": list(self.reasons),
            "assumptions": list(self.assumptions),
            "requires_human_review": self.requires_human_review,
        }


def identify_estimand(
    spec: CausalDAG,
    request: IdentificationRequest | None = None,
) -> IdentificationResult:
    """Identify a total-effect estimand by the observed back-door criterion.

    This deterministic gate deliberately refuses unsupported strategies. A
    successful result is conditional on the supplied DAG and assumptions; it
    is not evidence that either is scientifically true.
    """
    if not isinstance(spec, CausalDAG):
        raise TypeError("spec must be a CausalDAG")
    if request is None:
        request = IdentificationRequest()
    if not isinstance(request, IdentificationRequest):
        raise TypeError("request must be an IdentificationRequest")

    estimand = request.estimand.upper() if isinstance(request.estimand, str) else ""
    assumptions = tuple(sorted(item.strip() for item in spec.assumptions if item.strip()))
    if not isinstance(request.conditioned_on, (list, tuple)) or any(
        not isinstance(item, str) for item in request.conditioned_on
    ):
        return _result(
            IdentificationStatus.INVALID_QUERY,
            estimand,
            (),
            ("conditioning_variables_must_be_a_sequence_of_strings",),
            assumptions,
        )
    conditioning = tuple(dict.fromkeys(request.conditioned_on))

    graph_audit = audit_causal_dag(spec)
    if graph_audit.errors:
        return _result(
            IdentificationStatus.INVALID_QUERY,
            estimand,
            conditioning,
            tuple(f"graph_error:{item}" for item in graph_audit.errors),
            assumptions,
        )

    if estimand not in SUPPORTED_ESTIMANDS:
        return _result(
            IdentificationStatus.UNSUPPORTED_QUERY,
            estimand,
            conditioning,
            (f"unsupported_estimand:{estimand or '<blank>'}",),
            assumptions,
        )

    if request.effect_type != "total":
        return _result(
            IdentificationStatus.UNSUPPORTED_QUERY,
            estimand,
            conditioning,
            (f"unsupported_effect_type:{request.effect_type}",),
            assumptions,
        )

    conditioning_errors = _validate_conditioning(spec, estimand, conditioning)
    if conditioning_errors:
        return _result(
            IdentificationStatus.INVALID_QUERY,
            estimand,
            conditioning,
            conditioning_errors,
            assumptions,
        )

    missing_assumptions = tuple(sorted(REQUIRED_ASSUMPTIONS - set(assumptions)))
    if missing_assumptions:
        return _result(
            IdentificationStatus.INSUFFICIENT_ASSUMPTIONS,
            estimand,
            conditioning,
            tuple(f"missing_assumption:{item}" for item in missing_assumptions),
            assumptions,
        )

    graph = nx.DiGraph()
    graph.add_nodes_from(variable.name for variable in spec.variables)
    graph.add_edges_from(spec.edges)
    descendants = nx.descendants(graph, spec.treatment)
    ancestors = nx.ancestors(graph, spec.treatment) | nx.ancestors(graph, spec.outcome)

    candidates = tuple(
        sorted(
            variable.name
            for variable in spec.variables
            if variable.observed
            and variable.name in ancestors
            and variable.name not in {spec.treatment, spec.outcome, *conditioning}
            and variable.name not in descendants
            and variable.role not in {VariableRole.COLLIDER, VariableRole.LATENT}
        )
    )

    if len(candidates) > MAX_ADJUSTMENT_CANDIDATES:
        return _result(
            IdentificationStatus.UNSUPPORTED_QUERY,
            estimand,
            conditioning,
            (
                "candidate_search_space_exceeded:"
                f"{len(candidates)}>{MAX_ADJUSTMENT_CANDIDATES}",
            ),
            assumptions,
        )

    backdoor_graph = graph.copy()
    backdoor_graph.remove_edges_from(list(backdoor_graph.out_edges(spec.treatment)))
    mandatory = set(conditioning)
    minimal_sets: list[tuple[str, ...]] = []

    for size in range(len(candidates) + 1):
        for candidate_set in combinations(candidates, size):
            candidate_names = set(candidate_set)
            if any(set(accepted).issubset(candidate_names) for accepted in minimal_sets):
                continue
            if nx.is_d_separator(
                backdoor_graph,
                {spec.treatment},
                {spec.outcome},
                mandatory | candidate_names,
            ):
                minimal_sets.append(candidate_set)

    if not minimal_sets:
        reasons = ["no_observed_backdoor_adjustment_set"]
        reasons.extend(
            warning
            for warning in graph_audit.warnings
            if warning.startswith("latent_common_cause:")
        )
        return _result(
            IdentificationStatus.NON_IDENTIFIABLE,
            estimand,
            conditioning,
            tuple(reasons),
            assumptions,
        )

    adjustment_sets = tuple(minimal_sets)
    expression = _expression(
        estimand,
        spec.treatment,
        spec.outcome,
        adjustment_sets[0],
        conditioning,
    )
    return IdentificationResult(
        status=IdentificationStatus.IDENTIFIED,
        estimand=estimand,
        method="backdoor_adjustment",
        adjustment_sets=adjustment_sets,
        conditioning_variables=conditioning,
        expression=expression,
        reasons=(
            "backdoor_criterion_satisfied",
            "identification_conditional_on_declared_dag",
        ),
        assumptions=assumptions,
    )


def _validate_conditioning(
    spec: CausalDAG,
    estimand: str,
    conditioning: tuple[str, ...],
) -> tuple[str, ...]:
    if estimand == "CATE" and not conditioning:
        return ("cate_requires_conditioning_variables",)
    if estimand != "CATE" and conditioning:
        return ("conditioning_variables_require_cate",)

    variables = {variable.name: variable for variable in spec.variables}
    graph = nx.DiGraph(spec.edges)
    graph.add_nodes_from(variables)
    descendants = nx.descendants(graph, spec.treatment)
    errors: list[str] = []
    for name in conditioning:
        variable = variables.get(name)
        if name in {spec.treatment, spec.outcome}:
            errors.append(f"invalid_conditioning_target:{name}")
        elif variable is None:
            errors.append(f"unknown_conditioning_variable:{name}")
        elif not variable.observed:
            errors.append(f"unobserved_conditioning_variable:{name}")
        elif name in descendants:
            errors.append(f"post_treatment_conditioning:{name}")
        elif variable.role is VariableRole.COLLIDER:
            errors.append(f"collider_conditioning:{name}")
    return tuple(errors)


def _expression(
    estimand: str,
    treatment: str,
    outcome: str,
    adjustment: tuple[str, ...],
    conditioning: tuple[str, ...],
) -> str:
    z = ",".join(adjustment)
    v = ",".join(conditioning)
    suffix = f",{z}" if z else ""
    condition_suffix = f",{v}" if v else ""
    contrast = (
        f"E[{outcome}|{treatment}=1{suffix}{condition_suffix}]"
        f"-E[{outcome}|{treatment}=0{suffix}{condition_suffix}]"
    )
    if estimand == "ATE":
        return f"E_{{{z}}}[{contrast}]" if z else contrast
    if estimand == "ATT":
        return f"E_{{{z}|{treatment}=1}}[{contrast}]" if z else contrast
    return f"E_{{{z}|{v}}}[{contrast}]" if z else contrast


def _result(
    status: IdentificationStatus,
    estimand: str,
    conditioning: tuple[str, ...],
    reasons: tuple[str, ...],
    assumptions: tuple[str, ...],
) -> IdentificationResult:
    return IdentificationResult(
        status=status,
        estimand=estimand,
        method=None,
        adjustment_sets=(),
        conditioning_variables=conditioning,
        expression=None,
        reasons=reasons,
        assumptions=assumptions,
    )
