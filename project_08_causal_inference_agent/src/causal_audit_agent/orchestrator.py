from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from causal_audit_agent.advanced_estimation import (
    AdvancedEffectEstimate,
    AdvancedEstimationRequest,
    AdvancedEstimationStatus,
    estimate_advanced_effect,
)
from causal_audit_agent.benchmarking import (
    BenchmarkDataset,
    CausalMetrics,
    evaluate_predictions,
)
from causal_audit_agent.causal_graph import CausalDAG, GraphAudit, audit_causal_dag
from causal_audit_agent.contracts import (
    AnalysisState,
    CausalQuestion,
    QuestionType,
)
from causal_audit_agent.diagnostics import (
    DiagnosticReport,
    DiagnosticStatus,
    DiagnosticThresholds,
    diagnose,
)
from causal_audit_agent.estimation import (
    EffectEstimate,
    EstimationRequest,
    EstimationStatus,
    estimate_effect,
)
from causal_audit_agent.identification import (
    IdentificationRequest,
    IdentificationResult,
    IdentificationStatus,
    identify_estimand,
)
from causal_audit_agent.query_parser import Classification, classify_question
from causal_audit_agent.refutation import (
    RefutationReport,
    RefutationRequest,
    RefutationStatus,
    run_refutation_suite,
)
from causal_audit_agent.sensitivity import (
    SensitivityReport,
    SensitivityRequest,
    SensitivityStatus,
    analyze_hidden_confounding,
)


class PipelineStage(str, Enum):
    QUESTION = "question"
    GRAPH_AUDIT = "graph_audit"
    IDENTIFICATION = "identification"
    ESTIMATION = "estimation"
    ADVANCED_ESTIMATION = "advanced_estimation"
    DIAGNOSTICS = "diagnostics"
    REFUTATION = "refutation"
    SENSITIVITY = "sensitivity"
    BENCHMARK = "benchmark"
    DECISION = "decision"


@dataclass(frozen=True)
class AuditEvent:
    stage: PipelineStage
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage.value,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AnalysisConfig:
    identification: IdentificationRequest = field(default_factory=IdentificationRequest)
    estimation: EstimationRequest = field(default_factory=EstimationRequest)
    advanced_estimation: AdvancedEstimationRequest | None = field(
        default_factory=AdvancedEstimationRequest
    )
    diagnostic_thresholds: DiagnosticThresholds | None = field(
        default_factory=DiagnosticThresholds
    )
    refutation: RefutationRequest | None = field(default_factory=RefutationRequest)
    sensitivity: SensitivityRequest | None = field(default_factory=SensitivityRequest)


@dataclass(frozen=True)
class AnalysisRun:
    state: AnalysisState
    question: CausalQuestion
    classification: Classification | None
    graph_audit: GraphAudit | None
    identification: IdentificationResult | None
    estimate: EffectEstimate | None
    advanced_estimate: AdvancedEffectEstimate | None
    diagnostics: DiagnosticReport | None
    refutation: RefutationReport | None
    sensitivity: SensitivityReport | None
    benchmark_metrics: CausalMetrics | None
    audit_trail: tuple[AuditEvent, ...]
    warnings: tuple[str, ...]
    requires_human_review: bool = True

    def to_dict(self) -> dict[str, Any]:
        from causal_audit_agent.reporting import analysis_run_to_dict

        return analysis_run_to_dict(self)


def _event(stage: PipelineStage, status: object, detail: str) -> AuditEvent:
    value = getattr(status, "value", status)
    return AuditEvent(stage, str(value), detail)


def _blocked_run(
    *,
    state: AnalysisState,
    question: CausalQuestion,
    classification: Classification | None,
    graph_audit: GraphAudit | None = None,
    identification: IdentificationResult | None = None,
    audit_trail: Sequence[AuditEvent],
    warnings: Sequence[str],
) -> AnalysisRun:
    return AnalysisRun(
        state=state,
        question=question,
        classification=classification,
        graph_audit=graph_audit,
        identification=identification,
        estimate=None,
        advanced_estimate=None,
        diagnostics=None,
        refutation=None,
        sensitivity=None,
        benchmark_metrics=None,
        audit_trail=tuple(audit_trail),
        warnings=tuple(warnings),
    )


def _propensity_scores(
    data: pd.DataFrame,
    treatment: str,
    covariates: tuple[str, ...],
    random_state: int,
) -> np.ndarray:
    treated = data[treatment].to_numpy(dtype=int)
    if len(np.unique(treated)) != 2:
        raise ValueError("Treatment must contain both binary treatment groups")
    if not covariates:
        return np.full(len(data), float(treated.mean()), dtype=float)
    features = data.loc[:, list(covariates)].to_numpy(dtype=float)
    model = LogisticRegression(max_iter=2_000, random_state=random_state)
    model.fit(features, treated)
    return model.predict_proba(features)[:, 1]


def _identification_state(status: IdentificationStatus) -> AnalysisState:
    if status is IdentificationStatus.NON_IDENTIFIABLE:
        return AnalysisState.NON_IDENTIFIABLE
    if status is IdentificationStatus.INSUFFICIENT_ASSUMPTIONS:
        return AnalysisState.INSUFFICIENT_ASSUMPTIONS
    return AnalysisState.INVALID_CAUSAL_QUERY


def run_analysis(
    question: CausalQuestion,
    graph: CausalDAG,
    data: pd.DataFrame,
    *,
    config: AnalysisConfig | None = None,
    propensity: Sequence[float] | np.ndarray | None = None,
    benchmark: BenchmarkDataset | None = None,
) -> AnalysisRun:
    """Run the fail-closed causal workflow and retain an auditable stage trace."""

    active_config = config or AnalysisConfig()
    audit_trail: list[AuditEvent] = []
    warnings: list[str] = []

    try:
        question.validate()
        classification = classify_question(question.question)
    except (TypeError, ValueError) as exc:
        audit_trail.append(_event(PipelineStage.QUESTION, "INVALID", str(exc)))
        return _blocked_run(
            state=AnalysisState.INVALID_CAUSAL_QUERY,
            question=question,
            classification=None,
            audit_trail=audit_trail,
            warnings=(str(exc),),
        )

    audit_trail.append(
        _event(
            PipelineStage.QUESTION,
            classification.label,
            f"classifier confidence={classification.confidence:.3f}",
        )
    )
    if (
        question.question_type is not QuestionType.CAUSAL
        or classification.label in {QuestionType.PREDICTIVE, QuestionType.ASSOCIATIONAL}
    ):
        warning = "The submitted question is not an approved causal query"
        return _blocked_run(
            state=AnalysisState.INVALID_CAUSAL_QUERY,
            question=question,
            classification=classification,
            audit_trail=audit_trail,
            warnings=(warning,),
        )
    if classification.requires_human_review and classification.label is not QuestionType.CAUSAL:
        warning = "Question classification is ambiguous and requires human review"
        return _blocked_run(
            state=AnalysisState.HUMAN_REVIEW_REQUIRED,
            question=question,
            classification=classification,
            audit_trail=audit_trail,
            warnings=(warning,),
        )

    graph_audit = audit_causal_dag(graph)
    audit_trail.append(
        _event(
            PipelineStage.GRAPH_AUDIT,
            "PASS" if graph_audit.valid_for_identification_review else "BLOCKED",
            "; ".join(graph_audit.errors or graph_audit.warnings or ("DAG audit completed",)),
        )
    )
    warnings.extend(graph_audit.warnings)
    if not graph_audit.valid_for_identification_review:
        warnings.extend(graph_audit.errors)
        return _blocked_run(
            state=AnalysisState.INSUFFICIENT_ASSUMPTIONS,
            question=question,
            classification=classification,
            graph_audit=graph_audit,
            audit_trail=audit_trail,
            warnings=warnings,
        )

    identification = identify_estimand(graph, active_config.identification)
    audit_trail.append(
        _event(
            PipelineStage.IDENTIFICATION,
            identification.status,
            "; ".join(identification.reasons or ("Identification completed",)),
        )
    )
    warnings.extend(identification.reasons)
    if identification.status is not IdentificationStatus.IDENTIFIED:
        return _blocked_run(
            state=_identification_state(identification.status),
            question=question,
            classification=classification,
            graph_audit=graph_audit,
            identification=identification,
            audit_trail=audit_trail,
            warnings=warnings,
        )

    adjustment_set = (
        identification.adjustment_sets[0] if identification.adjustment_sets else ()
    )
    estimate = estimate_effect(
        data,
        question.treatment,
        question.outcome,
        identification,
        active_config.estimation,
        adjustment_set,
    )
    audit_trail.append(
        _event(
            PipelineStage.ESTIMATION,
            estimate.status,
            "; ".join(estimate.reasons or (f"method={estimate.method}",)),
        )
    )
    warnings.extend(estimate.reasons)
    if estimate.status is not EstimationStatus.ESTIMATED:
        return AnalysisRun(
            state=AnalysisState.IDENTIFIED_BUT_NOT_ESTIMABLE,
            question=question,
            classification=classification,
            graph_audit=graph_audit,
            identification=identification,
            estimate=estimate,
            advanced_estimate=None,
            diagnostics=None,
            refutation=None,
            sensitivity=None,
            benchmark_metrics=None,
            audit_trail=tuple(audit_trail),
            warnings=tuple(warnings),
        )

    advanced_estimate = None
    if active_config.advanced_estimation is not None:
        advanced_estimate = estimate_advanced_effect(
            data,
            question.treatment,
            question.outcome,
            identification,
            active_config.advanced_estimation,
            adjustment_set,
        )
        audit_trail.append(
            _event(
                PipelineStage.ADVANCED_ESTIMATION,
                advanced_estimate.status,
                "; ".join(
                    advanced_estimate.reasons
                    or (f"method={advanced_estimate.method}",)
                ),
            )
        )
        warnings.extend(advanced_estimate.reasons)

    diagnostic_report = None
    if active_config.diagnostic_thresholds is not None:
        scores = (
            np.asarray(propensity, dtype=float)
            if propensity is not None
            else _propensity_scores(
                data,
                question.treatment,
                adjustment_set,
                active_config.estimation.random_state,
            )
        )
        diagnostic_report = diagnose(
            data,
            question.treatment,
            adjustment_set,
            scores,
            estimand=identification.estimand,
            thresholds=active_config.diagnostic_thresholds,
        )
        audit_trail.append(
            _event(
                PipelineStage.DIAGNOSTICS,
                diagnostic_report.status,
                "; ".join(diagnostic_report.reasons or ("Diagnostics completed",)),
            )
        )
        warnings.extend(diagnostic_report.reasons)

    refutation_report = None
    if active_config.refutation is not None:
        refutation_report = run_refutation_suite(
            data,
            question.treatment,
            question.outcome,
            identification,
            active_config.estimation,
            active_config.refutation,
        )
        audit_trail.append(
            _event(
                PipelineStage.REFUTATION,
                refutation_report.status,
                "; ".join(refutation_report.reasons or ("Refutation completed",)),
            )
        )
        warnings.extend(refutation_report.reasons)

    sensitivity_report = None
    if active_config.sensitivity is not None:
        sensitivity_report = analyze_hidden_confounding(
            estimate,
            active_config.sensitivity,
        )
        audit_trail.append(
            _event(
                PipelineStage.SENSITIVITY,
                sensitivity_report.status,
                "; ".join(sensitivity_report.reasons or ("Sensitivity completed",)),
            )
        )
        warnings.extend(sensitivity_report.reasons)

    benchmark_metrics = None
    if benchmark is not None:
        unit_effects = (
            advanced_estimate.unit_effects
            if advanced_estimate is not None
            and advanced_estimate.status is AdvancedEstimationStatus.ESTIMATED
            else None
        )
        benchmark_metrics = evaluate_predictions(
            benchmark,
            estimated_ate=estimate.estimate,
            estimated_ite=unit_effects,
            confidence_interval=estimate.confidence_interval,
        )
        audit_trail.append(
            _event(
                PipelineStage.BENCHMARK,
                benchmark_metrics.status,
                "; ".join(benchmark_metrics.reasons or ("Benchmark evaluation completed",)),
            )
        )
        warnings.extend(benchmark_metrics.reasons)

    state = AnalysisState.HUMAN_REVIEW_REQUIRED
    if diagnostic_report is not None and diagnostic_report.status is DiagnosticStatus.REVIEW:
        state = AnalysisState.IDENTIFIED_BUT_NOT_ESTIMABLE
    elif (
        advanced_estimate is not None
        and advanced_estimate.status is not AdvancedEstimationStatus.ESTIMATED
    ):
        state = AnalysisState.IDENTIFIED_BUT_FRAGILE
    elif refutation_report is not None and refutation_report.status is not RefutationStatus.PASS:
        state = AnalysisState.IDENTIFIED_BUT_FRAGILE
    elif sensitivity_report is not None and sensitivity_report.status is not SensitivityStatus.ROBUST:
        state = AnalysisState.IDENTIFIED_BUT_FRAGILE
    elif all(
        component is not None
        for component in (diagnostic_report, refutation_report, sensitivity_report)
    ):
        state = AnalysisState.IDENTIFIED_AND_ROBUST

    audit_trail.append(
        _event(
            PipelineStage.DECISION,
            state,
            "Human approval remains mandatory before causal interpretation",
        )
    )
    return AnalysisRun(
        state=state,
        question=question,
        classification=classification,
        graph_audit=graph_audit,
        identification=identification,
        estimate=estimate,
        advanced_estimate=advanced_estimate,
        diagnostics=diagnostic_report,
        refutation=refutation_report,
        sensitivity=sensitivity_report,
        benchmark_metrics=benchmark_metrics,
        audit_trail=tuple(audit_trail),
        warnings=tuple(dict.fromkeys(warnings)),
    )
