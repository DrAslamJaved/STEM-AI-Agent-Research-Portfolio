from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import causal_audit_agent.orchestrator as orchestration
from causal_audit_agent.advanced_estimation import (
    AdvancedEffectEstimate,
    AdvancedEstimationStatus,
)
from causal_audit_agent.benchmarking import CausalMetrics, EvaluationStatus
from causal_audit_agent.causal_graph import (
    CausalDAG,
    GraphAudit,
    Variable,
    VariableRole,
)
from causal_audit_agent.contracts import (
    AnalysisState,
    CausalQuestion,
    QuestionType,
)
from causal_audit_agent.diagnostics import DiagnosticReport, DiagnosticStatus
from causal_audit_agent.estimation import EffectEstimate, EstimationStatus
from causal_audit_agent.identification import (
    IdentificationResult,
    IdentificationStatus,
)
from causal_audit_agent.query_parser import Classification
from causal_audit_agent.refutation import RefutationReport, RefutationStatus
from causal_audit_agent.reporting import (
    _serialize,
    analysis_run_to_dict,
    render_json,
    render_markdown,
)
from causal_audit_agent.sensitivity import SensitivityReport, SensitivityStatus


def _question(text: str = "What is the causal effect of T on Y?") -> CausalQuestion:
    return CausalQuestion(text, "T", "Y")


def _graph() -> CausalDAG:
    return CausalDAG(
        treatment="T",
        outcome="Y",
        variables=(
            Variable("X", VariableRole.CONFOUNDER, True, 0),
            Variable("T", VariableRole.TREATMENT, True, 1),
            Variable("Y", VariableRole.OUTCOME, True, 2),
        ),
        edges=(("X", "T"), ("X", "Y"), ("T", "Y")),
        assumptions=("No unmeasured confounding",),
        proposed_adjustment_set=("X",),
    )


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "X": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            "T": [0, 0, 1, 0, 1, 1],
            "Y": [0.0, 0.2, 1.1, 0.4, 1.6, 2.0],
        }
    )


def _classification(
    label: QuestionType = QuestionType.CAUSAL,
    *,
    review: bool = False,
) -> Classification:
    return Classification(label, 0.95, ("causal effect",), review)


def _graph_audit(valid: bool = True) -> GraphAudit:
    return GraphAudit(
        valid_for_identification_review=valid,
        errors=() if valid else ("DAG contains an invalid causal structure",),
        warnings=(),
        missing_assumptions=(),
        requires_human_review=True,
    )


def _identification(
    status: IdentificationStatus = IdentificationStatus.IDENTIFIED,
) -> IdentificationResult:
    return IdentificationResult(
        status=status,
        estimand="ATE",
        method="backdoor" if status is IdentificationStatus.IDENTIFIED else None,
        adjustment_sets=(("X",),) if status is IdentificationStatus.IDENTIFIED else (),
        conditioning_variables=(),
        expression="E[E[Y|T=1,X]-E[Y|T=0,X]]"
        if status is IdentificationStatus.IDENTIFIED
        else None,
        reasons=(),
        assumptions=("Conditional exchangeability",),
        requires_human_review=True,
    )


def _estimate(
    status: EstimationStatus = EstimationStatus.ESTIMATED,
) -> EffectEstimate:
    estimated = status is EstimationStatus.ESTIMATED
    return EffectEstimate(
        status=status,
        estimand="ATE",
        method="g_computation",
        estimate=1.25 if estimated else None,
        standard_error=0.1 if estimated else None,
        confidence_interval=(1.05, 1.45) if estimated else None,
        n_observations=6,
        adjustment_set=("X",),
        diagnostics={},
        reasons=(),
        requires_human_review=True,
    )


def _advanced(
    status: AdvancedEstimationStatus = AdvancedEstimationStatus.ESTIMATED,
) -> AdvancedEffectEstimate:
    estimated = status is AdvancedEstimationStatus.ESTIMATED
    return AdvancedEffectEstimate(
        status=status,
        estimand="ATE",
        method="cross_fitted_aipw",
        estimate=1.2 if estimated else None,
        standard_error=0.12 if estimated else None,
        confidence_interval=(0.96, 1.44) if estimated else None,
        n_observations=6,
        adjustment_set=("X",),
        conditioning_variables=(),
        unit_effects=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5) if estimated else None,
        diagnostics={},
        reasons=(),
        requires_human_review=True,
    )


def _diagnostics(
    status: DiagnosticStatus = DiagnosticStatus.PASS,
) -> DiagnosticReport:
    return DiagnosticReport(
        max_abs_smd=0.1,
        unweighted_max_abs_smd=0.2,
        overlap_fraction=1.0,
        effective_sample_size=5.8,
        effective_sample_fraction=0.97,
        max_weight=2.2,
        extreme_weight_fraction=0.0,
        clipped_propensity_fraction=0.0,
        covariate_smds={"X": 0.2},
        weighted_covariate_smds={"X": 0.1},
        estimand="ATE",
        n_observations=6,
        status=status,
        reasons=(),
        requires_human_review=True,
    )


def _refutation(
    status: RefutationStatus = RefutationStatus.PASS,
) -> RefutationReport:
    return RefutationReport(
        status=status,
        method="g_computation",
        estimand="ATE",
        baseline_estimate=1.25,
        checks=(),
        reasons=(),
        random_state=0,
        requires_human_review=True,
    )


def _sensitivity(
    status: SensitivityStatus = SensitivityStatus.ROBUST,
) -> SensitivityReport:
    return SensitivityReport(
        status=status,
        analysis="linear_ovb",
        estimand="ATE",
        method="g_computation",
        observed_estimate=1.25,
        null_value=0.0,
        standard_error=0.1,
        degrees_of_freedom=4,
        robustness_value=0.2,
        e_value=None,
        confidence_interval_e_value=None,
        scenarios=(),
        reasons=(),
        requires_human_review=True,
    )


def _metrics() -> CausalMetrics:
    return CausalMetrics(
        status=EvaluationStatus.COMPLETE,
        benchmark="IHDP",
        n_observations=6,
        estimated_ate=1.25,
        reference_ate=1.2,
        ate_bias=0.05,
        ate_error=0.05,
        pehe=0.15,
        confidence_interval=(1.05, 1.45),
        ci_covered=True,
        reasons=(),
        requires_human_review=True,
    )


def _patch_until_identification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestration, "classify_question", lambda _: _classification())
    monkeypatch.setattr(orchestration, "audit_causal_dag", lambda _: _graph_audit())
    monkeypatch.setattr(
        orchestration,
        "identify_estimand",
        lambda *_: _identification(),
    )


def _patch_full_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    advanced_status: AdvancedEstimationStatus = AdvancedEstimationStatus.ESTIMATED,
    diagnostic_status: DiagnosticStatus = DiagnosticStatus.PASS,
    refutation_status: RefutationStatus = RefutationStatus.PASS,
    sensitivity_status: SensitivityStatus = SensitivityStatus.ROBUST,
) -> None:
    _patch_until_identification(monkeypatch)
    monkeypatch.setattr(orchestration, "estimate_effect", lambda *_: _estimate())
    monkeypatch.setattr(
        orchestration,
        "estimate_advanced_effect",
        lambda *_: _advanced(advanced_status),
    )
    monkeypatch.setattr(
        orchestration,
        "diagnose",
        lambda *args, **kwargs: _diagnostics(diagnostic_status),
    )
    monkeypatch.setattr(
        orchestration,
        "run_refutation_suite",
        lambda *_: _refutation(refutation_status),
    )
    monkeypatch.setattr(
        orchestration,
        "analyze_hidden_confounding",
        lambda *_: _sensitivity(sensitivity_status),
    )
    monkeypatch.setattr(orchestration, "evaluate_predictions", lambda *args, **kwargs: _metrics())


def test_invalid_question_fails_closed():
    run = orchestration.run_analysis(CausalQuestion("", "T", "Y"), _graph(), _data())
    assert run.state is AnalysisState.INVALID_CAUSAL_QUERY
    assert run.estimate is None
    assert run.classification is None
    assert "No numerical causal estimate" in render_markdown(run)


def test_predictive_question_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        orchestration,
        "classify_question",
        lambda _: _classification(QuestionType.PREDICTIVE),
    )
    run = orchestration.run_analysis(_question(), _graph(), _data())
    assert run.state is AnalysisState.INVALID_CAUSAL_QUERY
    assert run.identification is None


def test_ambiguous_question_requires_review(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        orchestration,
        "classify_question",
        lambda _: _classification(QuestionType.AMBIGUOUS, review=True),
    )
    run = orchestration.run_analysis(_question(), _graph(), _data())
    assert run.state is AnalysisState.HUMAN_REVIEW_REQUIRED
    assert run.estimate is None


def test_invalid_dag_stops_before_identification(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(orchestration, "classify_question", lambda _: _classification())
    monkeypatch.setattr(orchestration, "audit_causal_dag", lambda _: _graph_audit(False))
    run = orchestration.run_analysis(_question(), _graph(), _data())
    assert run.state is AnalysisState.INSUFFICIENT_ASSUMPTIONS
    assert run.identification is None
    assert run.graph_audit is not None


@pytest.mark.parametrize(
    ("status", "state"),
    [
        (IdentificationStatus.NON_IDENTIFIABLE, AnalysisState.NON_IDENTIFIABLE),
        (
            IdentificationStatus.INSUFFICIENT_ASSUMPTIONS,
            AnalysisState.INSUFFICIENT_ASSUMPTIONS,
        ),
        (IdentificationStatus.UNSUPPORTED_QUERY, AnalysisState.INVALID_CAUSAL_QUERY),
    ],
)
def test_identification_gate_emits_no_estimate(
    monkeypatch: pytest.MonkeyPatch,
    status: IdentificationStatus,
    state: AnalysisState,
):
    monkeypatch.setattr(orchestration, "classify_question", lambda _: _classification())
    monkeypatch.setattr(orchestration, "audit_causal_dag", lambda _: _graph_audit())
    monkeypatch.setattr(
        orchestration,
        "identify_estimand",
        lambda *_: _identification(status),
    )
    run = orchestration.run_analysis(_question(), _graph(), _data())
    assert run.state is state
    assert run.estimate is None
    assert "No numerical causal estimate" in render_markdown(run)


def test_failed_estimation_is_not_estimable(monkeypatch: pytest.MonkeyPatch):
    _patch_until_identification(monkeypatch)
    monkeypatch.setattr(
        orchestration,
        "estimate_effect",
        lambda *_: _estimate(EstimationStatus.POSITIVITY_VIOLATION),
    )
    run = orchestration.run_analysis(_question(), _graph(), _data())
    assert run.state is AnalysisState.IDENTIFIED_BUT_NOT_ESTIMABLE
    assert run.estimate is not None
    assert run.estimate.estimate is None


def test_complete_robust_run_and_reports(monkeypatch: pytest.MonkeyPatch):
    _patch_full_pipeline(monkeypatch)
    run = orchestration.run_analysis(
        _question(),
        _graph(),
        _data(),
        propensity=np.full(6, 0.5),
        benchmark=object(),
    )
    assert run.state is AnalysisState.IDENTIFIED_AND_ROBUST
    assert run.advanced_estimate is not None
    assert run.diagnostics is not None
    assert run.refutation is not None
    assert run.sensitivity is not None
    assert run.benchmark_metrics is not None
    assert run.audit_trail[-1].stage is orchestration.PipelineStage.DECISION

    payload = analysis_run_to_dict(run)
    assert payload["decision"]["numerical_causal_conclusion_emitted"] is True
    assert payload["benchmark"]["pehe"] == pytest.approx(0.15)
    assert json.loads(render_json(run))["decision"]["state"] == "IDENTIFIED_AND_ROBUST"

    report = render_markdown(run)
    for heading in (
        "## Identification",
        "## Estimation",
        "## Advanced estimation",
        "## Overlap, balance, and weight diagnostics",
        "## Refutation and stress testing",
        "## Hidden-confounding sensitivity",
        "## Benchmark evaluation",
        "## Audit trail",
    ):
        assert heading in report


def test_diagnostic_review_marks_run_not_estimable(monkeypatch: pytest.MonkeyPatch):
    _patch_full_pipeline(monkeypatch, diagnostic_status=DiagnosticStatus.REVIEW)
    run = orchestration.run_analysis(
        _question(), _graph(), _data(), propensity=np.full(6, 0.5)
    )
    assert run.state is AnalysisState.IDENTIFIED_BUT_NOT_ESTIMABLE


@pytest.mark.parametrize(
    "overrides",
    [
        {"advanced_status": AdvancedEstimationStatus.POSITIVITY_VIOLATION},
        {"refutation_status": RefutationStatus.REVIEW},
        {"sensitivity_status": SensitivityStatus.SENSITIVE},
    ],
)
def test_fragility_states(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]):
    _patch_full_pipeline(monkeypatch, **overrides)
    run = orchestration.run_analysis(
        _question(), _graph(), _data(), propensity=np.full(6, 0.5)
    )
    assert run.state is AnalysisState.IDENTIFIED_BUT_FRAGILE


def test_optional_stages_can_be_skipped(monkeypatch: pytest.MonkeyPatch):
    _patch_until_identification(monkeypatch)
    monkeypatch.setattr(orchestration, "estimate_effect", lambda *_: _estimate())
    config = orchestration.AnalysisConfig(
        advanced_estimation=None,
        diagnostic_thresholds=None,
        refutation=None,
        sensitivity=None,
    )
    run = orchestration.run_analysis(_question(), _graph(), _data(), config=config)
    assert run.state is AnalysisState.HUMAN_REVIEW_REQUIRED
    report = render_markdown(run)
    assert "Advanced estimation was not requested" in report
    assert "Diagnostics were not run" in report
    assert "Refutation tests were not run" in report
    assert "Sensitivity analysis was not run" in report
    assert "No benchmark dataset" in report


def test_propensity_helper_supports_unadjusted_analysis():
    scores = orchestration._propensity_scores(_data(), "T", (), 0)
    assert np.allclose(scores, 0.5)


def test_propensity_helper_fits_adjusted_model():
    scores = orchestration._propensity_scores(_data(), "T", ("X",), 0)
    assert scores.shape == (6,)
    assert np.all((scores > 0.0) & (scores < 1.0))


def test_propensity_helper_rejects_one_treatment_group():
    data = _data().assign(T=1)
    with pytest.raises(ValueError, match="both binary treatment groups"):
        orchestration._propensity_scores(data, "T", ("X",), 0)


def test_serializer_rejects_unknown_component():
    with pytest.raises(TypeError, match="not serializable"):
        _serialize(object())
