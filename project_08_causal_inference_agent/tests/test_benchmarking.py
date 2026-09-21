from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import causal_audit_agent.benchmarking as module
from causal_audit_agent.benchmarking import (
    BenchmarkDataset,
    BenchmarkProvenance,
    CausalMetrics,
    EvaluationStatus,
    TruthLevel,
    evaluate_predictions,
    load_ihdp,
    load_lalonde,
    summarize_replications,
)


def ihdp_frame():
    return pd.DataFrame(
        {
            "t": [0, 1, 0, 1],
            "yf": [1.1, 3.2, 2.0, 4.1],
            "ycf": [3.0, 1.0, 4.0, 2.0],
            "mu0": [1.0, 1.2, 2.0, 2.1],
            "mu1": [3.0, 3.2, 4.0, 4.1],
            "x1": [0.1, 0.2, 0.3, 0.4],
            "x2": [1.0, 0.0, 1.0, 0.0],
        }
    )


def lalonde_frame():
    return pd.DataFrame(
        {
            "treat": [0, 1, 0, 1],
            "re78": [100.0, 150.0, 120.0, 180.0],
            "age": [20, 30, 40, 50],
            "educ": [10, 12, 9, 14],
        }
    )


def write_csv(tmp_path, name, frame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_load_ihdp_csv_derives_truth_and_provenance(tmp_path):
    path = write_csv(tmp_path, "ihdp.csv", ihdp_frame())
    dataset = load_ihdp(
        path,
        source_uri="https://example.test/ihdp.csv",
        version="replication-fixture-v1",
    )
    assert dataset.name == "IHDP"
    assert dataset.treatment == "t"
    assert dataset.outcome == "yf"
    assert dataset.covariates == ("x1", "x2")
    assert dataset.truth_level is TruthLevel.INDIVIDUAL
    assert dataset.true_ate == pytest.approx(2.0)
    np.testing.assert_allclose(dataset.individual_effect, [2.0, 2.0, 2.0, 2.0])
    assert dataset.reasons == ("counterfactual_means_available",)
    assert dataset.requires_human_review
    assert dataset.provenance.source_sha256 == sha256(path.read_bytes()).hexdigest()
    assert dataset.provenance.source_uri == "https://example.test/ihdp.csv"
    assert dataset.provenance.version == "replication-fixture-v1"
    assert dataset.provenance.replication == 0
    assert dataset.provenance.source_path == str(path.resolve())


def test_ihdp_metadata_is_json_compatible(tmp_path):
    dataset = load_ihdp(write_csv(tmp_path, "ihdp.csv", ihdp_frame()))
    payload = dataset.to_metadata()
    assert payload["truth_level"] == "INDIVIDUAL"
    assert payload["n_observations"] == 4
    assert payload["covariates"] == ["x1", "x2"]
    assert payload["individual_effect_available"]
    assert isinstance(payload["reasons"], list)
    assert payload["provenance"]["replication"] == 0


def test_load_ihdp_csv_with_explicit_columns(tmp_path):
    frame = ihdp_frame().rename(
        columns={"t": "T", "yf": "Y", "mu0": "M0", "mu1": "M1"}
    )
    path = write_csv(tmp_path, "custom.csv", frame)
    dataset = load_ihdp(
        path,
        treatment_column="T",
        outcome_column="Y",
        mu0_column="M0",
        mu1_column="M1",
        covariates=("x2", "x1"),
    )
    assert dataset.covariates == ("x2", "x1")


def test_load_multireplication_ihdp_npz(tmp_path):
    path = tmp_path / "ihdp.npz"
    x = np.stack(
        (
            np.arange(8, dtype=float).reshape(4, 2),
            np.arange(8, 16, dtype=float).reshape(4, 2),
        ),
        axis=2,
    )
    t = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=float)
    yf = np.array([[1, 10], [3, 20], [2, 30], [4, 40]], dtype=float)
    mu0 = np.array([[1, 8], [1, 18], [2, 28], [2, 38]], dtype=float)
    mu1 = np.array([[3, 10], [3, 20], [4, 30], [4, 40]], dtype=float)
    np.savez(path, x=x, t=t, yf=yf, mu0=mu0, mu1=mu1)
    dataset = load_ihdp(path, replication=1, version="two-replications")
    assert dataset.covariates == ("x1", "x2")
    np.testing.assert_allclose(dataset.data[["x1", "x2"]], x[:, :, 1])
    np.testing.assert_allclose(dataset.data["t"], t[:, 1])
    assert dataset.true_ate == pytest.approx(2.0)
    assert dataset.provenance.replication == 1


def test_load_ihdp_npz_transposes_replication_vectors(tmp_path):
    path = tmp_path / "transposed.npz"
    x = np.arange(8, dtype=float).reshape(4, 2, 1)
    t = np.array([[0, 1, 0, 1]], dtype=float)
    yf = np.array([[1, 3, 2, 4]], dtype=float)
    mu0 = np.array([[1, 1, 2, 2]], dtype=float)
    mu1 = np.array([[3, 3, 4, 4]], dtype=float)
    np.savez(path, x=x, t=t, yf=yf, mu0=mu0, mu1=mu1)
    assert load_ihdp(path).true_ate == pytest.approx(2.0)


def test_load_single_replication_ihdp_npz(tmp_path):
    path = tmp_path / "single.npz"
    np.savez(
        path,
        x=np.arange(8, dtype=float).reshape(4, 2),
        t=np.array([0, 1, 0, 1]),
        yf=np.array([1, 3, 2, 4]),
        mu0=np.array([1, 1, 2, 2]),
        mu1=np.array([3, 3, 4, 4]),
    )
    dataset = load_ihdp(path)
    assert dataset.data.shape == (4, 6)
    assert dataset.true_ate == 2.0


def test_load_lalonde_without_reference_preserves_truth_absence(tmp_path):
    path = write_csv(tmp_path, "lalonde.csv", lalonde_frame())
    dataset = load_lalonde(path, source_uri="local://lalonde", version="fixture-v1")
    assert dataset.name == "LALONDE"
    assert dataset.covariates == ("age", "educ")
    assert dataset.truth_level is TruthLevel.NONE
    assert dataset.true_ate is None
    assert dataset.individual_effect is None
    assert dataset.reasons == ("counterfactual_truth_unavailable",)
    assert dataset.provenance.replication is None


def test_load_lalonde_with_explicit_average_reference(tmp_path):
    dataset = load_lalonde(
        write_csv(tmp_path, "lalonde.csv", lalonde_frame()),
        covariates=("educ", "age"),
        reference_ate=25.0,
    )
    assert dataset.truth_level is TruthLevel.AVERAGE_REFERENCE
    assert dataset.true_ate == 25.0
    assert dataset.individual_effect is None
    assert dataset.reasons == (
        "experimental_average_effect_reference_supplied",
        "individual_truth_unavailable",
    )


def test_perfect_ihdp_predictions_have_zero_error_and_pehe(tmp_path):
    dataset = load_ihdp(write_csv(tmp_path, "ihdp.csv", ihdp_frame()))
    metrics = evaluate_predictions(
        dataset,
        estimated_ate=2.0,
        estimated_ite=[2.0, 2.0, 2.0, 2.0],
        confidence_interval=(1.8, 2.2),
    )
    assert metrics.status is EvaluationStatus.COMPLETE
    assert metrics.estimated_ate == 2.0
    assert metrics.reference_ate == 2.0
    assert metrics.ate_bias == 0.0
    assert metrics.ate_error == 0.0
    assert metrics.pehe == pytest.approx(0.0, abs=1e-12)
    assert metrics.confidence_interval == (1.8, 2.2)
    assert metrics.ci_covered is True
    assert metrics.reasons == ()
    assert metrics.requires_human_review


def test_ite_predictions_can_derive_ate_and_pehe(tmp_path):
    dataset = load_ihdp(write_csv(tmp_path, "ihdp.csv", ihdp_frame()))
    metrics = evaluate_predictions(
        dataset,
        estimated_ite=(value for value in [1.0, 2.0, 3.0, 2.0]),
        confidence_interval=(1.5, 2.5),
    )
    assert metrics.status is EvaluationStatus.COMPLETE
    assert metrics.estimated_ate == 2.0
    assert metrics.pehe == pytest.approx(np.sqrt(0.5))


def test_ihdp_without_ite_is_partial(tmp_path):
    dataset = load_ihdp(write_csv(tmp_path, "ihdp.csv", ihdp_frame()))
    metrics = evaluate_predictions(dataset, estimated_ate=2.2)
    assert metrics.status is EvaluationStatus.PARTIAL
    assert metrics.ate_bias == pytest.approx(0.2)
    assert metrics.ate_error == pytest.approx(0.2)
    assert metrics.pehe is None
    assert metrics.ci_covered is None
    assert metrics.reasons == (
        "pehe_unavailable:estimated_ite_not_supplied",
        "ci_coverage_not_requested",
    )


def test_noncovering_interval_is_reported(tmp_path):
    dataset = load_ihdp(write_csv(tmp_path, "ihdp.csv", ihdp_frame()))
    metrics = evaluate_predictions(
        dataset,
        estimated_ite=[3.0, 3.0, 3.0, 3.0],
        confidence_interval=(2.5, 3.5),
    )
    assert metrics.status is EvaluationStatus.COMPLETE
    assert metrics.ci_covered is False
    assert metrics.ate_error == 1.0
    assert metrics.pehe == 1.0


def test_lalonde_without_reference_has_insufficient_truth(tmp_path):
    dataset = load_lalonde(write_csv(tmp_path, "lalonde.csv", lalonde_frame()))
    metrics = evaluate_predictions(
        dataset, estimated_ate=20.0, confidence_interval=(10.0, 30.0)
    )
    assert metrics.status is EvaluationStatus.INSUFFICIENT_TRUTH
    assert metrics.reference_ate is None
    assert metrics.ate_error is None
    assert metrics.pehe is None
    assert metrics.ci_covered is None
    assert metrics.confidence_interval == (10.0, 30.0)
    assert metrics.reasons == ("average_effect_reference_unavailable",)


def test_lalonde_reference_supports_average_metrics_but_not_pehe(tmp_path):
    dataset = load_lalonde(
        write_csv(tmp_path, "lalonde.csv", lalonde_frame()), reference_ate=25.0
    )
    metrics = evaluate_predictions(
        dataset, estimated_ate=20.0, confidence_interval=(15.0, 30.0)
    )
    assert metrics.status is EvaluationStatus.PARTIAL
    assert metrics.ate_bias == -5.0
    assert metrics.ate_error == 5.0
    assert metrics.pehe is None
    assert metrics.ci_covered is True
    assert metrics.reasons == (
        "pehe_unavailable:individual_counterfactual_truth_absent",
    )


def test_metrics_serialization_is_json_compatible(tmp_path):
    dataset = load_ihdp(write_csv(tmp_path, "ihdp.csv", ihdp_frame()))
    payload = evaluate_predictions(dataset, estimated_ate=2.0).to_dict()
    assert payload["status"] == "PARTIAL"
    assert isinstance(payload["reasons"], list)
    assert payload["benchmark"] == "IHDP"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({}, "estimated_ate_or_ite_is_required"),
        ({"estimated_ate": np.nan}, "estimated_ate_must_be_finite"),
        ({"estimated_ate": True}, "estimated_ate_must_be_finite"),
        ({"estimated_ite": [1.0]}, "estimated_ite_length_mismatch"),
        ({"estimated_ite": [[1.0], [2.0], [3.0], [4.0]]}, "estimated_ite_length_mismatch"),
        ({"estimated_ite": [1.0, 2.0, np.inf, 4.0]}, "estimated_ite_must_be_finite"),
        ({"estimated_ite": ["a", "b", "c", "d"]}, "estimated_ite_must_be_numeric"),
        (
            {"estimated_ate": 5.0, "estimated_ite": [2.0, 2.0, 2.0, 2.0]},
            "estimated_ate_inconsistent_with_ite",
        ),
        ({"estimated_ate": 2.0, "confidence_interval": [1.0, 3.0]}, "confidence_interval_is_invalid"),
        ({"estimated_ate": 2.0, "confidence_interval": (1.0,)}, "confidence_interval_is_invalid"),
        ({"estimated_ate": 2.0, "confidence_interval": (np.nan, 3.0)}, "confidence_interval_is_invalid"),
        ({"estimated_ate": 2.0, "confidence_interval": (3.0, 1.0)}, "confidence_interval_is_invalid"),
    ],
)
def test_invalid_prediction_requests_return_status(tmp_path, kwargs, reason):
    dataset = load_ihdp(write_csv(tmp_path, "ihdp.csv", ihdp_frame()))
    metrics = evaluate_predictions(dataset, **kwargs)
    assert metrics.status is EvaluationStatus.INVALID_REQUEST
    assert reason in metrics.reasons
    assert metrics.ate_error is None


def test_benchmark_type_is_enforced():
    with pytest.raises(TypeError, match="benchmark must be a BenchmarkDataset"):
        evaluate_predictions({}, estimated_ate=1.0)


def test_replication_summary_calculates_all_metrics(tmp_path):
    dataset = load_ihdp(write_csv(tmp_path, "ihdp.csv", ihdp_frame()))
    results = [
        evaluate_predictions(
            dataset,
            estimated_ite=[2.0] * 4,
            confidence_interval=(1.0, 3.0),
        ),
        evaluate_predictions(
            dataset,
            estimated_ite=[3.0] * 4,
            confidence_interval=(2.5, 3.5),
        ),
    ]
    summary = summarize_replications(result for result in results)
    assert summary.n_results == 2
    assert summary.ate_metric_count == 2
    assert summary.pehe_metric_count == 2
    assert summary.coverage_metric_count == 2
    assert summary.mean_ate_error == 0.5
    assert summary.ate_rmse == pytest.approx(np.sqrt(0.5))
    assert summary.mean_pehe == pytest.approx(0.5)
    assert summary.confidence_interval_coverage == 0.5
    assert summary.reasons == ()
    payload = summary.to_dict()
    assert payload["reasons"] == []
    assert payload["n_results"] == 2


def test_replication_summary_tracks_unavailable_metrics(tmp_path):
    dataset = load_lalonde(write_csv(tmp_path, "lalonde.csv", lalonde_frame()))
    result = evaluate_predictions(dataset, estimated_ate=10.0)
    summary = summarize_replications([result])
    assert summary.ate_metric_count == 0
    assert summary.pehe_metric_count == 0
    assert summary.coverage_metric_count == 0
    assert summary.mean_ate_error is None
    assert summary.ate_rmse is None
    assert summary.mean_pehe is None
    assert summary.confidence_interval_coverage is None
    assert summary.reasons == (
        "ate_metrics_unavailable",
        "pehe_metrics_unavailable",
        "coverage_metrics_unavailable",
    )


def test_mixed_summary_uses_metric_specific_denominators(tmp_path):
    ihdp = load_ihdp(write_csv(tmp_path, "ihdp.csv", ihdp_frame()))
    first = evaluate_predictions(
        ihdp, estimated_ite=[2.0] * 4, confidence_interval=(1.0, 3.0)
    )
    second = evaluate_predictions(ihdp, estimated_ate=3.0)
    summary = summarize_replications([first, second])
    assert summary.ate_metric_count == 2
    assert summary.pehe_metric_count == 1
    assert summary.coverage_metric_count == 1
    assert summary.mean_ate_error == 0.5
    assert summary.reasons == ()


def test_summary_input_contracts():
    with pytest.raises(ValueError, match="at least one result"):
        summarize_replications([])
    with pytest.raises(TypeError, match="results must be an iterable"):
        summarize_replications(5)
    with pytest.raises(TypeError, match="every result"):
        summarize_replications(["bad"])


def test_summary_rejects_invalid_results_and_mixed_benchmarks(tmp_path):
    ihdp = load_ihdp(write_csv(tmp_path, "ihdp.csv", ihdp_frame()))
    lalonde = load_lalonde(
        write_csv(tmp_path, "lalonde.csv", lalonde_frame()), reference_ate=25.0
    )
    invalid = evaluate_predictions(ihdp)
    with pytest.raises(ValueError, match="invalid evaluation results"):
        summarize_replications([invalid])
    valid_ihdp = evaluate_predictions(ihdp, estimated_ate=2.0)
    valid_lalonde = evaluate_predictions(lalonde, estimated_ate=25.0)
    with pytest.raises(ValueError, match="cannot mix benchmarks"):
        summarize_replications([valid_ihdp, valid_lalonde])


@pytest.mark.parametrize("replication", [-1, True, 1.5])
def test_invalid_replication_values_are_rejected(tmp_path, replication):
    path = write_csv(tmp_path, "ihdp.csv", ihdp_frame())
    with pytest.raises(ValueError, match="non-negative integer"):
        load_ihdp(path, replication=replication)


def test_csv_rejects_nonzero_replication(tmp_path):
    path = write_csv(tmp_path, "ihdp.csv", ihdp_frame())
    with pytest.raises(ValueError, match="replication must be zero"):
        load_ihdp(path, replication=1)


def test_ihdp_rejects_unknown_extension(tmp_path):
    path = tmp_path / "ihdp.txt"
    path.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match=".csv or .npz"):
        load_ihdp(path)


def test_lalonde_rejects_non_csv(tmp_path):
    path = tmp_path / "lalonde.npz"
    np.savez(path, x=np.ones((2, 2)))
    with pytest.raises(ValueError, match="must be a .csv"):
        load_lalonde(path)


def test_missing_and_invalid_paths_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="was not found"):
        load_ihdp(tmp_path / "missing.csv")
    with pytest.raises(TypeError, match="path must be a string or Path"):
        load_ihdp(7)


def test_invalid_reference_ate_is_rejected(tmp_path):
    path = write_csv(tmp_path, "lalonde.csv", lalonde_frame())
    with pytest.raises(ValueError, match="reference_ate must be finite"):
        load_lalonde(path, reference_ate=np.inf)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(columns="x1"), "missing benchmark columns"),
        (lambda frame: frame.assign(x1=["a", "b", "c", "d"]), "must be numeric:x1"),
        (lambda frame: frame.assign(x1=[0.0, 1.0, np.nan, 2.0]), "must be finite:x1"),
        (lambda frame: frame.assign(t=[0, 0, 0, 0]), "both binary levels"),
        (lambda frame: frame.assign(t=[0, 1, 2, 1]), "both binary levels"),
    ],
)
def test_invalid_benchmark_columns_are_rejected(tmp_path, mutator, message):
    path = write_csv(tmp_path, "invalid.csv", mutator(ihdp_frame()))
    with pytest.raises(ValueError, match=message):
        load_ihdp(path, covariates=("x1", "x2"))


def test_treatment_outcome_and_covariates_must_be_distinct(tmp_path):
    path = write_csv(tmp_path, "ihdp.csv", ihdp_frame())
    with pytest.raises(ValueError, match="must be distinct"):
        load_ihdp(path, covariates=("t", "x1"))
    with pytest.raises(ValueError, match="must be distinct"):
        load_ihdp(path, treatment_column="yf")


def test_truth_columns_cannot_be_covariates(tmp_path):
    path = write_csv(tmp_path, "ihdp.csv", ihdp_frame())
    with pytest.raises(ValueError, match="truth columns cannot be used"):
        load_ihdp(path, covariates=("x1", "mu0"))
    with pytest.raises(ValueError, match="truth columns cannot be used"):
        load_ihdp(path, covariates=("x1", "ycf"))


@pytest.mark.parametrize(
    ("covariates", "message"),
    [
        ((), "at least one covariate"),
        ("x1", "iterable of column names"),
        (("x1", "x1"), "must be unique"),
        (("x1", ""), "non-empty strings"),
        (("x1", 2), "non-empty strings"),
    ],
)
def test_invalid_covariate_contracts_are_rejected(tmp_path, covariates, message):
    path = write_csv(tmp_path, "ihdp.csv", ihdp_frame())
    with pytest.raises((TypeError, ValueError), match=message):
        load_ihdp(path, covariates=covariates)


def test_npz_rejects_covariate_override(tmp_path):
    path = tmp_path / "single.npz"
    np.savez(
        path,
        x=np.ones((4, 2)),
        t=np.array([0, 1, 0, 1]),
        yf=np.ones(4),
        mu0=np.zeros(4),
        mu1=np.ones(4),
    )
    with pytest.raises(ValueError, match="cannot be overridden"):
        load_ihdp(path, covariates=("x1",))


def test_npz_requires_all_arrays(tmp_path):
    path = tmp_path / "missing.npz"
    np.savez(path, x=np.ones((4, 2)), t=np.array([0, 1, 0, 1]))
    with pytest.raises(ValueError, match="missing arrays"):
        load_ihdp(path)


@pytest.mark.parametrize("shape", [(4,), (4, 2, 1, 1)])
def test_npz_rejects_invalid_x_dimensions(tmp_path, shape):
    path = tmp_path / "badx.npz"
    np.savez(
        path,
        x=np.ones(shape),
        t=np.array([0, 1, 0, 1]),
        yf=np.ones(4),
        mu0=np.zeros(4),
        mu1=np.ones(4),
    )
    with pytest.raises(ValueError, match="x array must have two or three"):
        load_ihdp(path)


def test_npz_rejects_out_of_range_replication(tmp_path):
    path = tmp_path / "multi.npz"
    arrays = {
        "x": np.ones((4, 2, 1)),
        "t": np.ones((4, 1)),
        "yf": np.ones((4, 1)),
        "mu0": np.zeros((4, 1)),
        "mu1": np.ones((4, 1)),
    }
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="outside the IHDP NPZ range"):
        load_ihdp(path, replication=1)


def test_single_npz_rejects_nonzero_replication(tmp_path):
    path = tmp_path / "single.npz"
    np.savez(
        path,
        x=np.ones((4, 2)),
        t=np.array([0, 1, 0, 1]),
        yf=np.ones(4),
        mu0=np.zeros(4),
        mu1=np.ones(4),
    )
    with pytest.raises(ValueError, match="single-replication NPZ"):
        load_ihdp(path, replication=1)


def test_npz_vector_rejects_nonzero_replication(tmp_path):
    with pytest.raises(ValueError, match="single-replication array"):
        module._select_replication(np.ones(4), 1, 4)


def test_npz_vector_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="one or two dimensions"):
        module._select_replication(np.ones((2, 2, 2)), 0, 2)


def test_npz_vector_rejects_out_of_range_and_inconsistent_rows():
    with pytest.raises(ValueError, match="outside an IHDP array range"):
        module._select_replication(np.ones((4, 1)), 1, 4)
    with pytest.raises(ValueError, match="inconsistent observation counts"):
        module._select_replication(np.ones(3), 0, 4)


def test_provenance_requires_string_source_metadata(tmp_path):
    path = write_csv(tmp_path, "ihdp.csv", ihdp_frame())
    with pytest.raises(TypeError, match="must be strings"):
        load_ihdp(path, source_uri=7)
    with pytest.raises(TypeError, match="must be strings"):
        load_ihdp(path, version=7)


def test_configuration_and_protocol_are_present():
    root = Path(__file__).parents[1]
    configuration = yaml.safe_load(
        (root / "configs/benchmarks/default.yaml").read_text(encoding="utf-8")
    )
    assert configuration["ihdp"]["mu0_column"] == "mu0"
    assert configuration["evaluation"]["reject_lalonde_pehe"] is True
    protocol = (root / "docs/benchmark_protocol.md").read_text(encoding="utf-8")
    assert "must not be synthesized" in protocol
    assert "10.1198/jcgs.2010.08162" in protocol
    assert "LaLonde" in protocol


def test_provenance_serialization():
    provenance = BenchmarkProvenance("a", "b", "c", "d", 0)
    assert provenance.to_dict() == {
        "source_path": "a",
        "source_sha256": "b",
        "source_uri": "c",
        "version": "d",
        "replication": 0,
    }


def test_internal_mean_helper_covers_empty_and_nonempty_paths():
    assert module._mean_or_none([]) is None
    assert module._mean_or_none([True, False]) == 0.5
