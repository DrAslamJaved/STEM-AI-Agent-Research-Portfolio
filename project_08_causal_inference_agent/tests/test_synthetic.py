import numpy as np
import pytest

from causal_audit_agent.synthetic import ALLOWED_SCENARIOS, generate_synthetic


def test_generation_is_reproducible():
    a = generate_synthetic(seed=17)
    b = generate_synthetic(seed=17)
    assert a.data.equals(b.data)
    assert np.array_equal(a.true_ite, b.true_ite)
    assert a.true_ate == b.true_ate


@pytest.mark.parametrize("scenario", ALLOWED_SCENARIOS)
def test_every_declared_scenario_returns_valid_observed_data(scenario):
    generated = generate_synthetic(scenario, n=250, seed=11)
    assert generated.scenario == scenario
    assert generated.seed == 11
    assert generated.data.shape == (250, 5)
    assert list(generated.data) == ["X1", "X2", "T", "Y", "propensity"]
    assert set(generated.data["T"].unique()).issubset({0, 1})
    assert generated.data["propensity"].between(0.0, 1.0, inclusive="neither").all()
    assert generated.true_ite.shape == (250,)
    assert np.isfinite(generated.true_ite).all()
    assert np.isclose(generated.true_ate, generated.true_ite.mean())


def test_heterogeneous_effect_matches_ate():
    data = generate_synthetic("heterogeneous", n=500, seed=4)
    assert np.isclose(data.true_ate, data.true_ite.mean())


def test_null_effect_is_zero():
    generated = generate_synthetic("null")
    assert generated.true_ate == 0.0
    assert np.count_nonzero(generated.true_ite) == 0


def test_randomized_assignment_has_constant_half_propensity():
    generated = generate_synthetic("randomized", seed=3)
    assert np.all(generated.data["propensity"] == 0.5)


def test_hidden_confounder_is_not_exposed_to_the_estimator():
    generated = generate_synthetic("hidden_confounding", seed=3)
    assert "hidden_u" not in generated.data.columns
    assert "U" not in generated.data.columns


def test_positivity_stress_creates_many_extreme_propensities():
    generated = generate_synthetic("positivity_stress", n=5000, seed=8)
    propensity = generated.data["propensity"]
    extreme_share = ((propensity < 0.05) | (propensity > 0.95)).mean()
    assert extreme_share > 0.45


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"n": 49}, ValueError, "at least 50"),
        ({"n": 50.5}, TypeError, "integer"),
        ({"scenario": "unsupported"}, ValueError, "Unknown scenario"),
        ({"scenario": None}, TypeError, "string"),
        ({"seed": 1.5}, TypeError, "integer"),
    ],
)
def test_invalid_generation_request_fails_explicitly(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        generate_synthetic(**kwargs)
