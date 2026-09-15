from stem_research_agent.dti_splits import (
    cold_drug_split,
    cold_target_split,
    pair_random_split,
    split_diagnostics,
    validate_split_diagnostics,
)


def _records():
    return [
        {"drug_id": drug, "target_id": target, "label": int((drug, target) in {("D1", "T1"), ("D2", "T2")})}
        for drug in ("D1", "D2", "D3", "D4")
        for target in ("T1", "T2", "T3")
    ]


def test_pair_random_split_is_deterministic_and_has_no_pair_overlap():
    first = pair_random_split(_records(), test_fraction=0.25, seed=7)
    second = pair_random_split(_records(), test_fraction=0.25, seed=7)
    assert first == second
    diagnostic = split_diagnostics(*first, split_name="pair_random")
    assert diagnostic["shared_pair_count"] == 0
    assert diagnostic["shared_drug_count"] > 0
    assert diagnostic["shared_target_count"] > 0


def test_cold_drug_split_prevents_drug_overlap():
    train, test, held_out = cold_drug_split(_records(), test_fraction=0.25, seed=7)
    diagnostic = split_diagnostics(train, test, split_name="cold_drug")
    assert len(held_out) == 1
    assert diagnostic["shared_drug_count"] == 0
    assert diagnostic["shared_target_count"] == 3
    validate_split_diagnostics(diagnostic, cold_entity="drug")


def test_cold_target_split_prevents_target_overlap():
    train, test, held_out = cold_target_split(_records(), test_fraction=1 / 3, seed=7)
    diagnostic = split_diagnostics(train, test, split_name="cold_target")
    assert len(held_out) == 1
    assert diagnostic["shared_drug_count"] == 4
    assert diagnostic["shared_target_count"] == 0
    validate_split_diagnostics(diagnostic, cold_entity="target")


def test_validator_rejects_declared_cold_split_with_identity_overlap():
    train, test = pair_random_split(_records(), test_fraction=0.25, seed=7)
    diagnostic = split_diagnostics(train, test, split_name="incorrectly_declared_cold_drug")
    try:
        validate_split_diagnostics(diagnostic, cold_entity="drug")
    except ValueError as exc:
        assert "shares drug" in str(exc)
    else:
        raise AssertionError("Expected a cold-drug overlap failure.")
