from stem_research_agent.data_validation import validate_dti_records, validate_fuzzy_membership_vector

def test_valid_dti_records_pass() -> None:
    report = validate_dti_records([{"drug_id": "D1", "target_id": "T1", "label": 1},
                                   {"drug_id": "D2", "target_id": "T2", "label": 0}])
    assert report.is_valid
    assert report.metrics["n_unique_drugs"] == 2

def test_invalid_dti_records_are_rejected() -> None:
    report = validate_dti_records([{"drug_id": "D1", "target_id": "", "label": 4}])
    assert not report.is_valid
    assert any(issue.code == "missing_required_field" for issue in report.issues)

def test_duplicate_pairs_warn_without_mutation() -> None:
    report = validate_dti_records([{"drug_id": "D1", "target_id": "T1", "label": 1},
                                   {"drug_id": "D1", "target_id": "T1", "label": 0}])
    assert report.is_valid
    assert report.metrics["n_duplicate_pairs"] == 1

def test_memberships_in_closed_unit_interval_pass() -> None:
    report = validate_fuzzy_membership_vector([0.0, 0.2, 1.0])
    assert report.is_valid
    assert report.metrics["cardinality"] == 1.2

def test_invalid_memberships_are_rejected() -> None:
    report = validate_fuzzy_membership_vector([0.5, -0.1, 1.2])
    assert not report.is_valid
    assert [issue.code for issue in report.issues] == ["membership_out_of_bounds", "membership_out_of_bounds"]
