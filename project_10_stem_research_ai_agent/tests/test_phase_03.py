from stem_research_agent.dti_baselines import cold_drug_split, prevalence_score, seeded_random_split, split_overlap
from stem_research_agent.evaluation import evaluate_binary_predictions
from stem_research_agent.fuzzy_similarity import cardinality_balance_similarity, cosine_similarity, dice_similarity, jaccard_similarity

RECORDS = [{"drug_id":"D1","target_id":"T1","label":1},{"drug_id":"D1","target_id":"T2","label":0},
           {"drug_id":"D2","target_id":"T1","label":0},{"drug_id":"D2","target_id":"T2","label":1}]

def test_metrics_and_auc_are_perfect():
    result = evaluate_binary_predictions([0,1,0,1], [.1,.9,.2,.8])
    assert result.f1 == result.roc_auc == result.pr_auc == 1.0

def test_seeded_split_is_reproducible():
    assert seeded_random_split(RECORDS, test_fraction=.5, seed=7) == seeded_random_split(RECORDS, test_fraction=.5, seed=7)

def test_cold_drug_split_has_no_overlap():
    train, test = cold_drug_split(RECORDS, held_out_drugs={"D2"})
    assert split_overlap(train, test)["shared_drugs"] == 0

def test_prevalence_baseline():
    assert prevalence_score(RECORDS) == .5

def test_fuzzy_references_are_symmetric_bounded_and_reflexive():
    left, right = [.1,.7,1], [.4,.6,.2]
    for function in [jaccard_similarity, dice_similarity, cosine_similarity, cardinality_balance_similarity]:
        assert function(left,right) == function(right,left)
        assert 0 <= function(left,right) <= 1
        assert function(left,left) == 1
