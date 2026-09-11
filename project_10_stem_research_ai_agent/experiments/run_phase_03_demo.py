import json
from pathlib import Path
from stem_research_agent.dti_baselines import prevalence_score, seeded_random_split, split_overlap
from stem_research_agent.evaluation import evaluate_binary_predictions
from stem_research_agent.fuzzy_similarity import jaccard_similarity, dice_similarity, cosine_similarity, cardinality_balance_similarity

records = [{"drug_id":"D1","target_id":"T1","label":1},{"drug_id":"D1","target_id":"T2","label":0},
           {"drug_id":"D2","target_id":"T1","label":0},{"drug_id":"D2","target_id":"T2","label":1},
           {"drug_id":"D3","target_id":"T1","label":1},{"drug_id":"D3","target_id":"T2","label":0}]
train, test = seeded_random_split(records, test_fraction=1/3, seed=20260909)
score = prevalence_score(train)
result = {"seed":20260909, "model":"training_prevalence_baseline", "split_overlap":split_overlap(train,test),
          "metrics":evaluate_binary_predictions([row["label"] for row in test],[score]*len(test)).to_dict(),
          "fuzzy_similarity":{"jaccard":jaccard_similarity([.2,.8],[.4,.6]),"dice":dice_similarity([.2,.8],[.4,.6]),
          "cosine":cosine_similarity([.2,.8],[.4,.6]),"cardinality_balance":cardinality_balance_similarity([.2,.8],[.4,.6])}}
output = Path("results/phase_03_baseline_demo.json"); output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(output)
