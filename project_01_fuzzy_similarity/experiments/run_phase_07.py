"""Phase 07 runner. Execution requires the --execute safeguard."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fuzzy_similarity import Parameters, fuzzy_jaccard, fuzzy_dice, fuzzy_cosine, rational_similarity

def main():
 p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true");p.add_argument("--config",default="configs/phase_07.json");p.add_argument("--output",default="results/phase_07.json");a=p.parse_args()
 if not a.execute: raise SystemExit("Refusing execution without --execute.")
 c=json.loads(Path(a.config).read_text());r=random.Random(c["seed"]); rows=[]
 q=c.get("rational_parameters"); theta=Parameters(**q) if q else Parameters.jaccard_control()
 measures={"rational":lambda u,v:rational_similarity(u,v,theta),"jaccard":fuzzy_jaccard,"dice":fuzzy_dice,"cosine":fuzzy_cosine}
 for n in c["dimensions"]:
  for _ in range(c["random_pairs_per_dimension"]):
   u=[r.random() for _ in range(n)];v=[r.random() for _ in range(n)]
   rows.append({"n":n,**{k:f(u,v) for k,f in measures.items()}})
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps({"config":c,"rows":rows},indent=2));print(a.output)
if __name__=="__main__":main()
