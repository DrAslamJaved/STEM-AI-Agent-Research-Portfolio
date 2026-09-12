"""Phase 08R runner. Execute only with --execute; writes no output otherwise."""
import argparse, heapq, json, math, random, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fuzzy_similarity import Parameters, fuzzy_cosine, fuzzy_dice, fuzzy_jaccard, rational_similarity

CENTRES = (
    (.85,.85,.15,.15,.85,.85,.15,.15,.50,.50),
    (.85,.15,.85,.15,.85,.15,.85,.15,.50,.50),
    (.85,.15,.15,.85,.15,.85,.85,.15,.50,.50),
)
METHODS = {
    "theta1": lambda u,v: rational_similarity(u,v,Parameters(1,0,0,0,0,1,1,1,0,1)),
    "jaccard": fuzzy_jaccard, "dice": fuzzy_dice, "cosine": fuzzy_cosine,
}

def adjusted_rand_index(a, b):
    n=len(a); total=n*(n-1)//2
    ca=Counter(a); cb=Counter(b); cab=Counter(zip(a,b))
    choose2=lambda x: x*(x-1)//2
    index=sum(choose2(x) for x in cab.values())
    expected=sum(choose2(x) for x in ca.values())*sum(choose2(x) for x in cb.values())/total
    maximum=(sum(choose2(x) for x in ca.values())+sum(choose2(x) for x in cb.values()))/2
    return 1.0 if maximum==expected and a==b else (index-expected)/(maximum-expected)

def average_linkage(sim, k=3):
    n=len(sim); active={i:1 for i in range(n)}; members={i:[i] for i in range(n)}; scores={}; heap=[]
    def add(i,j,value):
        key=(min(i,j),max(i,j)); scores[key]=value; heapq.heappush(heap,(-value,key[0],key[1]))
    for i in range(n):
        for j in range(i+1,n): add(i,j,sim[i][j])
    next_id=n
    while len(active)>k:
        while True:
            neg,i,j=heapq.heappop(heap); key=(i,j)
            if i in active and j in active and scores.get(key)==-neg: break
        si,sj=active.pop(i),active.pop(j); new=next_id; next_id+=1
        members[new]=members.pop(i)+members.pop(j); active[new]=si+sj
        for other, so in list(active.items()):
            if other==new: continue
            vi=scores[(min(i,other),max(i,other))]; vj=scores[(min(j,other),max(j,other))]
            add(new,other,(si*vi+sj*vj)/(si+sj))
    labels=[0]*n
    for label, group in enumerate(members.values()):
        for item in group: labels[item]=label
    return labels

def dataset(seed):
    rng=random.Random(seed); x=[]; y=[]
    for label, centre in enumerate(CENTRES):
        for _ in range(100):
            x.append([min(1.0,max(0.0,z+rng.uniform(-.08,.08))) for z in centre]); y.append(label)
    return x,y

def one_replication(seed, name, fun):
    x,y=dataset(seed); n=len(x); sim=[[1.0]*n for _ in range(n)]; within=[]; between=[]
    for i in range(n):
        for j in range(i+1,n):
            value=fun(x[i],x[j]); sim[i][j]=sim[j][i]=value
            (within if y[i]==y[j] else between).append(value)
    pred=average_linkage(sim); w=sum(within)/len(within); b=sum(between)/len(between)
    return {"seed":seed,"ari":adjusted_rand_index(y,pred),"within_mean":w,"between_mean":b,
            "separation_gap":w-b,"cluster_sizes":[pred.count(i) for i in range(3)]}

def summary(rows):
    def stats(key):
        values=[row[key] for row in rows]; mean=sum(values)/len(values)
        return {"mean":mean,"std":math.sqrt(sum((v-mean)**2 for v in values)/len(values)),"min":min(values),"max":max(values)}
    return {key:stats(key) for key in ("ari","within_mean","between_mean","separation_gap")}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--execute",action="store_true"); p.add_argument("--output",default="results/phase_08_revised_clustering.json"); args=p.parse_args()
    if not args.execute: raise SystemExit("Refusing execution without --execute.")
    seeds=list(range(20260909,20260939)); methods={}
    for name,fun in METHODS.items():
        rows=[one_replication(seed,name,fun) for seed in seeds]; methods[name]={"summary":summary(rows),"replications":rows}
    out={"status":"executed","protocol":"phase_08_revised_application_protocol","n_replications":30,"seeds":seeds,"n_samples_per_replication":300,"dimension":10,"methods":methods}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2),encoding="utf-8"); print(path)
if __name__ == "__main__": main()
