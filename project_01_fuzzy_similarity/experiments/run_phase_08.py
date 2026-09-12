"""Approved Phase 08 synthetic clustering study; requires --execute."""
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fuzzy_similarity import Parameters,fuzzy_jaccard,fuzzy_dice,fuzzy_cosine,rational_similarity
def ari(a,b):
 n=len(a); pairs=lambda z:sum(v*(v-1)//2 for v in [list(z).count(q) for q in set(z)])
 both=sum(1 for i in range(n) for j in range(i) if a[i]==a[j] and b[i]==b[j]); ea=pairs(a); eb=pairs(b); total=n*(n-1)//2
 expected=ea*eb/total; denom=(ea+eb)/2-expected
 return 1.0 if denom==0 and a==b else (both-expected)/denom
def cluster(sim,k=3):
 groups=[{i} for i in range(len(sim))]
 while len(groups)>k:
  best=max(((sum(sim[i][j] for i in g for j in h)/(len(g)*len(h)),-min(g),-min(h),u,v) for u,g in enumerate(groups) for v,h in enumerate(groups) if u<v))
  u,v=best[-2:]; groups[u]|=groups[v];groups.pop(v)
 labels=[0]*len(sim)
 for q,g in enumerate(groups):
  for i in g: labels[i]=q
 return labels
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output',default='results/phase_08_clustering.json');a=p.parse_args()
 if not a.execute: raise SystemExit('Refusing execution without --execute.')
 r=random.Random(20260908); centres=([.15]*10,[.5]*10,[.85]*10); x=[]; y=[]
 for k,c in enumerate(centres):
  for _ in range(100): x.append([min(1,max(0,z+r.uniform(-.12,.12))) for z in c]);y.append(k)
 ms={'theta1':lambda u,v:rational_similarity(u,v,Parameters(1,0,0,0,0,1,1,1,0,1)),'jaccard':fuzzy_jaccard,'dice':fuzzy_dice,'cosine':fuzzy_cosine}
 out={'seed':20260908,'n_samples':300,'dimension':10,'status':'executed','methods':{}}
 for name,f in ms.items():
  within=[];between=[]; sim=[[1.0]*300 for _ in range(300)]
  for i in range(300):
   for j in range(i+1,300):
    value=f(x[i],x[j]);sim[i][j]=sim[j][i]=value;(within if y[i]==y[j] else between).append(value)
  pred=cluster(sim);wm=sum(within)/len(within);bm=sum(between)/len(between)
  out['methods'][name]={'within_mean':wm,'between_mean':bm,'separation_gap':wm-bm,'adjusted_rand_index':ari(y,pred),'cluster_sizes':[pred.count(q) for q in range(3)]}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2));print(a.output)
if __name__=='__main__':main()
