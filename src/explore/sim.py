"""Leave-benchmark-out simulation of the PAIEC streaming protocol.
For each held-out benchmark: pairs (subject x bench) with >=80 distinct items; items split 50/50 (fixed per bench)
into acquisition/eval; per pair a random acquisition order; budgets 0,1,3,7,15,31 nested.
Pooled evidence at budget n = acquired labels of ALL pairs in this bench (the rule says labels are pooled across pairs).
Predictors get (pair id, subject meta, budget-n labels of own pair, pooled labels of other pairs) -> one prob per eval item
(item-level info on eval items is never revealed, so predictions are per pair here).
Score: Brier ALC per pair, mean over pairs, then mean over benchmarks (also report pair-weighted)."""
import numpy as np, pandas as pd, sys, json
from scipy.special import expit, logit
B=[0,1,3,7,15,31]; W=[.1,.2,.2,.2,.2,.1]
def prep(R, bench, rng, max_pairs=60, max_eval=200):
    d=R[R.bench==bench]
    # collapse trials: take first trial per subject-item
    d=d.sort_values('trial').drop_duplicates(['subject_id','item_id'])
    cnt=d.groupby('subject_id').item_id.nunique(); subs=cnt[cnt>=80].index
    if len(subs)==0: return None
    if len(subs)>max_pairs: subs=rng.choice(subs,max_pairs,replace=False)
    d=d[d.subject_id.isin(subs)]
    items=d.item_id.unique(); rng.shuffle(items); half=len(items)//2
    acq=set(items[:half]); ev=set(items[half:])
    pairs=[]
    for s,g in d.groupby('subject_id'):
        a=g[g.item_id.isin(acq)]; e=g[g.item_id.isin(ev)]
        if len(a)<31 or len(e)==0: continue
        a=a.sample(frac=1,random_state=int(rng.integers(1e9)))
        e=e.sample(min(len(e),max_eval),random_state=int(rng.integers(1e9)))
        pairs.append(dict(s=s,name=g.normalized_name.iloc[0],re=g.reasoning_effort.iloc[0],acq=a.y.values,ev=e.y.values))
    return pairs
def evaluate(pairs, predictor, ctx):
    scores=[]
    for i,p in enumerate(pairs):
        alc=0
        for n,w in zip(B,W):
            own=p['acq'][:n]
            pooled=[q['acq'][:n] for j,q in enumerate(pairs) if j!=i]
            pr=predictor(p,own,pooled,ctx)
            alc+=w*np.mean((p['ev']-pr)**2)
        scores.append(alc)
    return np.array(scores)
# ---- predictors
def const(p,own,pooled,ctx): return 0.5
def oracle(p,own,pooled,ctx): return p['ev'].mean()
def unsmoothed(p,own,pooled,ctx): return own.mean() if len(own) else 0.5
def laplace(p,own,pooled,ctx): return (own.sum()+1)/(len(own)+2)
def pooled_mean(p,own,pooled,ctx,k=4.0,k0=2.0,g=0.5):
    allp=np.concatenate(pooled) if pooled else np.array([])
    mu=(allp.sum()+k0*g)/(len(allp)+k0)               # bench mean from other subjects
    return (own.sum()+k*mu)/(len(own)+k)
def pooled_logit(p,own,pooled,ctx,tau=1.0,k0=2.0,g=0.5):
    # bench offset from pooled labels; pair deviation with normal prior sd tau via Laplace approx on own labels
    allp=np.concatenate(pooled) if pooled else np.array([])
    mu=(allp.sum()+k0*g)/(len(allp)+k0); m=logit(np.clip(mu,1e-3,1-1e-3))
    th=ctx.get('theta',{}).get(p['name'],0.0)
    if len(own)==0: return float(expit(m+th*ctx.get('wth',0)))
    # MAP for delta: own ~ Bern(expit(m+delta)), delta~N(th*wth,tau^2)
    d=th*ctx.get('wth',0)
    for _ in range(25):
        q=expit(m+d); grad=(own-q).sum()-(d-th*ctx.get('wth',0))/tau**2; hess=-len(own)*q*(1-q)-1/tau**2
        d-=grad/hess
    return float(expit(m+d))
if __name__=='__main__':
    R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','normalized_name','reasoning_effort','trial','y'])
    rng=np.random.default_rng(0)
    benches=[b for b in R.bench.unique()]
    res={}
    preds={'oracle':oracle,'const':const,'unsmoothed':unsmoothed,'laplace':laplace,'pooled_mean':pooled_mean,'pooled_logit':pooled_logit,'pm_k2':lambda p,o,pl,c:pooled_mean(p,o,pl,c,k=2),'pm_k8':lambda p,o,pl,c:pooled_mean(p,o,pl,c,k=8),'pl_tau05':lambda p,o,pl,c:pooled_logit(p,o,pl,c,tau=0.5),'pl_tau2':lambda p,o,pl,c:pooled_logit(p,o,pl,c,tau=2.0)}
    for b in benches:
        pairs=prep(R,b,rng)
        if not pairs or len(pairs)<2: continue
        res[b]={k:float(evaluate(pairs,f,{}).mean()) for k,f in preds.items()}
        res[b]['npairs']=len(pairs)
        print(b,len(pairs),{k:round(v,4) for k,v in res[b].items() if k!='npairs'},flush=True)
    df=pd.DataFrame(res).T; df.to_csv('data/sim_v1.csv')
    print(df.drop(columns='npairs').mean().round(4).to_dict())
