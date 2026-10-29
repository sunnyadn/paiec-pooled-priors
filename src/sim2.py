import numpy as np, pandas as pd
from scipy.special import expit, logit
from ability import pair_table, Ability
B=[0,1,3,7,15,31]; W=[.1,.2,.2,.2,.2,.1]
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','normalized_name','provider','release_date','reasoning_effort','trial','y'])
R=R.sort_values('trial').drop_duplicates(['bench','subject_id','item_id'])
PT=pair_table(R)
def run(b,rng,max_pairs=60,max_eval=200,scales=(0,0.5,1.0),taus=(0.7,1.5)):
    d=R[R.bench==b]; cnt=d.groupby('subject_id').item_id.nunique(); subs=cnt[cnt>=80].index
    if len(subs)<2: return None
    if len(subs)>max_pairs: subs=rng.choice(subs,max_pairs,replace=False)
    ab=Ability().fit(PT[PT.bench!=b])
    d=d[d.subject_id.isin(subs)]; its=d.item_id.unique(); rng.shuffle(its); acq=set(its[:len(its)//2])
    pairs=[]
    for s,g in d.groupby('subject_id'):
        a=g[g.item_id.isin(acq)].sample(frac=1,random_state=int(rng.integers(1e9))); e=g[~g.item_id.isin(acq)]
        if len(a)<31 or len(e)==0: continue
        e=e.sample(min(len(e),max_eval),random_state=int(rng.integers(1e9)))
        r0=g.iloc[0]; th=ab.predict(r0.normalized_name,r0.provider,r0.release_date,r0.reasoning_effort)
        pairs.append(dict(th=th,a=a.y.values,e=e.y.values))
    if len(pairs)<2: return None
    th=np.array([p['th'] for p in pairs]); th=th-th.mean()   # relative within bench (unknown mean absorbed by pooled mu)
    out={}
    for sc in scales:
        for tau in taus:
            sc_all=[]
            for i,p in enumerate(pairs):
                alc=0
                for n,w in zip(B,W):
                    others=[j for j in range(len(pairs)) if j!=i]
                    # pooled bench intercept: MAP of m given others' labels with offsets sc*th_j
                    m=0.0
                    if n:
                        ys=np.concatenate([pairs[j]['a'][:n] for j in others]); offs=np.repeat(sc*th[others],n)
                        for _ in range(20):
                            q=expit(m+offs); g_=(ys-q).sum()-m/2.0; h=-(q*(1-q)).sum()-1/2.0; m-=g_/h
                    # own deviation delta ~ N(sc*th_i, tau^2)
                    own=p['a'][:n]; d0=sc*th[i]; dlt=d0
                    for _ in range(20):
                        q=expit(m+dlt); g_=(own-q).sum()-(dlt-d0)/tau**2; h=-len(own)*q*(1-q)-1/tau**2; dlt-=g_/h
                    pr=expit(m+dlt); alc+=w*np.mean((p['e']-pr)**2)
                sc_all.append(alc)
            out[f's{sc}_t{tau}']=float(np.mean(sc_all))
    out['npairs']=len(pairs); return out
if __name__=='__main__':
    rng=np.random.default_rng(0); res={}
    for b in R.bench.unique():
        r=run(b,rng)
        if r: res[b]=r; print(b,{k:round(v,4) for k,v in r.items()},flush=True)
    df=pd.DataFrame(res).T; df.to_csv('data/sim2.csv'); print(df.drop(columns='npairs').mean().round(4).to_dict())
