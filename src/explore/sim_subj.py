"""Subject ability transfer: does a subject's relative strength on OTHER benchmarks (public lookup) improve the pair prior?
Leave-one-bench-out, pooled-labels protocol, pair-level only (no item offsets)."""
import numpy as np, pandas as pd, sys
from scipy.special import logit, expit
B=[0,1,3,7,15,31]; W=[.1,.2,.2,.2,.2,.1]; KP=8
R=pd.read_parquet('data/long.parquet',columns=['bench','normalized_name','item_id','trial','y'])
R=R.sort_values('trial').drop_duplicates(['bench','normalized_name','item_id'])
g=R.groupby(['bench','normalized_name']).y.agg(['mean','size']).reset_index(); g=g[g['size']>=20]
g['lg']=np.log((g['mean']*g['size']+.5)/((1-g['mean'])*g['size']+.5)); g['z']=g.lg-g.groupby('bench').lg.transform('mean')
Z={(r.bench,r.normalized_name):r.z for r in g.itertuples()}
byname=g.groupby('normalized_name')
def zo(b,name,kz=1.0):
    v=[Z[(bb,name)] for bb in g.bench[g.normalized_name==name] if bb!=b] if name in byname.groups else []
    return (np.sum(v)/(len(v)+kz)) if v else 0.0, len(v)
def run(b,rng,betas,reps=2):
    d=R[R.bench==b]; cnt=d.groupby('normalized_name').item_id.nunique(); subs=cnt[cnt>=60].index
    if len(subs)<3: return None
    if len(subs)>60: subs=rng.choice(subs,60,replace=False)
    d=d[d.normalized_name.isin(subs)]; its=d.item_id.unique()
    zs={s:zo(b,s)[0] for s in subs}
    keys=['none','adapt']+[f'b{x}' for x in betas]; res={k:[] for k in keys}
    for rep in range(reps):
        rng.shuffle(its); acq=set(its[:len(its)//2]); pairs=[]
        for s,gg in d.groupby('normalized_name'):
            a=gg[gg.item_id.isin(acq)].sample(frac=1,random_state=int(rng.integers(1e9))); e=gg[~gg.item_id.isin(acq)]
            if len(a)<31 or len(e)<10: continue
            pairs.append((s,a.y.values,e.y.values))
        if len(pairs)<3: continue
        for i,(s,ay,ey) in enumerate(pairs):
            alc={k:0.0 for k in keys}; oth=[q for j,q in enumerate(pairs) if j!=i]
            for n,w in zip(B,W):
                pool=np.concatenate([q[1][:n] for q in oth]) if n else np.array([])
                mu=(pool.sum()+1)/(len(pool)+2); own=ay[:n]
                def brier(prior): return w*np.mean((ey-(own.sum()+KP*prior)/(n+KP))**2)
                alc['none']+=brier(mu)
                for x in betas: alc[f'b{x}']+=brier(expit(logit(mu)+x*zs[s]))
                # adaptive slope: regress other subjects' logit pair rates (shrunk) on their zo, ridge
                if n:
                    zq=np.array([zs[q[0]] for q in oth]); rq=np.array([logit((q[1][:n].sum()+KP*mu)/(n+KP))-logit(mu) for q in oth])
                    beta=np.sum(zq*rq)/(np.sum(zq*zq)+2.0)
                    beta=beta*(1+KP/n)  # undo shrinkage attenuation roughly
                else: beta=0.0
                alc['adapt']+=brier(expit(logit(mu)+np.clip(beta,0,1.5)*zs[s]))
            for k in keys: res[k].append(alc[k])
    return {k:float(np.mean(v)) for k,v in res.items() if v}
if __name__=='__main__':
    rng=np.random.default_rng(5); out={}; betas=[0.25,0.5,0.75,1.0]
    for b in sorted(R.bench.unique()):
        r=run(b,rng,betas)
        if r: out[b]=r; print(b,{k:round(v,4) for k,v in r.items()},flush=True)
    df=pd.DataFrame(out).T; df.to_csv('data/sim_subj.csv')
    bl=sorted(df.index); tune=bl[0::2]; test=bl[1::2]
    print('TUNE',df.loc[tune].mean().round(4).to_dict()); print('TEST',df.loc[test].mean().round(4).to_dict())
