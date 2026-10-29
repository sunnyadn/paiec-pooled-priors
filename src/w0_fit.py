"""Prior weights W0 for the multi-dim features: pooled ridge of (item accuracy - benchmark mean) on within-benchmark
standardized features. Writes data/w0_lobo.json {bench: W0 fit without that bench, '_all': fit on all}."""
import json, numpy as np, pandas as pd
F=json.load(open('data/llm/multidim_all.json'))
R=pd.read_parquet('data/long.parquet',columns=['bench','item_id','y'])
Z={}
for b,d in F.items():
    acc=R[R.bench==b].groupby('item_id').y.mean()
    ids=[i for i in d if i in acc.index]
    X=np.array([d[i] for i in ids],float); X[:,0]=np.log(np.clip(X[:,0],2,98)/(100-np.clip(X[:,0],2,98)))
    X=(X-X.mean(0))/(X.std(0)+1e-9); y=acc[ids].values-acc[ids].values.mean()
    Z[b]=(X,y)
def fit(bs,lam=50.0):
    X=np.vstack([Z[b][0] for b in bs]); y=np.concatenate([Z[b][1] for b in bs])
    return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@y)
out={b:fit([c for c in Z if c!=b]).round(4).tolist() for b in Z}; out['_all']=fit(list(Z)).round(4).tolist()
json.dump(out,open('data/w0_lobo.json','w')); print(out['_all'])
for b in Z:   # how well does the LOBO prior alone rank items in the held-out bench?
    from scipy.stats import spearmanr
    print(b, round(spearmanr(Z[b][0]@np.array(out[b]),Z[b][1]).correlation,3))
