import json, glob, numpy as np, pandas as pd, sys
sys.path.insert(0,'src'); import adele_common as A
L=json.load(open('data/llm/adele/list.json')); ANN={}
for r in json.load(open('data/llm/adele/annotations.json')).values(): ANN.update({k:v for k,v in (r['levels'] or {}).items() if v})
rat={}
for f in glob.glob('data/llm/batches_opus_fixed/*.json'):
    for x in json.load(open(f))['ratings']: rat[str(x['id'])]=float(x['p'])
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','trial','y']); R=R.sort_values('trial').drop_duplicates(['bench','subject_id','item_id']); R['item_id']=R.item_id.astype(str)
names=A.DIMS+['UG']; rows=[]
for b,ids in L.items():
    acc=R[(R.bench==b)&R.item_id.isin(ids)].groupby('item_id').y.mean()
    ids=[i for i in acc.index if f'{b}:{i}' in ANN and i in rat]
    a=acc[ids].values; X=np.array([ANN[f'{b}:{i}'] for i in ids]); op=np.array([rat[i] for i in ids])
    c={n: np.corrcoef(X[:,j],a)[0,1] if X[:,j].std()>0 else np.nan for j,n in enumerate(names)}
    top=sorted(((v,n) for n,v in c.items() if not np.isnan(v)), key=lambda t:-abs(t[0]))[:4]
    sd={n: X[:,j].std() for j,n in enumerate(names)}
    print(f"{b:10s} items {len(ids)} mean acc {a.mean():.2f} | corr(Opus rating, item acc) {np.corrcoef(op,a)[0,1]:+.2f} | strongest demand corr: "+', '.join(f'{n} {v:+.2f}' for v,n in top)+f" | dims with no variation: {[n for n in names if sd[n]==0]}")
