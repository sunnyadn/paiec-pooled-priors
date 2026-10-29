import json, glob, numpy as np, pandas as pd
rat={}
for f in glob.glob('data/llm/api_haiku/*.json'):
    try:
        for x in json.load(open(f))['ratings']: rat[str(x['id'])]=float(x['p'])
    except Exception: pass
items=json.load(open('data/llm/items_to_rate.json'))
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','y'])
R=R[R.bench.isin(list(items))].drop_duplicates(['bench','subject_id','item_id'])
R['p']=R.item_id.astype(str).map(rat); R=R.dropna(subset=['p'])
R['z']=np.log(np.clip(R.p,2,98)/(100-np.clip(R.p,2,98)))
R['zs']=R.groupby('bench').z.transform(lambda v:(v-v.mean())/(v.std() or 1))
sg=lambda x:1/(1+np.exp(-x))
for b in [0,0.3,0.5,0.7,1.0]:
    tot=[];base=[]
    for (bn,s),g in R.groupby(['bench','subject_id']):
        if len(g)<30: continue
        y=g.y.values; z=g.zs.values; a=0.
        for _ in range(30):
            p=sg(a+b*z); a+= (y-p).sum()/max((p*(1-p)).sum(),1e-6); a=np.clip(a,-8,8)
        tot.append(np.mean((y-sg(a+b*z))**2)); base.append(y.mean()*(1-y.mean()))
    print(b, round(np.mean(base),4), round(np.mean(tot),4), len(tot))
print(R.groupby('bench').apply(lambda g: np.corrcoef(g.zs, g.y)[0,1]).round(3))
