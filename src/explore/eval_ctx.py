"""Compare plain vs in-context LLM ratings: Spearman with item accuracy (items not shown as examples), and Brier ALC via sim_llm with fixed split."""
import json, glob, os, re, numpy as np, pandas as pd
from scipy.stats import spearmanr
def load(dirn):
    rat={}
    for f in glob.glob(f'data/llm/{dirn}/*.json'):
        b=os.path.basename(f).split('__')[0]; m=re.search(r'\{.*\}',open(f).read(),re.S)
        try:
            for x in json.loads(m.group(0))['ratings']: rat.setdefault(b,{})[str(x['id'])]=float(x['p'])
        except Exception: pass
    return rat
P=load('batches_haiku'); C=load('ctx_batches'); meta=json.load(open('data/llm/ctx_meta.json'))
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','trial','y'])
R=R.sort_values('trial').drop_duplicates(['bench','subject_id','item_id'])
rows=[]
for b,m in meta.items():
    acc=R[R.bench==b].groupby('item_id').y.mean()
    ids=[i for i in m['E'] if i in P.get(b,{}) and i in C.get(b,{})]
    if len(ids)<30: continue
    a=acc[ids].values; rows.append(dict(bench=b,n=len(ids),plain=spearmanr(a,[P[b][i] for i in ids])[0],ctx=spearmanr(a,[C[b][i] for i in ids])[0]))
df=pd.DataFrame(rows); print(df.round(3).to_string()); print(df[['plain','ctx']].mean().round(3).to_dict())
from scipy.special import logit
def zz(v): v=logit(np.clip(np.array(v)/100,.02,.98)); return (v-v.mean())/(v.std()+1e-9)
rows=[]
for b,m in meta.items():
    acc=R[R.bench==b].groupby('item_id').y.mean()
    ids=[i for i in m['E'] if i in P.get(b,{}) and i in C.get(b,{})]
    if len(ids)<30: continue
    avg=zz([P[b][i] for i in ids])+zz([C[b][i] for i in ids])
    rows.append(dict(bench=b,avg=spearmanr(acc[ids].values,avg)[0]))
print(pd.DataFrame(rows).round(3).to_string(), pd.DataFrame(rows).avg.mean().round(3))
