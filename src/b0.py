"""Budget-0 predictor study: p = sigmoid(a + b*logit(LLM rating) + c*ability_prior(name) + d*year + ...), fit leave-one-benchmark-out
on the 12 rated benchmarks; ability prior from the OTHER training benchmarks only."""
import pandas as pd, numpy as np, json, glob, os
from sklearn.linear_model import LogisticRegression
SIM=list(json.load(open('data/llm/items_to_rate.json')))
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','normalized_name','item_id','trial','y'])
R=R.sort_values('trial').drop_duplicates(['bench','subject_id','item_id'])
# subject attributes
S=pd.concat([pd.read_parquet(f) for f in glob.glob('data/mdb/*/formatted_tables/subjects.parquet')])[['subject_id','normalized_name','provider','release_date','reasoning_effort']].drop_duplicates('subject_id')
# ability prior per name from non-sim benches
T=R[~R.bench.isin(SIM) & R.normalized_name.notna()]
acc=T.groupby(['bench','normalized_name']).y.mean().clip(.02,.98).reset_index()
acc['lg']=np.log(acc.y/(1-acc.y)); acc['c']=acc.lg-acc.groupby('bench').lg.transform('mean')
ab=acc.groupby('normalized_name').c.agg(['sum','count']); ab['prior']=ab['sum']/(ab['count']+2.0)
print('names with prior',len(ab))
# rated items
rat={}
for f in glob.glob('data/llm/api_haiku/*.json'):
    for x in json.load(open(f))['ratings']: rat[x['id']]=x['p']
D=R[R.bench.isin(SIM) & R.item_id.isin(rat)].merge(S[['subject_id','provider','release_date','reasoning_effort']],on='subject_id',how='left')
D['r']=np.log(np.clip(D.item_id.map(rat)/100,.02,.98)/(1-np.clip(D.item_id.map(rat)/100,.02,.98)))
D['ab']=D.normalized_name.map(ab['prior']).fillna(0.0); D['has_ab']=D.normalized_name.map(ab['prior']).notna().astype(float)
yr=pd.to_datetime(D.release_date,errors='coerce'); D['yr']=((yr.dt.year+yr.dt.month/12)-2024).fillna(0); D['has_yr']=yr.notna().astype(float)
D['reason']=D.reasoning_effort.notna().astype(float)
print('coverage: ab',D.has_ab.mean().round(3),'yr',D.has_yr.mean().round(3))
feats={'const':[],'rating':['r'],'ab':['ab','has_ab'],'rating+ab':['r','ab','has_ab'],'rating+ab+yr':['r','ab','has_ab','yr','has_yr','reason']}
# per-bench equal weight Brier, leave-one-bench-out
for name,F in feats.items():
    out=[]
    for b in SIM:
        tr=D[D.bench!=b]; te=D[D.bench==b]
        if len(te)==0: continue
        # weight rows so each bench counts equally in training
        w=1.0/tr.groupby('bench').bench.transform('size')
        if F:
            m=LogisticRegression(C=1.0).fit(tr[F],tr.y,sample_weight=w); p=m.predict_proba(te[F])[:,1]
        else:
            p=np.full(len(te),np.average(tr.y,weights=w))
        # pair-averaged Brier like the metric
        te=te.assign(e=(te.y-p)**2); out.append(te.groupby('subject_id').e.mean().mean())
    print(f'{name:14s} budget0 Brier {np.mean(out):.4f}  (0.5 -> 0.25)')
