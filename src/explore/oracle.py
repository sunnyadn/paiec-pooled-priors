"""Headroom: Brier of oracles that know the full response matrix of the benchmark (same subject filter as sim_pp)."""
import pandas as pd, numpy as np, json
items_all=json.load(open('data/llm/items_to_rate.json'))
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','trial','y'])
R=R.sort_values('trial').drop_duplicates(['bench','subject_id','item_id'])
rng=np.random.default_rng(5); res=[]
sig=lambda x:1/(1+np.exp(-x))
for b,its in items_all.items():
    ids=[x['item_id'] for x in its]
    d=R[(R.bench==b)&R.item_id.isin(ids)]; cnt=d.groupby('subject_id').item_id.nunique(); subs=cnt[cnt>=80].index
    if len(subs)<3: continue
    if len(subs)>30: subs=rng.choice(subs,30,replace=False)
    d=d[d.subject_id.isin(subs)]
    M=d.pivot_table(index='subject_id',columns='item_id',values='y'); Y=M.values; m=~np.isnan(Y); Y0=np.nan_to_num(Y)
    pair=lambda P: np.nanmean(np.where(m,(Y0-P)**2,np.nan),axis=1).mean()
    # oracle 1: subject true accuracy
    acc=np.nanmean(Y,axis=1,keepdims=True); o1=pair(np.broadcast_to(acc,Y.shape))
    # oracle 2: item mean over OTHER subjects (leave-one-out), no ability
    s=np.nansum(Y,0,keepdims=True); n=m.sum(0,keepdims=True); loo=np.where(m,(s-Y0)/np.maximum(n-1,1),s/np.maximum(n,1)); o2=pair(np.clip(loo,.01,.99))
    # oracle 3: Rasch on full matrix, item difficulty leave-one-subject-out approximated by fitting with ridge
    th=np.zeros((Y.shape[0],1)); de=np.zeros((1,Y.shape[1]))
    for _ in range(50):
        P=sig(th+de); g=np.where(m,Y0-P,0); h=np.where(m,P*(1-P),0)
        th+= (g.sum(1,keepdims=True)-th/4)/(h.sum(1,keepdims=True)+1/4)
        P=sig(th+de); g=np.where(m,Y0-P,0); h=np.where(m,P*(1-P),0)
        de+= (g.sum(0,keepdims=True)-de/4)/(h.sum(0,keepdims=True)+1/4)
    o3=pair(np.clip(sig(th+de),.01,.99))
    res.append((b,round(o1,4),round(o2,4),round(o3,4))); print(res[-1],flush=True)
r=np.array([x[1:] for x in res]); W=0.1*0.25
print('mean per-budget floor: acc-oracle %.4f item-LOO %.4f rasch-full %.4f'%tuple(r.mean(0)))
print('implied ALC floor (budget0=0.25, rest=floor): acc %.4f item %.4f rasch %.4f'%tuple(W+0.9*r.mean(0)))
