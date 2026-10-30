"""Residual correlation between named models across training benchmarks (excluding the simulation benchmarks).
Residual = y - model's accuracy on that benchmark - item mean residual (Rasch-like double centering, in prob space).
Output data/twin_corr.json: {"nameA||nameB": [rho, n_items]} for pairs with >= 200 shared items."""
import pandas as pd, numpy as np, json
EXCL=set(json.load(open('data/llm/items_to_rate.json')))
R=pd.read_parquet('data/long.parquet',columns=['bench','normalized_name','item_id','trial','y'])
R=R[R.normalized_name.notna() & ~R.bench.isin(EXCL)]
R=R.sort_values('trial').drop_duplicates(['bench','normalized_name','item_id'])
S={}   # pair -> [sum xy, sum xx, sum yy, n]
for b,d in R.groupby('bench'):
    if d.item_id.nunique()>60000: d=d[d.item_id.isin(pd.Series(d.item_id.unique()).sample(60000,random_state=0))]
    M=d.pivot_table(index='item_id',columns='normalized_name',values='y',aggfunc='mean')
    if M.shape[1]<2: continue
    X=M.values.astype(np.float32)
    X=X-np.nanmean(X,axis=0,keepdims=True)                 # remove model accuracy
    X=X-np.nanmean(X,axis=1,keepdims=True)                 # remove item difficulty
    m=~np.isnan(X); X0=np.where(m,X,0)
    xy=X0.T@X0; xx=(X0**2).T@m.astype(np.float32); n=m.T.astype(np.float32)@m.astype(np.float32)
    names=list(M.columns)
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            if n[i,j]<30: continue
            k=names[i]+'||'+names[j]; a=S.setdefault(k,[0,0,0,0])
            a[0]+=xy[i,j]; a[1]+=xx[i,j]; a[2]+=xx[j,i]; a[3]+=n[i,j]
    print(b,M.shape,flush=True)
out={k:[float(a[0]/np.sqrt(a[1]*a[2]+1e-9)),int(a[3])] for k,a in S.items() if a[3]>=200}
json.dump(out,open('data/twin_corr.json','w'))
r=np.array([v[0] for v in out.values()]); print('pairs',len(out),'rho quantiles',np.quantile(r,[.1,.5,.9,.99]).round(3))
for k,v in sorted(out.items(),key=lambda x:-x[1][0])[:12]: print(k,v)
