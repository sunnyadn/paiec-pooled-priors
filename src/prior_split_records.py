"""Diagnostic (10-30): Table 1 data-only prior vs deployed chain, split by whether the subject has records on other public benchmarks."""
import json, numpy as np, pandas as pd
E=json.load(open('data/llm/prior_eval_all.json')); O=json.load(open('data/llm/prior_eval_oall.json')); S=json.load(open('data/llm/prior_eval_smiss.json')); H=json.load(open('data/llm/prior_eval_hmiss.json'))
num=lambda x: x[0] if isinstance(x[0],(int,float)) else np.nan
est=np.array([num(o) for o in O],float); skip=np.array([bool(o[1].get('skip')) for o in O])
miss=[i for i in range(len(O)) if np.isnan(est[i]) and not skip[i]]
for j,i in enumerate(miss): est[i]=num(S[j]) if not np.isnan(num(S[j])) else num(H[j])
acc=np.array([e['acc'] for e in E]); bench=np.array([e['bench'] for e in E]); calib=set(bench[skip])
R=pd.read_parquet('data/long.parquet',columns=['bench','normalized_name','y'])
C=R.groupby(['normalized_name','bench']).y.mean().reset_index()
nid={n:k for k,n in enumerate(C.normalized_name.unique())}; bid={b:k for k,b in enumerate(C.bench.unique())}
ni,bj=C.normalized_name.map(nid).values,C.bench.map(bid).values
y=np.clip(C.y.values,.02,.98); z=np.log(y/(1-y)); cell={(n,b):k for k,(n,b) in enumerate(zip(C.normalized_name,C.bench))}
def tf(keep,lam=1.0):
    a,b,mu=np.zeros(len(nid)),np.zeros(len(bid)),z[keep].mean()
    for _ in range(50):
        a=np.bincount(ni[keep],(z-mu-b[bj])[keep],len(nid))/(np.bincount(ni[keep],None,len(nid))+lam)
        b=np.bincount(bj[keep],(z-mu-a[ni])[keep],len(bid))/(np.bincount(bj[keep],None,len(bid))+lam)
    return mu,a,b
sig=lambda v:1/(1+np.exp(-v))
m=~skip&~np.isin(bench,list(calib)); P1=[];has=[]
cnt=C.groupby('normalized_name').bench.nunique()
for i in np.where(m)[0]:
    n,b_=E[i]['subject']['normalized_name'],E[i]['bench']; keep=np.ones(len(C),bool)
    if (n,b_) in cell: keep[cell[(n,b_)]]=False
    other=cnt.get(n,0)-(1 if (n,b_) in cell else 0); has.append(other>0)
    mu,a,b=tf(keep); P1.append(sig(mu+(a[nid[n]] if n in nid else 0.0)))
P1=np.array(P1); has=np.array(has); a_=acc[m]; L=0.5+0.75*(est[m]-0.5)
ok=~np.isnan(L)
for nm,sel in (('all',np.ones(len(a_),bool)),('has other records',has),('no other records',~has)):
    s=sel&ok
    r=lambda p: np.sqrt(np.mean((p[s]-a_[s])**2))
    print(nm, int(s.sum()), 'const %.3f data %.3f llm %.3f'%(r(np.full(len(a_),.5)), r(P1), r(L)))
