"""Run submission/model.py under a local imitation of the platform protocol on held-out public benchmarks."""
import sys, time, importlib.util, numpy as np, pandas as pd
B=[0,1,3,7,15,31]; W=[.1,.2,.2,.2,.2,.1]
spec=importlib.util.spec_from_file_location('model',sys.argv[1]); M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
benches=[l.strip() for l in open('data/item_benches.txt') if l.strip()][:int(sys.argv[2]) if len(sys.argv)>2 else None]
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','normalized_name','provider','release_date','reasoning_effort','harness','trial','y'])
R=R.sort_values('trial').drop_duplicates(['bench','subject_id','item_id'])
rng=np.random.default_rng(0); out={}
for b in benches:
    t0=time.time()
    items=pd.read_parquet(f'data/mdb/{b}/formatted_tables/items.parquet',columns=['item_id','content','item_features']).drop_duplicates('item_id').set_index('item_id')
    d=R[(R.bench==b)&R.item_id.isin(items.index)]
    cnt=d.groupby('subject_id').item_id.nunique(); subs=cnt[cnt>=80].index
    if len(subs)<2: continue
    if len(subs)>30: subs=rng.choice(subs,30,replace=False)
    d=d[d.subject_id.isin(subs)]; its=d.item_id.unique(); rng.shuffle(its); acq=set(its[:len(its)//2])
    idict=lambda i:{'item_content':str(items.at[i,'content']),'item_features':str(items.at[i,'item_features']),'interactors':'','benchmark_id':'bench_x'}
    pairs=[]
    for s,g in d.groupby('subject_id'):
        a=g[g.item_id.isin(acq)].sample(frac=1,random_state=int(rng.integers(1e9))); e=g[~g.item_id.isin(acq)]
        if len(a)<31 or len(e)==0: continue
        e=e.sample(min(len(e),60),random_state=int(rng.integers(1e9))); r0=g.iloc[0]
        subj={'normalized_name':str(r0.normalized_name),'provider':str(r0.provider),'release_date':str(r0.release_date),'access_date':'','harness':str(r0.harness),'harness_version':'','reasoning_effort':str(r0.reasoning_effort),'subject_features_extra':s}
        pairs.append((subj,[(idict(i),y) for i,y in zip(a.item_id,a.y)],[(idict(i),y) for i,y in zip(e.item_id,e.y)]))
    alc=np.zeros(len(pairs))
    for n,w in zip(B,W):
        labeled=[[[subj,it],y] for subj,acqs,_ in pairs for it,y in acqs[:n]]
        for pi,(subj,_,evs) in enumerate(pairs):
            pr=np.array([M.predict([subj,it],labeled) for it,_ in evs]); y=np.array([y for _,y in evs])
            alc[pi]+=w*np.mean((y-pr)**2)
    out[b]=alc.mean(); print(b,len(pairs),round(alc.mean(),4),f'{time.time()-t0:.0f}s',flush=True)
print('MEAN',round(float(np.mean(list(out.values()))),4))
