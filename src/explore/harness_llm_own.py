"""Run submission/model.py on rated benchmarks with cached Sonnet ratings standing in for the API."""
import sys, json, glob, re, os, importlib.util, numpy as np, pandas as pd
B=[0,1,3,7,15,31]; W=[.1,.2,.2,.2,.2,.1]
spec=importlib.util.spec_from_file_location('model',sys.argv[1]); M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
rdir=sys.argv[2] if len(sys.argv)>2 else 'data/llm/batches'
rat={}
for f in glob.glob(rdir+'/*.json'):
    m=re.search(r'\{.*\}',open(f).read(),re.S)
    try:
        for x in json.loads(m.group(0))['ratings']: rat[str(x['id'])]=float(x['p'])
    except Exception: pass
items_all=json.load(open('data/llm/items_to_rate.json'))
txt2id={(x['text'][:4000]+' ').lower().strip():x['item_id'] for b in items_all for x in items_all[b]}
use_llm = os.environ.get('USE_LLM','1')=='1'
M._cfg=lambda: {'provider':'cache','model':'x','key':'x'} if use_llm else {}
M._call=lambda text: rat.get(txt2id.get(text.strip()))
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','normalized_name','trial','y'])
R=R.sort_values('trial').drop_duplicates(['bench','subject_id','item_id'])
rng=np.random.default_rng(5); out={}
for b,its_l in items_all.items():
    ids=[x['item_id'] for x in its_l]; tx={x['item_id']:x['text'] for x in its_l}
    d=R[(R.bench==b)&R.item_id.isin(ids)]; cnt=d.groupby('subject_id').item_id.nunique(); subs=cnt[cnt>=60].index
    if len(subs)<3: continue
    if len(subs)>30: subs=rng.choice(subs,30,replace=False)
    d=d[d.subject_id.isin(subs)]; its=np.array(ids); rng.shuffle(its); acq=set(its[:len(its)//2])
    idict=lambda i:{'item_content':tx[i],'item_features':'','interactors':'','benchmark_id':'bx'}
    pairs=[]
    for s,g in d.groupby('subject_id'):
        a=g[g.item_id.isin(acq)].sample(frac=1,random_state=int(rng.integers(1e9))); e=g[~g.item_id.isin(acq)]
        if len(a)<31 or len(e)<10: continue
        subj={'normalized_name':str(g.normalized_name.iloc[0]),'subject_features_extra':s}
        pairs.append((subj,[(idict(i),y) for i,y in zip(a.item_id,a.y)],[(idict(i),y) for i,y in zip(e.item_id,e.y)]))
    alc=np.zeros(len(pairs))
    for n,w in zip(B,W):
        for pi,(subj,_,evs) in enumerate(pairs):
            labeled=[[[subj,it],y] for it,y in pairs[pi][1][:n]]
            pr=np.array([M.predict([subj,it],labeled) for it,_ in evs]); y=np.array([y for _,y in evs]); alc[pi]+=w*np.mean((y-pr)**2)
    out[b]=alc.mean(); print(b,len(pairs),round(alc.mean(),4),flush=True)
print('MEAN',round(float(np.mean(list(out.values()))),4))
