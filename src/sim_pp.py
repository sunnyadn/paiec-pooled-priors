"""Per-pair split simulation (matches the rules): each subject-benchmark pair splits its items 50/50 into
acquisition/evaluation pools; candidates stream in random order; an acquisition policy reveals up to 31.
Runs submission/model.py predict() with cached API ratings (RDIR) standing in for the LLM.
usage: sim_pp.py MODEL_PY POLICY(random|cover) [RDIR] ; env USE_LLM=0/1, NSUB (max subjects, default 30)"""
import sys, json, glob, re, os, zlib, importlib.util, numpy as np, pandas as pd
B=[0,1,3,7,15,31]; W=[.1,.2,.2,.2,.2,.1]
spec=importlib.util.spec_from_file_location('model',sys.argv[1]); M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
POL=sys.argv[2]; rdir=sys.argv[3] if len(sys.argv)>3 else 'data/llm/api_haiku'
rat={}
for f in glob.glob(rdir+'/*.json'):
    try:
        for x in json.load(open(f))['ratings']: rat[str(x['id'])]=float(x['p'])
    except Exception: pass
if rdir.endswith('multidim_all.json'):   # vector features {bench:{item_id:[...]}}
    rat={i:v for d in json.load(open(rdir)).values() for i,v in d.items()}
W0L=json.load(open(os.environ['W0FILE'])) if os.environ.get('W0FILE') else {}
items_all=json.load(open('data/llm/items_to_rate.json'))
txt2id={}
for b in items_all:
    for x in items_all[b]:
        txt2id[(x['text'][:4000]+' ').lower().strip()]=x['item_id']; txt2id[(x['text'][:4000]+' ').strip()]=x['item_id']; txt2id[x['text'].strip()]=x['item_id']
use_llm=os.environ.get('USE_LLM','1')=='1'
M._cfg=lambda: {'provider':'cache','model':'x','key':'x'} if use_llm else {}
M._call=lambda text: rat.get(txt2id.get(text.strip()))
M._disk_path=lambda: '/dev/null'
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','normalized_name','trial','y'])
R=R.sort_values('trial').drop_duplicates(['bench','subject_id','item_id'])
NSUB=int(os.environ.get('NSUB','30')); rng=np.random.default_rng(int(os.environ.get('SEED','5'))); out={}; DUMPS=[]; PB={}
benches=os.environ.get('BENCHES'); benches=benches.split(',') if benches else list(items_all)
for b in benches:
    if os.environ.get('FULL'):   # full item pool from the database (no LLM ratings exist for most items)
        IT=pd.read_parquet(f'data/mdb/{b}/formatted_tables/items.parquet',columns=['item_id','content'])
        its_l=[{'item_id':i,'text':str(t)} for i,t in zip(IT.item_id,IT.content)]
    else: its_l=items_all[b]
    ids=[x['item_id'] for x in its_l]; tx={x['item_id']:x['text'] for x in its_l}
    if os.environ.get('NITEM'):   # small online regime: a random subset of items per benchmark
        ids=list(rng.choice(ids,min(int(os.environ['NITEM']),len(ids)),replace=False))
    d=R[(R.bench==b)&R.item_id.isin(ids)]; cnt=d.groupby('subject_id').item_id.nunique(); subs=cnt[cnt>=int(os.environ.get('MINITEM','80'))].index
    if len(subs)<int(os.environ.get('MINSUB','3')): continue
    if len(subs)>NSUB: subs=rng.choice(subs,NSUB,replace=False)
    d=d[d.subject_id.isin(subs)]
    idict=lambda i:{'item_content':tx[i],'item_features':'','interactors':'','benchmark_id':'bx'}
    pairs=[]
    for s,g in d.groupby('subject_id'):
        g=g.sample(frac=1,random_state=int(rng.integers(1e9)))
        h=len(g)//2; acq=g.iloc[:h]; ev=g.iloc[h:h+int(os.environ.get('EVALCAP','100000'))]          # per-pair 50/50 split; acq order = stream order
        subj={'normalized_name':str(g.normalized_name.iloc[0]),'subject_features_extra':s}
        pairs.append(dict(subj=subj,stream=list(zip(acq.item_id,acq.y)),ev=[(idict(i),y) for i,y in zip(ev.item_id,ev.y)],pos=0,got=[]))
    if os.environ.get('ORACLE_THETA'):   # diagnostic: replace the ability estimate by the subject's true accuracy on this benchmark
        tru={M._key(p['subj']):float(np.mean([y for _,y in p['stream']]+[y for _,y in p['ev']])) for p in pairs}
        M._pair_rate_res=lambda st,sk,_t=tru: _t.get(sk,0.5)
    if b in W0L: M.W0=W0L[b]
    labeled=[]; seen=set(); lab_y={}; alc=np.zeros(len(pairs))
    racq=np.random.default_rng(zlib.crc32(b.encode()))   # separate stream so policies see identical splits
    for n,w in zip(B,W):
        for p in pairs:   # acquire up to n labels per pair, pairs in turn
            while len(p['got'])<n and p['pos']<len(p['stream']):
                i,y=p['stream'][p['pos']]; p['pos']+=1
                rem_items=len(p['stream'])-p['pos']+1; rem_lab=31-len(p['got'])
                if POL=='random': take=racq.random()<min(1,rem_lab/rem_items)
                elif POL=='stage' and len(p['got'])<int(os.environ.get('STAGEK','3')):
                    ys=lab_y.get(i,[])   # outcomes other subjects revealed for this item
                    ok=len(ys)>=1 and (len(ys)<2 or 0.15<np.mean(ys)<0.85)
                    take=ok or p.get('skips',0)>=0.25*len(p['stream']) or rem_items<=rem_lab
                    if not take: p['skips']=p.get('skips',0)+1
                else: take=(i not in seen) or rem_items<=rem_lab
                if take:
                    p['got'].append(i); seen.add(i); labeled.append([[p['subj'],idict(i)],int(y)]); lab_y.setdefault(i,[]).append(int(y))
        for pi,p in enumerate(pairs):
            pr=np.array([M.predict([p['subj'],it],labeled if n else []) for it,_ in p['ev']]); y=np.array([y for _,y in p['ev']])
            alc[pi]+=w*np.mean((y-pr)**2); PB.setdefault(n,[]).append(np.mean((y-pr)**2))
            if os.environ.get('DUMP'): DUMPS.append(pd.DataFrame(dict(bench=b,n=n,pair=pi,p=pr,y=y)))
    out[b]=alc.mean(); print(b,len(pairs),round(alc.mean(),4),'distinct revealed',len(seen),flush=True)
print('MEAN',round(float(np.mean(list(out.values()))),4),'(benchmark-averaged; reproduce.py weights pairs equally)'); print('PER_BUDGET',{n:round(float(np.mean(v)),4) for n,v in sorted(PB.items())})
if os.environ.get('DUMP'): pd.concat(DUMPS).to_parquet(os.environ['DUMP'])
