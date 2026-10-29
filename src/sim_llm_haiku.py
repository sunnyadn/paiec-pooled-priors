"""LLM difficulty rating with on-the-fly calibration from pooled residuals, vs char kNN, vs both. Items restricted to rated subset."""
import numpy as np, pandas as pd, json, glob, re, os
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from scipy.special import logit
B=[0,1,3,7,15,31]; W=[.1,.2,.2,.2,.2,.1]; KP=8
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','trial','y'])
R=R.sort_values('trial').drop_duplicates(['bench','subject_id','item_id'])
rat={}
for f in glob.glob('data/llm/batches_haiku/*.json'):
    b=os.path.basename(f).split('__')[0]; s=open(f).read()
    m=re.search(r'\{.*\}',s,re.S)
    try:
        for x in json.loads(m.group(0))['ratings']: rat.setdefault(b,{})[str(x['id'])]=float(x['p'])
    except Exception: print('bad',f)
items_all=json.load(open('data/llm/items_to_rate.json'))
def run(b,rng,lams=(25.0,),reps=3):
    ids=[x['item_id'] for x in items_all[b] if x['item_id'] in rat.get(b,{})]
    if len(ids)<100: return None
    texts={x['item_id']:x['text'] for x in items_all[b]}
    d=R[(R.bench==b)&R.item_id.isin(ids)]
    cnt=d.groupby('subject_id').item_id.nunique(); subs=cnt[cnt>=60].index
    if len(subs)<3: return None
    if len(subs)>60: subs=rng.choice(subs,60,replace=False)
    d=d[d.subject_id.isin(subs)]
    hv=HashingVectorizer(analyzer='char_wb',ngram_range=(3,5),n_features=2**18,alternate_sign=False)
    X=TfidfTransformer().fit_transform(hv.transform([texts[i] for i in ids])); idx={i:k for k,i in enumerate(ids)}
    z=logit(np.clip(np.array([rat[b][i] for i in ids])/100,0.02,0.98)); z=(z-z.mean())/(z.std()+1e-9)
    keys=['none','knn','llm','both']; res={k:[] for k in keys}
    for rep in range(reps):
        its=np.array(ids); rng.shuffle(its); acq=set(its[:len(its)//2])
        pairs=[]
        for s,g in d.groupby('subject_id'):
            a=g[g.item_id.isin(acq)].sample(frac=1,random_state=int(rng.integers(1e9))); e=g[~g.item_id.isin(acq)]
            if len(a)<31 or len(e)<10: continue
            pairs.append(dict(a_i=np.array([idx[i] for i in a.item_id]),a_y=a.y.values,e_i=np.array([idx[i] for i in e.item_id]),e_y=e.y.values))
        if len(pairs)<3: continue
        for i,p in enumerate(pairs):
            alc={k:0.0 for k in keys}; others=[q for j,q in enumerate(pairs) if j!=i]
            for n,w in zip(B,W):
                pool=np.concatenate([q['a_y'][:n] for q in others]) if n else np.array([])
                mu=(pool.sum()+1)/(len(pool)+2); own=p['a_y'][:n]; pm=(own.sum()+KP*mu)/(len(own)+KP)
                base=np.full(len(p['e_y']),pm); alc['none']+=w*np.mean((p['e_y']-base)**2)
                ri={}
                if n:
                    for q in others:
                        qy=q['a_y'][:n]; qm=(qy.sum()+KP*mu)/(n+KP)
                        for it,y in zip(q['a_i'][:n],qy): ri.setdefault(it,[]).append(y-qm)
                if not ri:
                    for k in keys[1:]: alc[k]+=w*np.mean((p['e_y']-base)**2)
                    continue
                L=np.array(list(ri.keys())); rv=np.array([np.mean(v) for v in ri.values()]); rc=np.array([len(v) for v in ri.values()])
                S=(X[p['e_i']]@X[L].T).toarray(); kk=min(30,S.shape[1]); top=np.argpartition(-S,kk-1,axis=1)[:,:kk]
                sw=np.take_along_axis(S,top,1); wg=sw*rc[top]/(rc[top]+1); knn=(wg*rv[top]).sum(1)/(wg.sum(1)+1.0)
                # calibrate slope of residual on z using revealed items (weighted by counts), ridge toward 0
                zl=z[L]; zc=zl-np.average(zl,weights=rc); beta=np.sum(rc*zc*rv)/(np.sum(rc*zc*zc)+lams[0])
                llm=beta*(z[p['e_i']]-np.average(zl,weights=rc))
                alc['knn']+=w*np.mean((p['e_y']-np.clip(pm+knn,.01,.99))**2)
                alc['llm']+=w*np.mean((p['e_y']-np.clip(pm+llm,.01,.99))**2)
                alc['both']+=w*np.mean((p['e_y']-np.clip(pm+knn+llm,.01,.99))**2)
            for k in keys: res[k].append(alc[k])
    return {k:float(np.mean(v)) for k,v in res.items() if v}
if __name__=='__main__':
    rng=np.random.default_rng(3); out={}
    for b in items_all:
        r=run(b,rng)
        if r: out[b]=r; print(b,{k:round(v,4) for k,v in r.items()},flush=True)
    df=pd.DataFrame(out).T; df.to_csv('data/sim_llm_haiku.csv'); print(df.mean().round(4).to_dict())
    # correlation of rating with item mean accuracy
