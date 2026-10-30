"""Item-level extension: within the held-out benchmark, transfer difficulty from pooled acquired labels
to eval items via text similarity (hashed char n-gram TF-IDF, pure sklearn, no external model)."""
import numpy as np, pandas as pd, json, sys
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from scipy.special import expit, logit
B=[0,1,3,7,15,31]; W=[.1,.2,.2,.2,.2,.1]
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','trial','y'])
benches=[l.strip() for l in open('data/item_benches.txt') if l.strip()]
def text_of(df):
    f=df['item_features'].astype(str) if 'item_features' in df else ''
    return (df['content'].astype(str).str.slice(0,4000)+' '+f).values
def run(bench,rng,max_pairs=60,max_eval=200,lams=(0,0.75,1.0,1.25),ks=(30,60),kps=(2,4,8)):
    items=pd.read_parquet(f'data/mdb/{bench}/formatted_tables/items.parquet',columns=['item_id','content','item_features'])
    d=R[R.bench==bench].sort_values('trial').drop_duplicates(['subject_id','item_id'])
    d=d[d.item_id.isin(items.item_id)]
    cnt=d.groupby('subject_id').item_id.nunique(); subs=cnt[cnt>=80].index
    if len(subs)<2: return None
    if len(subs)>max_pairs: subs=rng.choice(subs,max_pairs,replace=False)
    d=d[d.subject_id.isin(subs)]
    its=d.item_id.unique(); rng.shuffle(its); half=len(its)//2; acqset=set(its[:half])
    items=items[items.item_id.isin(its)].drop_duplicates('item_id').set_index('item_id')
    hv=HashingVectorizer(analyzer='char_wb',ngram_range=(3,5),n_features=2**18,alternate_sign=False)
    X=TfidfTransformer().fit_transform(hv.transform(text_of(items.reset_index())))
    idx={iid:i for i,iid in enumerate(items.index)}
    pairs=[]
    for s,g in d.groupby('subject_id'):
        a=g[g.item_id.isin(acqset)].sample(frac=1,random_state=int(rng.integers(1e9)))
        e=g[~g.item_id.isin(acqset)]
        if len(a)<31 or len(e)==0: continue
        e=e.sample(min(len(e),max_eval),random_state=int(rng.integers(1e9)))
        pairs.append(dict(a_i=np.array([idx[i] for i in a.item_id]),a_y=a.y.values,e_i=np.array([idx[i] for i in e.item_id]),e_y=e.y.values))
    if len(pairs)<2: return None
    res={f'kp{kp}_lam{l}_k{k}':[] for kp in kps for l in lams for k in ks}; res['oracle_pair']=[]
    for i,p in enumerate(pairs):
        alc={key:0.0 for key in res}
        others=[q for j,q in enumerate(pairs) if j!=i]
        for n,w in zip(B,W):
            alc['oracle_pair']+=w*np.mean((p['e_y']-p['e_y'].mean())**2)
            pool_y=np.concatenate([q['a_y'][:n] for q in others]) if n else np.array([])
            mu=(pool_y.sum()+1)/(len(pool_y)+2); own=p['a_y'][:n]
            for kp in kps:
                pm=(own.sum()+kp*mu)/(len(own)+kp)
                res_items={}
                if n:
                    for q in others:
                        qy=q['a_y'][:n]; qm=(qy.sum()+kp*mu)/(n+kp)
                        for it,y in zip(q['a_i'][:n],qy): res_items.setdefault(it,[]).append(y-qm)
                    for it,y in zip(p['a_i'][:n],own): res_items.setdefault(it,[]).append(y-pm)
                if res_items:
                    L=np.array(list(res_items.keys())); rv=np.array([np.mean(v) for v in res_items.values()]); rc=np.array([len(v) for v in res_items.values()])
                    S=(X[p['e_i']]@X[L].T).toarray()
                for l in lams:
                    for k in ks:
                        if l==0 or not res_items: pr=np.full(len(p['e_y']),pm)
                        else:
                            kk=min(k,S.shape[1]); top=np.argpartition(-S,kk-1,axis=1)[:,:kk]
                            sw=np.take_along_axis(S,top,1); vals=rv[top]; cnts=rc[top]; wgt=sw*cnts/(cnts+1)
                            off=(wgt*vals).sum(1)/(wgt.sum(1)+1.0); pr=np.clip(pm+l*off,0.01,0.99)
                        alc[f'kp{kp}_lam{l}_k{k}']+=w*np.mean((p['e_y']-pr)**2)
        for key in res: res[key].append(alc[key])
    return {k:float(np.mean(v)) for k,v in res.items()}|{'npairs':len(pairs)}
if __name__=='__main__':
    rng=np.random.default_rng(0); out={}
    for b in benches:
        r=run(b,rng)
        if r: out[b]=r; print(b,{k:round(v,4) for k,v in r.items()},flush=True)
    df=pd.DataFrame(out).T; df.to_csv('data/sim_item_v2.csv'); print(df.drop(columns='npairs').mean().round(4).to_dict())
