"""Item-group offsets from item_features key=value categories (pooled residuals), vs kNN char-ngram, vs both. KP=8."""
import numpy as np, pandas as pd, re
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
B=[0,1,3,7,15,31]; W=[.1,.2,.2,.2,.2,.1]; KP=8
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','trial','y'])
R=R.sort_values('trial').drop_duplicates(['bench','subject_id','item_id'])
benches=[l.strip() for l in open('data/item_benches.txt') if l.strip()]
def kv(s):
    out=[]
    for part in str(s).split(';'):
        if '=' in part:
            k,v=part.split('=',1); k=k.strip(); v=v.strip()
            if len(v)<=60 and not re.search(r'(^|_)(id|sha|hash|path|index|uuid)',k.lower()): out.append(k+'='+v)
    return out
def run(bench,rng,max_pairs=60,max_eval=200,G=(2,5,10),L=(0.5,1.0)):
    items=pd.read_parquet(f'data/mdb/{bench}/formatted_tables/items.parquet',columns=['item_id','content','item_features']).drop_duplicates('item_id')
    d=R[(R.bench==bench)&R.item_id.isin(items.item_id)]
    cnt=d.groupby('subject_id').item_id.nunique(); subs=cnt[cnt>=80].index
    if len(subs)<2: return None
    if len(subs)>max_pairs: subs=rng.choice(subs,max_pairs,replace=False)
    d=d[d.subject_id.isin(subs)]; its=d.item_id.unique(); rng.shuffle(its); acq=set(its[:len(its)//2])
    items=items.set_index('item_id').loc[its]; idx={iid:i for i,iid in enumerate(its)}
    groups=[kv(s) for s in items.item_features.values]
    hv=HashingVectorizer(analyzer='char_wb',ngram_range=(3,5),n_features=2**18,alternate_sign=False)
    X=TfidfTransformer().fit_transform(hv.transform((items.content.astype(str).str.slice(0,4000)+' '+items.item_features.astype(str)).values))
    pairs=[]
    for s,g in d.groupby('subject_id'):
        a=g[g.item_id.isin(acq)].sample(frac=1,random_state=int(rng.integers(1e9))); e=g[~g.item_id.isin(acq)]
        if len(a)<31 or len(e)==0: continue
        e=e.sample(min(len(e),max_eval),random_state=int(rng.integers(1e9)))
        pairs.append(dict(a_i=np.array([idx[i] for i in a.item_id]),a_y=a.y.values,e_i=np.array([idx[i] for i in e.item_id]),e_y=e.y.values))
    if len(pairs)<2: return None
    keys=['none','knn']+[f'grp_g{g}_l{l}' for g in G for l in L]+[f'both_g{g}' for g in G]
    res={k:[] for k in keys}
    for i,p in enumerate(pairs):
        alc={k:0.0 for k in keys}; others=[q for j,q in enumerate(pairs) if j!=i]
        for n,w in zip(B,W):
            pool=np.concatenate([q['a_y'][:n] for q in others]) if n else np.array([])
            mu=(pool.sum()+1)/(len(pool)+2); own=p['a_y'][:n]; pm=(own.sum()+KP*mu)/(len(own)+KP)
            base=np.full(len(p['e_y']),pm)
            alc['none']+=w*np.mean((p['e_y']-base)**2)
            ri={}
            if n:
                for q in others:
                    qy=q['a_y'][:n]; qm=(qy.sum()+KP*mu)/(n+KP)
                    for it,y in zip(q['a_i'][:n],qy): ri.setdefault(it,[]).append(y-qm)
            if not ri:
                for k in keys[1:]: alc[k]+=w*np.mean((p['e_y']-base)**2)
                continue
            Li=np.array(list(ri.keys())); rv=np.array([np.mean(v) for v in ri.values()]); rc=np.array([len(v) for v in ri.values()])
            S=(X[p['e_i']]@X[Li].T).toarray(); kk=min(30,S.shape[1]); top=np.argpartition(-S,kk-1,axis=1)[:,:kk]
            sw=np.take_along_axis(S,top,1); wg=sw*rc[top]/(rc[top]+1); knn=(wg*rv[top]).sum(1)/(wg.sum(1)+1.0)
            alc['knn']+=w*np.mean((p['e_y']-np.clip(pm+knn,.01,.99))**2)
            # group stats: sum residual and count per key=value
            gs={}
            for it in ri:
                for g in groups[it]:
                    a=gs.setdefault(g,[0.0,0]); a[0]+=sum(ri[it]); a[1]+=len(ri[it])
            for g_ in G:
                offs=np.zeros(len(p['e_i']))
                for t,it in enumerate(p['e_i']):
                    vals=[gs[g][0]/(gs[g][1]+g_) for g in groups[it] if g in gs]
                    offs[t]=np.mean(vals) if vals else 0.0
                for l in L: alc[f'grp_g{g_}_l{l}']+=w*np.mean((p['e_y']-np.clip(pm+l*offs,.01,.99))**2)
                alc[f'both_g{g_}']+=w*np.mean((p['e_y']-np.clip(pm+0.5*offs+0.5*knn*1.0+0.5*knn*0,.01,.99))**2)
        for k in keys: res[k].append(alc[k])
    return {k:float(np.mean(v)) for k,v in res.items()}
if __name__=='__main__':
    rng=np.random.default_rng(2); out={}
    for b in benches:
        r=run(b,rng)
        if r: out[b]=r; print(b,{k:round(v,4) for k,v in sorted(r.items(),key=lambda x:x[1])[:3]},flush=True)
    df=pd.DataFrame(out).T; df.to_csv('data/sim_group.csv'); print(df.mean().sort_values().round(4).to_dict())
