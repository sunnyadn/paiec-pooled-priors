"""Compare item representations (char-ngram TF-IDF vs MiniLM embeddings) and prob vs logit offsets. kp=8."""
import numpy as np, pandas as pd, sys
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from scipy.special import expit, logit
B=[0,1,3,7,15,31]; W=[.1,.2,.2,.2,.2,.1]; KP=8
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','trial','y'])
R=R.sort_values('trial').drop_duplicates(['bench','subject_id','item_id'])
benches=[l.strip() for l in open('data/item_benches.txt') if l.strip()]
from sentence_transformers import SentenceTransformer
ST=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def reps(texts):
    hv=HashingVectorizer(analyzer='char_wb',ngram_range=(3,5),n_features=2**18,alternate_sign=False)
    Xc=TfidfTransformer().fit_transform(hv.transform(texts))
    Xe=ST.encode(list(texts),batch_size=64,normalize_embeddings=True,show_progress_bar=False)
    return {'char':lambda a,b:(Xc[a]@Xc[b].T).toarray(),'emb':lambda a,b:Xe[a]@Xe[b].T}
def run(bench,rng,max_pairs=40,max_eval=150,max_items=4000):
    items=pd.read_parquet(f'data/mdb/{bench}/formatted_tables/items.parquet',columns=['item_id','content','item_features']).drop_duplicates('item_id')
    d=R[(R.bench==bench)&R.item_id.isin(items.item_id)]
    cnt=d.groupby('subject_id').item_id.nunique(); subs=cnt[cnt>=80].index
    if len(subs)<2: return None
    if len(subs)>max_pairs: subs=rng.choice(subs,max_pairs,replace=False)
    d=d[d.subject_id.isin(subs)]; its=d.item_id.unique(); rng.shuffle(its); its=its[:max_items]; d=d[d.item_id.isin(its)]
    half=len(its)//2; acq=set(its[:half])
    items=items.set_index('item_id').loc[its]
    texts=(items.content.astype(str).str.slice(0,2000)+' '+items.item_features.astype(str)).values
    S=reps(texts); idx={iid:i for i,iid in enumerate(its)}
    pairs=[]
    for s,g in d.groupby('subject_id'):
        a=g[g.item_id.isin(acq)].sample(frac=1,random_state=int(rng.integers(1e9))); e=g[~g.item_id.isin(acq)]
        if len(a)<31 or len(e)==0: continue
        e=e.sample(min(len(e),max_eval),random_state=int(rng.integers(1e9)))
        pairs.append(dict(a_i=np.array([idx[i] for i in a.item_id]),a_y=a.y.values,e_i=np.array([idx[i] for i in e.item_id]),e_y=e.y.values))
    if len(pairs)<2: return None
    variants=[(r,mode,l) for r in ('char','emb') for mode in ('prob','logit') for l in (0.75,1.0,1.5)]
    res={f'{r}_{m}_{l}':[] for r,m,l in variants}; res['none']=[]
    for i,p in enumerate(pairs):
        alc={k:0.0 for k in res}; others=[q for j,q in enumerate(pairs) if j!=i]
        for n,w in zip(B,W):
            pool_y=np.concatenate([q['a_y'][:n] for q in others]) if n else np.array([])
            mu=(pool_y.sum()+1)/(len(pool_y)+2); own=p['a_y'][:n]; pm=(own.sum()+KP*mu)/(len(own)+KP)
            alc['none']+=w*np.mean((p['e_y']-pm)**2)
            ri={}
            if n:
                for q in others:
                    qy=q['a_y'][:n]; qm=(qy.sum()+KP*mu)/(n+KP)
                    for it,y in zip(q['a_i'][:n],qy): ri.setdefault(it,[]).append((y-qm, logit(np.clip(qm,.02,.98))))
            for r,m,l in variants:
                key=f'{r}_{m}_{l}'
                if not ri: alc[key]+=w*np.mean((p['e_y']-pm)**2); continue
                L=np.array(list(ri.keys())); rc=np.array([len(v) for v in ri.values()])
                if m=='prob': rv=np.array([np.mean([a for a,_ in v]) for v in ri.values()])
                else:
                    # item logit residual: logit(item mean, smoothed) - mean pair logit
                    rv=np.array([logit(np.clip((np.mean([a for a,_ in v])+np.mean([expit(b) for _,b in v])*len(v)+0.5)/(len(v)+1),.02,.98))-np.mean([b for _,b in v]) for v in ri.values()])
                Sm=S[r](p['e_i'],L); kk=min(30,Sm.shape[1]); top=np.argpartition(-Sm,kk-1,axis=1)[:,:kk]
                sw=np.clip(np.take_along_axis(Sm,top,1),0,None); vals=rv[top]; cn=rc[top]; wg=sw*cn/(cn+1)
                off=(wg*vals).sum(1)/(wg.sum(1)+1.0)
                pr=np.clip(pm+l*off,.01,.99) if m=='prob' else np.clip(expit(logit(np.clip(pm,.01,.99))+l*off),.01,.99)
                alc[key]+=w*np.mean((p['e_y']-pr)**2)
        for k in res: res[k].append(alc[k])
    return {k:float(np.mean(v)) for k,v in res.items()}
if __name__=='__main__':
    rng=np.random.default_rng(1); out={}
    for b in benches:
        r=run(b,rng)
        if r: out[b]=r; print(b,{k:round(v,4) for k,v in sorted(r.items(),key=lambda x:x[1])[:4]},flush=True)
    df=pd.DataFrame(out).T; df.to_csv('data/sim_item3.csv'); print(df.mean().sort_values().round(4).to_dict())
