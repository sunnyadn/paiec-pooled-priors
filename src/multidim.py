"""Multi-dimensional LLM item features vs single difficulty rating (Haiku API). Within each benchmark, 2-fold CV ridge
of item accuracy on the features; compare Spearman with the single pre-rating. usage: multidim.py BENCHES N"""
import json, sys, re, os, urllib.request, random, glob, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
key=json.load(open('submission/llm_config.json'))['key']; usage=[0,0]
DIMS=["p_correct","reasoning_steps","knowledge_obscurity","format_strictness","output_length","ambiguity","refusal_expected","needs_tools","long_context"]
SYS=("You will see one test item from an AI evaluation benchmark (text may be truncated; images are not shown). Do not answer it. "
     "Rate it on these scales and reply with ONLY a JSON object with exactly these keys:\n"
     "p_correct: probability 0-100 that a typical current LLM gets it right; reasoning_steps: 1-10; knowledge_obscurity: 1-10; "
     "format_strictness: 1-10 (how exact the required output format is); output_length: 1-10 (how long a correct response must be); "
     "ambiguity: 1-10; refusal_expected: 0-10 (how likely a safety-tuned model is to refuse); needs_tools: 0-10 (needs code execution, tools or environment interaction); "
     "long_context: 0-10 (needs tracking a lot of context).")
def call(text):
    req=urllib.request.Request("https://api.anthropic.com/v1/messages",data=json.dumps({"model":"claude-haiku-5-5","max_tokens":1024,
        "output_config":{"effort":"low"},"system":SYS,"messages":[{"role":"user","content":"<item>\n"+text+"\n</item>"}]}).encode(),
        headers={"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"})
    try:
        o=json.load(urllib.request.urlopen(req,timeout=60)); usage[0]+=o['usage']['input_tokens']; usage[1]+=o['usage']['output_tokens']
        t="".join(b.get("text","") for b in o["content"]); m=re.search(r"\{.*\}",t,re.S); d=json.loads(m.group(0))
        return [float(d[k]) for k in DIMS]
    except Exception: return None
items=json.load(open('data/llm/items_to_rate.json')); R=pd.read_parquet('data/long.parquet',columns=['bench','item_id','y'])
old={}
for f in glob.glob('data/llm/api_haiku/*.json'):
    for x in json.load(open(f))['ratings']: old[x['id']]=x['p']
N=int(sys.argv[2]); allf={}
for b in sys.argv[1].split(','):
    its=random.Random(0).sample(items[b],min(N,len(items[b])))
    import pandas as _pd; IT=_pd.read_parquet(f'data/mdb/{b}/formatted_tables/items.parquet',columns=['item_id','content','item_features'])
    full={i:(str(c),str(f)) for i,c,f in zip(IT.item_id,IT.content,IT.item_features)}
    def view(it):
        c,f=full.get(it['item_id'],(it['text'],''))
        if os.environ.get('VIEW')=='full' and len(c)>6000: c=c[:4000]+"\n...[middle omitted]...\n"+c[-2000:]
        elif os.environ.get('VIEW')!='full': c=c[:1500]
        return c+("\n[item features] "+f if os.environ.get('VIEW')=='full' and f and f!='None' else "")
    with ThreadPoolExecutor(12) as ex: F=list(ex.map(lambda it: call(view(it)),its))
    acc=R[R.bench==b].groupby('item_id').y.mean()
    rows=[(acc.get(it['item_id']),old.get(it['item_id']),f) for it,f in zip(its,F) if f is not None]
    rows=[r for r in rows if r[0] is not None and r[1] is not None]
    a=np.array([r[0] for r in rows]); o=np.array([r[1] for r in rows]); X=np.array([r[2] for r in rows])
    rng=np.random.default_rng(0); fold=rng.integers(0,2,len(a)); pred=np.zeros(len(a))
    for k in (0,1):
        m=RidgeCV(alphas=[1,10,100,1000]).fit(X[fold!=k],a[fold!=k]); pred[fold==k]=m.predict(X[fold==k])
    singles={d:round(spearmanr(X[:,j],a).correlation,2) for j,d in enumerate(DIMS)}
    print(b,'n',len(a),'rho pre-rating',round(spearmanr(o,a).correlation,3),'rho multi-dim CV',round(spearmanr(pred,a).correlation,3),singles,flush=True)
    allf[b]={it['item_id']:f for it,f in zip(its,F) if f is not None}
json.dump(allf,open('data/llm/multidim_'+os.environ.get('VIEW','trunc')+'.json','w'))
print('tokens',usage,'cost $',round(usage[0]*.1e-6+usage[1]*.5e-6,3))
