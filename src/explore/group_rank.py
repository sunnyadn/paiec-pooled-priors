"""Small test: LLM ranks groups of 10 items by difficulty; each item in K groups; score = mean normalized rank.
Compare Spearman with true item accuracy vs the per-item pre-rating on the same items. usage: group_rank.py BENCHES N K"""
import json, sys, re, urllib.request, random, glob, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import spearmanr
key=json.load(open('submission/llm_config.json'))['key']; usage=[0,0]
SYS=("You will see several test items from one AI evaluation benchmark (text may be truncated; images are not shown). "
     "Do not answer them. Rank them from EASIEST to HARDEST for a typical current large language model. "
     "Reply with only the item numbers in order, comma-separated, e.g. 3,1,2.")
def call(group):
    body="\n\n".join(f"<item {k+1}>\n{t[:800]}\n</item {k+1}>" for k,t in enumerate(group))
    req=urllib.request.Request("https://api.anthropic.com/v1/messages",data=json.dumps({"model":"claude-haiku-5-5","max_tokens":2000,
        "output_config":{"effort":"low"},"system":SYS,"messages":[{"role":"user","content":body}]}).encode(),
        headers={"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"})
    try:
        o=json.load(urllib.request.urlopen(req,timeout=90)); usage[0]+=o['usage']['input_tokens']; usage[1]+=o['usage']['output_tokens']
        t="".join(b.get("text","") for b in o["content"]).strip().splitlines()[-1]
        order=[int(x)-1 for x in re.findall(r"\d+",t)]
        return order if sorted(order)==list(range(len(group))) else None
    except Exception: return None
items=json.load(open('data/llm/items_to_rate.json')); R=pd.read_parquet('data/long.parquet',columns=['bench','item_id','y'])
old={}
for f in glob.glob('data/llm/api_haiku/*.json'):
    for x in json.load(open(f))['ratings']: old[x['id']]=x['p']
N,K=int(sys.argv[2]),int(sys.argv[3])
for b in sys.argv[1].split(','):
    its=random.Random(0).sample(items[b],N); rng=random.Random(1); groups=[]
    for _ in range(K):
        idx=list(range(N)); rng.shuffle(idx); groups+= [idx[i:i+10] for i in range(0,N,10)]
    with ThreadPoolExecutor(12) as ex: orders=list(ex.map(lambda g: call([its[i]['text'] for i in g]),groups))
    sc={i:[] for i in range(N)}; bad=0
    for g,o in zip(groups,orders):
        if o is None: bad+=1; continue
        for pos,k in enumerate(o): sc[g[k]].append(pos/(len(g)-1))   # 0 easiest .. 1 hardest
    acc=R[R.bench==b].groupby('item_id').y.mean()
    rows=[(acc.get(its[i]['item_id']),old.get(its[i]['item_id']),np.mean(v)) for i,v in sc.items() if v]
    rows=[r for r in rows if r[0] is not None and r[1] is not None]
    a,o,g=map(np.array,zip(*rows))
    print(b,'n',len(rows),'failed groups',bad,'rho pre-rating',round(spearmanr(o,a).correlation,3),'rho group-rank',round(-spearmanr(g,a).correlation,3),
          'rho avg',round(spearmanr(pd.Series(o).rank()-pd.Series(g).rank(),a).correlation,3),flush=True)
print('tokens',usage,'cost $',round(usage[0]*.1e-6+usage[1]*.5e-6,3))
