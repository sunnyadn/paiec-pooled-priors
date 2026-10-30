"""Test 'solve then report confidence' as an item-difficulty signal vs the current pre-solve rating.
Haiku via API, per item; correlation with true item accuracy (mean over all subjects). usage: solve_conf.py BENCH[,BENCH] N"""
import json, sys, re, urllib.request, random, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import spearmanr
c=json.load(open('submission/llm_config.json')); key=c['key']
SYS=("You will see one test item from an AI evaluation benchmark (text may be truncated; images are not shown). "
     "Attempt it briefly, then estimate the probability (0-100) that a typical current large language model answers it correctly. "
     "End your reply with a final line 'CONFIDENCE: <number>'.")
usage=[0,0]
def call(text):
    body={"model":"claude-haiku-5-5","max_tokens":1500,"output_config":{"effort":"low"},"system":SYS,
          "messages":[{"role":"user","content":"<item>\n"+text[:1500]+"\n</item>"}]}
    req=urllib.request.Request("https://api.anthropic.com/v1/messages",data=json.dumps(body).encode(),
        headers={"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"})
    try:
        o=json.load(urllib.request.urlopen(req,timeout=60)); usage[0]+=o['usage']['input_tokens']; usage[1]+=o['usage']['output_tokens']
        t="".join(b.get("text","") for b in o["content"]); m=re.findall(r"CONFIDENCE:\s*(\d+(?:\.\d+)?)",t)
        return float(m[-1]) if m else None
    except Exception: return None
items=json.load(open('data/llm/items_to_rate.json'))
R=pd.read_parquet('data/long.parquet',columns=['bench','item_id','y'])
old={}
import glob
for f in glob.glob('data/llm/api_haiku/*.json'):
    for x in json.load(open(f))['ratings']: old[x['id']]=x['p']
N=int(sys.argv[2]); out={}
for b in sys.argv[1].split(','):
    its=random.Random(0).sample(items[b],min(N,len(items[b])))
    with ThreadPoolExecutor(12) as ex: conf=list(ex.map(lambda it: call(it['text']),its))
    acc=R[R.bench==b].groupby('item_id').y.mean()
    rows=[(acc.get(it['item_id']),old.get(it['item_id']),cf) for it,cf in zip(its,conf)]
    rows=[r for r in rows if None not in r and not np.isnan(r[0])]
    a,o,cf=map(np.array,zip(*rows))
    print(b,'n',len(rows),'rho(pre-rating,acc)',round(spearmanr(o,a).correlation,3),'rho(solve-conf,acc)',round(spearmanr(cf,a).correlation,3),flush=True)
    out[b]={it['item_id']:cf for it,cf in zip(its,conf) if cf is not None}
json.dump(out,open('data/llm/solve_conf_'+sys.argv[1].replace(',','_')+'.json','w'))
print('tokens in/out',usage,'cost $',round(usage[0]*0.1e-6+usage[1]*0.5e-6,3))
