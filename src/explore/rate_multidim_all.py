"""Multi-dim full-view Haiku features for every sampled item (12 benches) -> data/llm/multidim_all.json {bench:{item_id:[9 floats]}}"""
import json, os, sys, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.argv=['x','none','0']; os.environ['VIEW']='full'
src=open('src/multidim.py').read().split("items=json.load")[0]; exec(src)   # reuse call(), DIMS, SYS
items=json.load(open('data/llm/items_to_rate.json')); out={}
if os.path.exists('data/llm/multidim_all.json'): out=json.load(open('data/llm/multidim_all.json'))
for b,its in items.items():
    if b in out: continue
    IT=pd.read_parquet(f'data/mdb/{b}/formatted_tables/items.parquet',columns=['item_id','content','item_features'])
    full={i:(str(c),str(f)) for i,c,f in zip(IT.item_id,IT.content,IT.item_features)}
    def view(it):
        c,f=full.get(it['item_id'],(it['text'],''))
        if len(c)>6000: c=c[:4000]+"\n...[middle omitted]...\n"+c[-2000:]
        return c+("\n[item features] "+f if f and f!='None' else "")
    with ThreadPoolExecutor(16) as ex: F=list(ex.map(lambda it: call(view(it)),its))
    out[b]={it['item_id']:f for it,f in zip(its,F) if f is not None}
    json.dump(out,open('data/llm/multidim_all.json','w')); print(b,len(out[b]),'/',len(its),'tokens',usage,flush=True)
print('cost $',round(usage[0]*.1e-6+usage[1]*.5e-6,3))
