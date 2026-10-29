# rate every sampled item through the deployed submission code (per-item API call); writes batches-format files for sim_llm2 (RDIR)
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, 'submission'); import model
M, OUTD = sys.argv[1], f'data/llm/{sys.argv[2]}'
model._cfg()['model'] = M; os.makedirs(OUTD, exist_ok=True)
items = json.load(open('data/llm/items_to_rate.json'))
def one(it):
    try: return model._call(it['text'])
    except Exception: return None
for b, its in items.items():
    f = f'{OUTD}/{b}__00.json'
    if os.path.exists(f): continue
    with ThreadPoolExecutor(12) as ex: ps = list(ex.map(one, its))
    json.dump({'ratings': [{'id': it['item_id'], 'p': p} for it, p in zip(its, ps) if p is not None]}, open(f, 'w'))
    print(b, sum(p is not None for p in ps), '/', len(its), flush=True)
