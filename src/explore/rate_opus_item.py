"""Per-item Opus ratings with the deployed rating prompt, using key2 (experiment key). -> data/llm/opus_item/ratings.json"""
import json, os, random, sys, importlib.util
from concurrent.futures import ThreadPoolExecutor
spec = importlib.util.spec_from_file_location('m', 'submission/model_v24.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
cfg = dict(json.load(open('submission/llm_config.json'))); cfg['key'] = open(os.path.expanduser('~/.config/paiec/key2')).read().strip()
M._cfg = lambda: cfg
items = json.load(open('data/llm/items_to_rate.json'))
N = int(os.environ.get('NPER', '50')); rng = random.Random(0)
todo = [(b, x) for b in items for x in rng.sample(items[b], min(N, len(items[b])))]
os.makedirs('data/llm/opus_item', exist_ok=True); out = 'data/llm/opus_item/ratings.json'
done = json.load(open(out)) if os.path.exists(out) else {}
todo = [(b, x) for b, x in todo if str(x['item_id']) not in done]
def one(bx):
    try: return str(bx[1]['item_id']), M._call(bx[1]['text'])
    except Exception: return str(bx[1]['item_id']), None
with ThreadPoolExecutor(12) as ex:
    for k, v in ex.map(one, todo): done[k] = v
json.dump(done, open(out, 'w')); print(len(done), 'rated;', sum(v is None for v in done.values()), 'None')
