"""Rate simulation items with GPT using the deployed per-item rating prompt (same text the predictor sends to Opus).
usage: rate_gpt_items.py [PER_BENCH]  -> data/llm/gpt_item/ratings.json {item_id: p}; resumable, skips rated items"""
import json, os, re, sys, urllib.request, importlib.util, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 6
spec = importlib.util.spec_from_file_location('m', 'submission/model_v26.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
KEY = open(os.path.expanduser('~/.config/paiec/openai_key')).read().strip()
MODEL = os.environ.get('GPT_MODEL', 'gpt-5.6-sol')
items = json.load(open('data/llm/items_to_rate.json'))
os.makedirs('data/llm/gpt_item', exist_ok=True); F = 'data/llm/gpt_item/ratings.json'
done = json.load(open(F)) if os.path.exists(F) else {}
todo = []
for b, its in items.items():
    sel = its if N >= len(its) else [its[i] for i in np.random.default_rng(1).choice(len(its), N, replace=False)]
    todo += [x for x in sel if str(x['item_id']) not in done]


def rate(x):
    user = "<item>\n" + x['text'][:1500] + "\n</item>\n\n" + M.RATE_TAIL
    for att in range(4):
        try:
            req = urllib.request.Request('https://api.openai.com/v1/responses', data=json.dumps({
                'model': MODEL, 'instructions': M.RATE_PROMPT.strip(), 'input': user, 'reasoning': {'effort': 'low'}}).encode(),
                headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
            out = json.load(urllib.request.urlopen(req, timeout=120))
            text = ''.join(c.get('text', '') for o in out.get('output', []) if o.get('type') == 'message' for c in o.get('content', []))
            nums = re.findall(r"\d+(?:\.\d+)?", text)
            return str(x['item_id']), (min(100., max(0., float(nums[-1]))) if nums else None), out.get('usage', {})
        except Exception:
            time.sleep(2 ** att)
    return str(x['item_id']), None, {}


print('to rate', len(todo), flush=True)
usage = []
with ThreadPoolExecutor(8) as ex:
    for k, (i, v, u) in enumerate(ex.map(rate, todo)):
        if v is not None:
            done[i] = v
        usage.append(u)
        if k % 100 == 99:
            json.dump(done, open(F, 'w'))
json.dump(done, open(F, 'w'))
it = sum(u.get('input_tokens', 0) for u in usage); ot = sum(u.get('output_tokens', 0) for u in usage)
print('rated', len(done), 'input tokens', it, 'output tokens', ot, 'est $', round(it * 4e-6 + ot * 20e-6, 2))
