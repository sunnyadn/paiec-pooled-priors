"""Opus per-item ratings with the deployed prompt at two reasoning efforts, on a fixed sample of simulation items.
usage: rate_effort_test.py PER_BENCH EFFORT  -> data/llm/eff_<EFFORT>/ratings.json (batch format, for item_signal_oracle GDIR/ODIR)"""
import json, os, re, sys, urllib.request, importlib.util, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
N, EFF = int(sys.argv[1]), sys.argv[2]
spec = importlib.util.spec_from_file_location('m', 'submission/model_v26.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
KEY = open(os.path.expanduser('~/.config/paiec/key2')).read().strip()
items = json.load(open('data/llm/items_to_rate.json'))
sel = [x for b, its in items.items() for x in ([its[i] for i in np.random.default_rng(1).choice(len(its), min(N, len(its)), replace=False)])]
D = f'data/llm/eff_{EFF}'; os.makedirs(D, exist_ok=True); F = D + '/ratings.json'
done = {r['id']: r['p'] for r in json.load(open(F))['ratings']} if os.path.exists(F) else {}


def rate(x):
    user = "<item>\n" + x['text'][:1500] + "\n</item>\n\n" + M.RATE_TAIL
    for att in range(4):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps({
                "model": "claude-opus-5-5", "max_tokens": 4096, "output_config": {"effort": EFF},
                "system": M.RATE_PROMPT.strip(), "messages": [{"role": "user", "content": user}]}).encode(),
                headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            out = json.load(urllib.request.urlopen(req, timeout=180))
            nums = re.findall(r"\d+(?:\.\d+)?", "".join(b.get("text", "") for b in out.get("content", [])))
            return str(x['item_id']), (min(100., max(0., float(nums[-1]))) if nums else None), out.get('usage', {})
        except Exception:
            time.sleep(2 ** att)
    return str(x['item_id']), None, {}


todo = [x for x in sel if str(x['item_id']) not in done]
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(rate, todo))
for i, v, _ in res:
    if v is not None:
        done[i] = v
json.dump({'ratings': [{'id': k, 'p': v} for k, v in done.items()]}, open(F, 'w'))
it = sum(u.get('input_tokens', 0) for _, _, u in res); ot = sum(u.get('output_tokens', 0) for _, _, u in res)
print(EFF, 'rated', len(done), 'of', len(sel), 'in', it, 'out', ot, 'est $', round(it * 4e-6 + ot * 20e-6, 2))
