"""Haiku x5 answer agreement for all simulation items (data/llm/items_to_rate.json). Key2. Resumable.
-> data/llm/consistency_all.json {item_id: agree}"""
import json, os, re, time, urllib.request, collections, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, 'src')
K = open(os.path.expanduser('~/.config/paiec/key2')).read().strip()
items = json.load(open('data/llm/items_to_rate.json'))
SYS = "Answer the question or complete the task. Be brief. End your reply with a last line of the form 'FINAL: <answer>' giving only the final answer (a letter, number, word or short phrase)."
def norm(t):
    m = re.findall(r'FINAL:\s*(.+)', t)
    a = (m[-1] if m else t.strip().splitlines()[-1] if t.strip() else '').strip().lower()
    return re.sub(r'[\s\.\*\(\)"\'`]+', ' ', a).strip()[:60]
def ask(text):
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps({"model": "claude-haiku-5-5", "max_tokens": 2048,
        "output_config": {"effort": "low"}, "system": SYS, "messages": [{"role": "user", "content": text[:3000]}]}).encode(),
        headers={"x-api-key": K, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for att in range(5):
        try:
            o = json.load(urllib.request.urlopen(req, timeout=90)); return norm("".join(b.get('text', '') for b in o['content']))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 529): return None
            time.sleep(3 * (att + 1))
        except Exception: return None
    return None
out = 'data/llm/consistency_all.json'
done = json.load(open(out)) if os.path.exists(out) else {}
todo = [(str(x['item_id']), x['text']) for b in items for x in items[b] if str(x['item_id']) not in done]
print('todo', len(todo), flush=True)
pool = ThreadPoolExecutor(40)
def one(it):
    i, t = it
    outs = [a for a in pool.map(ask, [t] * 5) if a is not None] if False else None
    return i
def item(it):
    i, t = it
    with ThreadPoolExecutor(5) as ex: outs = [a for a in ex.map(ask, [t] * 5) if a is not None]
    return i, (collections.Counter(outs).most_common(1)[0][1] / len(outs) if outs else None)
n = 0
with ThreadPoolExecutor(8) as ex:
    for i, v in ex.map(item, todo):
        done[i] = v; n += 1
        if n % 200 == 0:
            json.dump(done, open(out, 'w')); print(n, flush=True)
json.dump(done, open(out, 'w')); print('done', len(done), sum(v is None for v in done.values()), 'None')
