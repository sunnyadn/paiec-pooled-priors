"""Opus answers each item once (pseudo-reference); match = share of Haiku's 5 answers equal to it. Key2."""
import json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
K = open(os.path.expanduser('~/.config/paiec/key2')).read().strip()
items = json.load(open('data/llm/items_to_rate.json')); tx = {str(x['item_id']): x['text'] for b in items for x in items[b]}
C = json.load(open('data/llm/consistency.json'))
SYS = "Answer the question or complete the task. Be brief. End your reply with a last line of the form 'FINAL: <answer>' giving only the final answer (a letter, number, word or short phrase)."
def norm(t):
    m = re.findall(r'FINAL:\s*(.+)', t)
    a = (m[-1] if m else t.strip().splitlines()[-1] if t.strip() else '').strip().lower()
    return re.sub(r'[\s\.\*\(\)"\'`]+', ' ', a).strip()[:60]
def ask(i):
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps({"model": "claude-opus-5-5", "max_tokens": 4096,
        "output_config": {"effort": "low"}, "system": SYS, "messages": [{"role": "user", "content": tx[i][:3000]}]}).encode(),
        headers={"x-api-key": K, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for att in range(5):
        try:
            o = json.load(urllib.request.urlopen(req, timeout=120)); return i, norm("".join(b.get('text', '') for b in o['content']))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 529): return i, None
            time.sleep(4 * (att + 1))
        except Exception: return i, None
    return i, None
ids = [i for i, v in C.items() if v]
with ThreadPoolExecutor(4) as ex: R = dict(ex.map(ask, ids))
out = {}
for i in ids:
    a = R.get(i); hs = C[i]['answers']
    out[i] = None if a is None else sum(h == a for h in hs) / len(hs)
json.dump({'opus_answer': R, 'match': out}, open('data/llm/panel_opus.json', 'w'))
print(len(out), 'items;', sum(v is None for v in out.values()), 'None')
