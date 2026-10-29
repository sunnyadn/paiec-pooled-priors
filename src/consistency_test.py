"""'Simulated examinee': Haiku answers each item 5 times; agreement with the modal answer as a difficulty signal.
Items = those in data/llm/subj_opus (5 benchmarks). Key2. -> data/llm/consistency.json"""
import json, glob, os, re, time, urllib.request, collections
from concurrent.futures import ThreadPoolExecutor
K = open(os.path.expanduser('~/.config/paiec/key2')).read().strip()
items = json.load(open('data/llm/items_to_rate.json')); tx = {str(x['item_id']): x['text'] for b in items for x in items[b]}
ids = sorted({str(r[0]) for f in glob.glob('data/llm/subj_opus/*.json') for r in json.load(open(f))})
SYS = "Answer the question or complete the task. Be brief. End your reply with a last line of the form 'FINAL: <answer>' giving only the final answer (a letter, number, word or short phrase)."
def norm(t):
    m = re.findall(r'FINAL:\s*(.+)', t)
    a = (m[-1] if m else t.strip().splitlines()[-1] if t.strip() else '').strip().lower()
    return re.sub(r'[\s\.\*\(\)"\'`]+', ' ', a).strip()[:60]
def one(i):
    outs = []
    for k in range(5):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps({"model": "claude-haiku-5-5", "max_tokens": 2048,
            "output_config": {"effort": "low"}, "system": SYS, "messages": [{"role": "user", "content": tx[i][:3000]}]}).encode(),
            headers={"x-api-key": K, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        for att in range(4):
            try:
                o = json.load(urllib.request.urlopen(req, timeout=90)); outs.append(norm("".join(b.get('text', '') for b in o['content']))); break
            except urllib.error.HTTPError as e:
                if e.code != 429: break
                time.sleep(4 * (att + 1))
            except Exception: break
    if not outs: return i, None
    c = collections.Counter(outs).most_common(1)[0][1]
    return i, {'agree': c / len(outs), 'n': len(outs), 'answers': outs}
with ThreadPoolExecutor(6) as ex: res = dict(ex.map(one, ids))
json.dump(res, open('data/llm/consistency.json', 'w')); print(len(res), 'items;', sum(v is None for v in res.values()), 'None')
