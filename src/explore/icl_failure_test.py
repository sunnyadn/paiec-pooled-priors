"""Idea from divergent run (inv-independent/opus #3): give an LLM the system's revealed graded items on this benchmark
and ask it to predict which unrevealed items the same system gets wrong. Oracle-intercept test of item discrimination.
Key2, Sonnet. 12 sim benches x 2 named systems; reveal R items, predict 30."""
import json, os, re, random, time, urllib.request, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
K = open(os.path.expanduser('~/.config/paiec/key2')).read().strip()
REVEAL = int(os.environ.get('REVEAL', '15')); MODEL = os.environ.get('MODEL', 'claude-sonnet-5-5')
items = json.load(open('data/llm/items_to_rate.json')); tx = {str(x['item_id']): x['text'] for b in items for x in items[b]}
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'normalized_name', 'y']).dropna(subset=['normalized_name'])
R['item_id'] = R.item_id.astype(str); R = R[R.bench.isin(set(items)) & R.item_id.isin(set(tx))].drop_duplicates(['bench', 'subject_id', 'item_id'])
rng = random.Random(0); jobs = []
for b, g in R.groupby('bench'):
    subs = g.groupby('subject_id').filter(lambda h: len(h) >= REVEAL + 30).subject_id.unique().tolist()
    for s in rng.sample(subs, min(2, len(subs))):
        h = g[g.subject_id == s].sample(frac=1, random_state=rng.randint(0, 10**6))
        if 0.1 < h.y.mean() < 0.9: jobs.append((b, s, h.normalized_name.iloc[0], h.iloc[:REVEAL], h.iloc[REVEAL:REVEAL + 30]))
SYS = ("You predict which benchmark items a specific AI system answers correctly. You are given items this system has "
       "already attempted on this benchmark, marked CORRECT or WRONG, and new items. Infer what kinds of items this system "
       "fails on, then for each new item give the probability (0-100) that the same system answers it correctly. "
       "Reply only with lines 'ID: probability'.")
def call(job):
    b, s, name, rev, ev = job
    shown = "\n\n".join(f"[{'CORRECT' if y else 'WRONG'}] {tx[i][:400]}" for i, y in zip(rev.item_id, rev.y))
    new = "\n\n".join(f"ID {k}: {tx[i][:400]}" for k, i in enumerate(ev.item_id))
    user = f"System: {name}\n\nAttempted items:\n{shown}\n\nNew items:\n{new}\n\nFor each new item, one line 'ID: probability'."
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps({"model": MODEL, "max_tokens": 4096,
        "output_config": {"effort": "low"}, "system": SYS, "messages": [{"role": "user", "content": user}]}).encode(),
        headers={"x-api-key": K, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for att in range(5):
        try:
            o = json.load(urllib.request.urlopen(req, timeout=180)); t = "".join(x.get('text', '') for x in o['content']); break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 529): return None
            time.sleep(5 * (att + 1))
        except Exception: return None
    else: return None
    p = {int(m[0]): float(m[1]) for m in re.findall(r'(\d+)\s*:\s*(\d+(?:\.\d+)?)', t)}
    return [(b, s, i, y, p.get(k)) for k, (i, y) in enumerate(zip(ev.item_id, ev.y))]
with ThreadPoolExecutor(4) as ex: res = [r for r in ex.map(call, jobs) if r]
rows = [r for rr in res for r in rr if r[4] is not None]
D = pd.DataFrame(rows, columns=['b', 's', 'i', 'y', 'p']); D.to_csv(f'data/llm/icl_failure_R{REVEAL}.csv', index=False)
H = {}
import glob
for f in glob.glob('data/llm/api_haiku/*.json'):
    for x in json.load(open(f))['ratings']: H[str(x['id'])] = float(x['p'])
lg = lambda p: np.log(np.clip(p, 2, 98) / (100 - np.clip(p, 2, 98))); sg = lambda x: 1 / (1 + np.exp(-x))
D['zi'] = lg(D.p); D['zh'] = lg(D.i.map(H).fillna(50))
for c in ['zi', 'zh']: D[c] = D.groupby(['b', 's'])[c].transform(lambda v: (v - v.mean()) / (v.std() or 1))
def ev(cols, bs):
    tot = []
    for _, g in D.groupby(['b', 's']):
        y = g.y.values; x = sum(b_ * g[c].values for b_, c in zip(bs, cols)) if cols else 0 * y; a = 0.
        for _ in range(30):
            q = sg(a + x); a = np.clip(a + (y - q).sum() / max((q * (1 - q)).sum(), 1e-6), -8, 8)
        tot.append(np.mean((y - sg(a + x)) ** 2))
    return np.mean(tot)
print('pairs', D.groupby(['b', 's']).ngroups, 'rows', len(D))
print('within-pair corr: icl %.3f haiku %.3f' % (D.groupby(['b', 's']).apply(lambda g: g.zi.corr(g.y)).mean(), D.groupby(['b', 's']).apply(lambda g: g.zh.corr(g.y)).mean()))
print('none %.4f' % ev([], []))
print('icl', min((round(ev(['zi'], [b]), 4), b) for b in [0.2, 0.4, 0.6]))
print('haiku', min((round(ev(['zh'], [b]), 4), b) for b in [0.2, 0.4, 0.6]))
print('both', min((round(ev(['zh', 'zi'], [a, b]), 4), a, b) for a in [0.2, 0.4] for b in [0.2, 0.4, 0.6]))
