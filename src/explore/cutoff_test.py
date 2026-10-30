"""Divergent idea (inv-blackbox / borrow-forecast #3): items that need knowledge newer than the system's training
cutoff should be harder for that system. Haiku tags each item with the latest year of knowledge it requires; test the
interaction with the system's release date within pair (oracle intercept). Key2."""
import json, os, re, time, glob, urllib.request, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
K = open(os.path.expanduser('~/.config/paiec/key2')).read().strip()
items = json.load(open('data/llm/items_to_rate.json')); tx = {str(x['item_id']): x['text'] for b in items for x in items[b]}
out = 'data/llm/item_year.json'; Y = json.load(open(out)) if os.path.exists(out) else {}
SYS = ("For the benchmark item below, give the most recent calendar year of real-world knowledge (events, products, "
       "versions, publications, data) that a system must know to answer it. If it needs no time-specific knowledge "
       "(math, logic, code reasoning, general facts stable before 2015), answer 0. Reply with only the year or 0.")
def tag(i):
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps({"model": "claude-haiku-5-5", "max_tokens": 512,
        "output_config": {"effort": "low"}, "system": SYS, "messages": [{"role": "user", "content": tx[i][:2000]}]}).encode(),
        headers={"x-api-key": K, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for att in range(5):
        try:
            o = json.load(urllib.request.urlopen(req, timeout=60)); n = re.findall(r'\d{1,4}', "".join(b.get('text', '') for b in o['content']))
            return i, (int(n[-1]) if n else None)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 529): return i, None
            time.sleep(3 * (att + 1))
        except Exception: return i, None
    return i, None
todo = [i for i in tx if i not in Y]
with ThreadPoolExecutor(12) as ex:
    for i, v in ex.map(tag, todo): Y[i] = v
json.dump(Y, open(out, 'w'))
yrs = [v for v in Y.values() if v]; print('tagged', len(Y), 'time-specific', len(yrs), 'median year', np.median(yrs) if yrs else None)
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'y']); R['item_id'] = R.item_id.astype(str)
R = R[R.bench.isin(set(items)) & R.item_id.isin(set(tx))].drop_duplicates(['bench', 'subject_id', 'item_id'])
rel = {}
for b in items:
    try: S = pd.read_parquet(f'data/mdb/{b}/formatted_tables/subjects.parquet', columns=['subject_id', 'release_date'])
    except Exception: continue
    for s, d in zip(S.subject_id, S.release_date):
        try: rel[(b, str(s))] = float(str(d)[:4]) + (float(str(d)[5:7]) - 1) / 12
        except Exception: pass
R['rel'] = [rel.get((b, str(s))) for b, s in zip(R.bench, R.subject_id)]; R['iy'] = R.item_id.map(Y)
R = R.dropna(subset=['rel', 'iy']); R = R[R.iy >= 0]
R['gap'] = np.where(R.iy > 0, np.clip(R.iy + 0.5 - R.rel, -3, 3), np.nan)
print('rows with release date', len(R), 'time-specific rows', R.gap.notna().sum())
T = R[R.gap.notna()].copy(); T['after'] = (T.gap > 0).astype(float)
print('accuracy when item year > release:', T[T.after == 1].y.mean().round(3), 'n', int(T.after.sum()), '| <= release:', T[T.after == 0].y.mean().round(3))
sg = lambda x: 1 / (1 + np.exp(-x))
def ev(col, b):
    tot = []
    for _, g in T.groupby(['bench', 'subject_id']):
        if len(g) < 10 or g[col].std() == 0: continue
        y = g.y.values; x = b * g[col].values; a = 0.
        for _ in range(30):
            q = sg(a + x); a = np.clip(a + (y - q).sum() / max((q * (1 - q)).sum(), 1e-6), -8, 8)
        tot.append(np.mean((y - sg(a + x)) ** 2))
    return np.mean(tot), len(tot)
for b in [0.0, -0.3, -0.6, -1.0]: print('after-cutoff weight', b, ev('after', b))
