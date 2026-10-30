"""Item-level signal given the TRUE pair accuracy (oracle intercept): Haiku vs Opus vs multidim ratings; the slope is picked from four values on the same responses (exploratory)."""
import json, glob, os, numpy as np, pandas as pd
def load_dir(d):
    r = {}
    for f in glob.glob(d + '/*.json'):
        try:
            for x in json.load(open(f))['ratings']: r[str(x['id'])] = float(x['p'])
        except Exception: pass
    return r
H = load_dir('data/llm/api_haiku'); O = load_dir(os.environ.get('ODIR','data/llm/batches_opus')); MD = json.load(open('data/llm/multidim_all.json'))
items = json.load(open('data/llm/items_to_rate.json'))
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'y'])
R = R[R.bench.isin(list(items))].drop_duplicates(['bench', 'subject_id', 'item_id']); R['item_id'] = R.item_id.astype(str)
lg = lambda p: np.log(np.clip(p, 2, 98) / (100 - np.clip(p, 2, 98)))
sg = lambda x: 1 / (1 + np.exp(-x))
R['zh'] = R.item_id.map(H).pipe(lambda s: lg(s)); R['zo'] = R.item_id.map(O).pipe(lambda s: lg(s))
R['md0'] = R.apply(lambda r: MD.get(r.bench, {}).get(r.item_id, [np.nan])[0], axis=1).pipe(lambda s: lg(s))
G = load_dir(os.environ['GDIR']) if os.environ.get('GDIR') else (json.load(open('data/llm/gpt_item/ratings.json')) if os.path.exists('data/llm/gpt_item/ratings.json') else {})
cols = ['zh', 'zo', 'md0']
if G:   # cross-vendor item ratings: compare every column on the items GPT rated
    R['zg'] = R.item_id.map(G).pipe(lambda s: lg(s)); cols.append('zg')
R = R.dropna(subset=cols)
for c in cols: R[c] = R.groupby('bench')[c].transform(lambda v: (v - v.mean()) / (v.std() or 1))
R['zho'] = (R.zh + R.zo) / 2
if G: R['zog'] = (R.zo + R.zg) / 2
print('rows', len(R), 'benches', R.bench.nunique())
def ev(col, b):
    tot = []; base = []
    for _, g in R.groupby(['bench', 'subject_id']):
        if len(g) < int(os.environ.get('MINN', '30')): continue
        y = g.y.values; x = b * g[col].values if col else 0 * y; a = 0.
        for _ in range(30):
            p = sg(a + x); a = np.clip(a + (y - p).sum() / max((p * (1 - p)).sum(), 1e-6), -8, 8)
        tot.append(np.mean((y - sg(a + x)) ** 2)); base.append(y.mean() * (1 - y.mean()))
    return np.mean(base), np.mean(tot)
none = ev(None, 0)[0]
for col in ['zh', 'zo', 'md0', 'zho'] + (['zg', 'zog'] if G else []):
    best = min((ev(col, b)[1], b) for b in [0.3, 0.5, 0.7, 0.9])
    print(col, 'oracle-intercept Brier %.4f (b=%.1f) vs none %.4f' % (best[0], best[1], none))
