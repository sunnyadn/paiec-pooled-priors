"""Does the SAME model's residual on similar items from other (training) benchmarks predict its errors on a new
benchmark, beyond item difficulty? Oracle-intercept test, no API. Targets: the 12 rated simulation benchmarks."""
import json, glob, numpy as np, pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
rng = np.random.default_rng(0)
items_rated = json.load(open('data/llm/items_to_rate.json')); TB = list(items_rated)
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'normalized_name', 'y'])
R = R.dropna(subset=['normalized_name']); R['item_id'] = R.item_id.astype(str)
R = R.drop_duplicates(['bench', 'subject_id', 'item_id'])
R['imean'] = R.groupby(['bench', 'item_id']).y.transform('mean')
R['res'] = R.y - R.imean
# per (name, bench, item) mean residual on training (non-target) benches
T = R[~R.bench.isin(TB)].groupby(['normalized_name', 'bench', 'item_id']).res.mean().reset_index()
names = set(R[R.bench.isin(TB)].normalized_name) & set(T.normalized_name)
print('target names with history elsewhere:', len(names))
# texts
def texts_for(bench, ids):
    try:
        I = pd.read_parquet(f'data/mdb/{bench}/formatted_tables/items.parquet', columns=['item_id', 'content'])
    except Exception:
        return {}
    I['item_id'] = I.item_id.astype(str); I = I[I.item_id.isin(ids)]
    return dict(zip(I.item_id, I.content.astype(str).str[:1500]))
HV = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 5), n_features=2**18, alternate_sign=False, norm='l2')
POOL_MAX = 3000; K = 20
pool_cache = {}
def pool(name):
    if name in pool_cache: return pool_cache[name]
    t = T[T.normalized_name == name]
    if len(t) > POOL_MAX: t = t.sample(POOL_MAX, random_state=0)
    tx = []
    for b, g in t.groupby('bench'):
        d = texts_for(b, set(g.item_id)); tx += [(d.get(i), r) for i, r in zip(g.item_id, g.res) if d.get(i)]
    if len(tx) < 100: pool_cache[name] = None; return None
    X = HV.transform([a for a, _ in tx]); pool_cache[name] = (X, np.array([r for _, r in tx])); return pool_cache[name]
rows = []
for b in TB:
    tx = texts_for(b, set(R[R.bench == b].item_id))
    d = R[(R.bench == b) & R.normalized_name.isin(names) & R.item_id.isin(set(tx))]
    subs = d.groupby('subject_id').filter(lambda g: len(g) >= 40).subject_id.unique()
    for s in rng.choice(subs, min(6, len(subs)), replace=False) if len(subs) else []:
        g = d[d.subject_id == s]; nm = g.normalized_name.iloc[0]; P = pool(nm)
        if P is None: continue
        g = g.sample(min(150, len(g)), random_state=0)
        Xq = HV.transform([tx[i] for i in g.item_id]); S = (Xq @ P[0].T).toarray()
        top = np.argsort(-S, axis=1)[:, :K]; w = np.take_along_axis(S, top, 1)
        f = (w * P[1][top]).sum(1) / (w.sum(1) + 1e-9)
        for i, y, fi in zip(g.item_id, g.y, f): rows.append((b, s, i, y, fi))
D = pd.DataFrame(rows, columns=['b', 's', 'i', 'y', 'f']); print('rows', len(D), 'pairs', D.groupby(['b', 's']).ngroups)
D['fz'] = D.groupby('b').f.transform(lambda v: (v - v.mean()) / (v.std() or 1))
H = {}
for f in glob.glob('data/llm/api_haiku/*.json'):
    for x in json.load(open(f))['ratings']: H[str(x['id'])] = float(x['p'])
lg = lambda p: np.log(np.clip(p, 2, 98) / (100 - np.clip(p, 2, 98))); sg = lambda x: 1 / (1 + np.exp(-x))
D['zh'] = lg(D.i.map(H)); D = D.dropna(subset=['zh']); D['zh'] = D.groupby('b').zh.transform(lambda v: (v - v.mean()) / (v.std() or 1))
def ev(cols, bs):
    tot = []
    for _, g in D.groupby(['b', 's']):
        y = g.y.values; x = sum(b_ * g[c].values for b_, c in zip(bs, cols)) if cols else 0 * y; a = 0.
        for _ in range(30):
            p = sg(a + x); a = np.clip(a + (y - p).sum() / max((p * (1 - p)).sum(), 1e-6), -8, 8)
        tot.append(np.mean((y - sg(a + x)) ** 2))
    return np.mean(tot)
print('none %.4f' % ev([], []))
print('same-model retrieval', min((round(ev(['fz'], [b]), 4), b) for b in [0.1, 0.2, 0.3, 0.5]))
print('haiku', min((round(ev(['zh'], [b]), 4), b) for b in [0.3, 0.5, 0.7]))
print('both', min((round(ev(['zh', 'fz'], [a, b]), 4), a, b) for a in [0.3, 0.5] for b in [0.1, 0.2, 0.3]))
print('corr f,y within pair', D.groupby(['b', 's']).apply(lambda g: np.corrcoef(g.fz, g.y)[0, 1] if g.fz.std() > 0 else np.nan).mean().round(3))
