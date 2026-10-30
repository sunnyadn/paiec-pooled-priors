"""Lightweight learned difficulty: ridge on hashed text features trained on items of the NON-simulation training
benchmarks (relative item difficulty = item logit accuracy minus its benchmark mean), applied to the 12 simulation
benchmarks (never seen in training). Oracle-intercept test vs Haiku/Opus ratings."""
import json, glob, numpy as np, pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
items = json.load(open('data/llm/items_to_rate.json')); TB = set(items)
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'y']); R['item_id'] = R.item_id.astype(str)
R = R.drop_duplicates(['bench', 'subject_id', 'item_id'])
im = R.groupby(['bench', 'item_id']).y.agg(['mean', 'size']).reset_index(); im = im[im['size'] >= 5]
lg = lambda p: np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98)))
im['d'] = lg(im['mean']); im['d'] = im.d - im.groupby('bench').d.transform('mean')
rng = np.random.default_rng(0); X, Y = [], []
for b, g in im[~im.bench.isin(TB)].groupby('bench'):
    try: I = pd.read_parquet(f'data/mdb/{b}/formatted_tables/items.parquet', columns=['item_id', 'content'])
    except Exception: continue
    I['item_id'] = I.item_id.astype(str); g = g.merge(I, on='item_id')
    if len(g) > 2000: g = g.sample(2000, random_state=0)
    X += list(g.content.astype(str).str[:2000]); Y += list(g.d)
print('train items', len(X))
from sentence_transformers import SentenceTransformer
EMB = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')


XE = EMB.encode(X, batch_size=64, normalize_embeddings=True, show_progress_bar=False); M = Ridge(alpha=1.0).fit(XE, Y)
tx = {str(x['item_id']): x['text'] for b in items for x in items[b]}
pred = dict(zip(tx, M.predict(EMB.encode([t[:2000] for t in tx.values()], batch_size=64, normalize_embeddings=True, show_progress_bar=False))))
H = {}
for f in glob.glob('data/llm/api_haiku/*.json'):
    for x in json.load(open(f))['ratings']: H[str(x['id'])] = float(x['p'])
T = R[R.bench.isin(TB) & R.item_id.isin(set(tx))].copy()
T['zt'] = T.item_id.map(pred); T['zh'] = np.log(np.clip(T.item_id.map(H), 2, 98) / (100 - np.clip(T.item_id.map(H), 2, 98)))
T = T.dropna(subset=['zt', 'zh'])
for c in ['zt', 'zh']: T[c] = T.groupby('bench')[c].transform(lambda v: (v - v.mean()) / (v.std() or 1))
imt = T.groupby(['bench', 'item_id']).agg(acc=('y', 'mean'), zt=('zt', 'first'), zh=('zh', 'first')).reset_index()
print(imt.groupby('bench').apply(lambda g: pd.Series({'learned': g.zt.corr(g.acc), 'haiku': g.zh.corr(g.acc)})).round(2))
sg = lambda x: 1 / (1 + np.exp(-x))
subs = T.groupby(['bench', 'subject_id']).filter(lambda g: len(g) >= 30)
subs = subs[subs.subject_id.isin(subs.drop_duplicates(['bench', 'subject_id']).groupby('bench').head(15).subject_id)]
def ev(cols, bs):
    tot = []
    for _, g in subs.groupby(['bench', 'subject_id']):
        y = g.y.values; x = sum(b_ * g[c].values for b_, c in zip(bs, cols)) if cols else 0 * y; a = 0.
        for _ in range(30):
            p = sg(a + x); a = np.clip(a + (y - p).sum() / max((p * (1 - p)).sum(), 1e-6), -8, 8)
        tot.append(np.mean((y - sg(a + x)) ** 2))
    return np.mean(tot)
print('none %.4f' % ev([], []))
print('learned', min((round(ev(['zt'], [b]), 4), b) for b in [0.1, 0.3, 0.5]))
print('haiku', min((round(ev(['zh'], [b]), 4), b) for b in [0.3, 0.5]))
print('both', min((round(ev(['zh', 'zt'], [a, b]), 4), a, b) for a in [0.3, 0.5] for b in [0.1, 0.3]))
