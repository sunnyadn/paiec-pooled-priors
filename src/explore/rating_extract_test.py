"""Does task extraction help the per-item difficulty rating? Rate the same items from the raw and the extracted text with the
deployed Opus rating prompt, and compare each rating's correlation with the item's true accuracy across subjects.
usage: rating_extract_test.py BENCH [N]  -> data/llm/rx_<BENCH>.json"""
import json, os, sys, importlib.util
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import spearmanr
sys.path.insert(0, 'src')
from extract_task import extract, strip_shared
B = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
spec = importlib.util.spec_from_file_location('m', 'submission/model_v26.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
M._cfg = lambda: {'provider': 'anthropic', 'model': 'claude-opus-5-5', 'key': open(os.path.expanduser('~/.config/paiec/key2')).read().strip()}
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'y']); R = R[R.bench == B]
acc = R.groupby(['item_id', 'subject_id']).y.mean().groupby('item_id').mean(); nsub = R.groupby('item_id').subject_id.nunique()   # subjects weighted equally
ids = list(nsub[nsub >= max(2, nsub.max() // 2)].index)
ids = list(np.random.default_rng(0).choice(ids, min(N, len(ids)), replace=False))
I = pd.read_parquet(f'data/mdb/{B}/formatted_tables/items.parquet', columns=['item_id', 'content', 'item_features']).drop_duplicates('item_id').set_index('item_id')
ids = [i for i in ids if i in I.index]
raw = [(str(I.loc[i, 'content'])[:4000] + ' ' + str(I.loc[i, 'item_features'] or '')).strip() for i in ids]   # as the predictor's _text
first = [extract(str(I.loc[i, 'content'])) for i in ids]   # task first, nothing dropped
strip, _ = strip_shared(first)                               # task first, sentences shared by >=20% of items dropped
fmt = lambda xs: [(e[:4000] + ' ' + str(I.loc[i, 'item_features'] or '')).strip() for e, i in zip(xs, ids)]
first, strip = fmt(first), fmt(strip)
f = f'data/llm/rx_{B}.json'
if not os.path.exists(f):
    with ThreadPoolExecutor(8) as ex:
        out = {'ids': [str(i) for i in ids], 'raw': list(ex.map(M._call, raw)), 'first': list(ex.map(M._call, first)), 'strip': list(ex.map(M._call, strip))}
    json.dump(out, open(f, 'w'))
out = json.load(open(f)); a = np.array([acc[i] for i in ids])
V = {k: np.array([np.nan if v is None else v for v in out[k]], float) for k in ('raw', 'first', 'strip')}
m = ~np.isnan(V['raw']) & ~np.isnan(V['first']) & ~np.isnan(V['strip'])   # the same items for every variant
for k, r in V.items():
    print(B, k, 'failed ratings', int(np.isnan(r).sum()), 'of', len(r))
for k, r in V.items():
    print(B, k, 'items', m.sum(), 'distinct ratings', len(set(r[m])), 'spearman with item accuracy', round(spearmanr(r[m], a[m]).correlation, 3))
