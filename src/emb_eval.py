"""Evaluate precomputed embeddings (train.npy/test.npy aligned with /tmp/emb_data.json) as cross-benchmark difficulty predictors."""
import json, sys, glob, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
tr, te = np.load(sys.argv[1]), np.load(sys.argv[2]); D = json.load(open('/tmp/emb_data.json'))
best = None
for a in [1.0, 10.0, 100.0]:
    M = Ridge(alpha=a).fit(tr, D['train_y']); pred = dict(zip(D['test_ids'], M.predict(te)))
    items = json.load(open('data/llm/items_to_rate.json')); TB = set(items)
    R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'y']); R['item_id'] = R.item_id.astype(str)
    R = R[R.bench.isin(TB) & R.item_id.isin(set(pred))].drop_duplicates(['bench', 'subject_id', 'item_id'])
    im = R.groupby(['bench', 'item_id']).y.mean().reset_index(); im['p'] = im.item_id.map(pred)
    c = im.groupby('bench').apply(lambda g: g.p.corr(g.y))
    print('alpha', a, 'mean corr %.3f' % c.mean(), c.round(2).to_dict())
