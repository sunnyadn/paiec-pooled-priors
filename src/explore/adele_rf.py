"""ADeLe random-forest control (exploratory, zero calls), same rows/folds/ratings/level as adele_pilot.py analyse: forest [level, Opus] vs
+19 ADeLe features; primary: training benchmarks weighted equally; secondary (declared in advance): rows equal; strongest linear baseline
alongside. usage: adele_rf.py  ->  data/llm/adele/rf_predictions.csv"""
import glob, json, os, sys, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(__file__)); import adele_common as A
D = A.D; ITEMS = json.load(open('data/llm/items_to_rate.json')); L = json.load(open(f'{D}/list.json')); ANNR = json.load(open(f'{D}/annotations.json'))
keys = [(b, i) for b in sorted(L) for i in L[b]]; frozen = {'|'.join(f'{b}:{i}' for b, i in keys[j:j + 5]) for j in range(0, len(keys), 5)}
assert set(ANNR) == frozen and len(frozen) == 180, 'annotation records differ from the 180 frozen tasks'
ANN = {}
for r in ANNR.values(): ANN.update({k: v for k, v in (r.get('levels') or {}).items() if v})
assert len(keys) == 900 and all(f'{b}:{i}' in ANN for b, i in keys), 'some of the 900 frozen items lack a valid annotation'
rat = {}
for f in glob.glob('data/llm/batches_opus_fixed/*.json'):
    for x in json.load(open(f))['ratings']: rat[str(x['id'])] = float(x['p'])
assert all(i in rat for b, i in keys), 'some of the 900 frozen items lack a fence-free Opus rating'
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'trial', 'y'])
R = R.sort_values('trial').drop_duplicates(['bench', 'subject_id', 'item_id']); R['item_id'] = R.item_id.astype(str)
lg = lambda p: np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98))); F = [f'f{j}' for j in range(19)]; rows = []
for b, ids in L.items():
    other = {str(x['item_id']) for x in ITEMS[b]} - set(ids)   # level from the bench's other 150 items, as in adele_pilot.py
    lev = R[(R.bench == b) & R.item_id.isin(other)].groupby('subject_id').y.agg(['sum', 'count'])
    g = R[(R.bench == b) & R.item_id.isin(ids)].join(lev, on='subject_id'); g = g[g['count'] >= 20].copy()   # subjects with too few level items: as before
    r_ = g.item_id.map(rat).clip(2.0, 98.0); g['opus'] = np.log(r_ / (100.0 - r_)); g['level'] = lg((g['sum'] + 1) / (g['count'] + 2))
    g[F] = np.array([ANN[f'{b}:{i}'] for i in g.item_id]); g['bench'] = b; rows.append(g)
G = pd.concat(rows); assert sorted(G.bench.unique()) == sorted(L) and len(G) == 27691, f'rows {len(G)} (expected 27691, as in analysis_fixed.txt)'
RF = dict(n_estimators=300, min_samples_leaf=50, max_features='sqrt', n_jobs=4, random_state=0)   # fixed, not tuned
def wt(df, equal_bench):   # equal_bench: every training benchmark carries the same total weight; mean 1 per row either way
    if not equal_bench: return np.ones(len(df))
    w = 1.0 / df.bench.map(df.bench.value_counts()).values; return w * len(w) / w.sum()
print('rows', len(G), '| items', G.groupby('bench').item_id.nunique().to_dict(), '| forest', RF); OUT = []
for name, eq in (('PRIMARY: training benchmarks weighted equally', True), ('SECONDARY (declared in advance): rows weighted equally', False)):
    print(f'== {name}'); res = []
    for hb in sorted(G.bench.unique()):
        tr, te = G[G.bench != hb], G[G.bench == hb]; y = te.y.values; assert len(tr) >= 1000 and len(te) >= 100
        lin = LogisticRegression(C=1.0, max_iter=5000).fit(np.column_stack([tr.level, tr.opus, tr.level * tr.opus]), tr.y)   # strongest linear baseline, rows equal
        pl = lin.predict_proba(np.column_stack([te.level, te.opus, te.level * te.opus]))[:, 1]; w = wt(tr, eq)
        p0 = RandomForestClassifier(**RF).fit(tr[['level', 'opus']].values, tr.y, sample_weight=w).predict_proba(te[['level', 'opus']].values)[:, 1]
        p1 = RandomForestClassifier(**RF).fit(tr[['level', 'opus'] + F].values, tr.y, sample_weight=w).predict_proba(te[['level', 'opus'] + F].values)[:, 1]
        assert all(np.isfinite(p).all() for p in (pl, p0, p1))
        bl, b0, b1 = (((y - p) ** 2).mean() for p in (pl, p0, p1)); res.append((b0, b1, bl))
        OUT.append(pd.DataFrame(dict(variant='equal_bench' if eq else 'rows', bench=hb, subject_id=te.subject_id.values, item_id=te.item_id.values,
                                     y=y, level=te.level.values, p_linear=pl, p_forest_base=p0, p_forest_adele=p1)))
        print(f'{hb:10s} rows {len(te):6d} | linear base {bl:.4f} | forest base {b0:.4f} | forest +ADeLe {b1:.4f} | +ADeLe - forest base {b1 - b0:+.4f} | +ADeLe - linear {b1 - bl:+.4f}')
    X = np.array(res); rng = np.random.default_rng(0)
    for lab, d in (('+ADeLe - forest base', X[:, 1] - X[:, 0]), ('+ADeLe - linear base', X[:, 1] - X[:, 2])):
        bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        print(f'  {lab}: mean {d.mean():+.4f}, 90% by benchmark [{np.percentile(bs, 5):+.4f}, {np.percentile(bs, 95):+.4f}], better in {np.sum(d < 0)} of {len(d)}')
    print(f'  means: linear base {X[:, 2].mean():.4f} | forest base {X[:, 0].mean():.4f} | forest +ADeLe {X[:, 1].mean():.4f}')
pd.concat(OUT).to_csv(f'{D}/rf_predictions.csv', index=False)
