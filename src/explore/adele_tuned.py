"""ADeLe pilot, tuned tree models (exploratory, zero calls): nested leave-one-benchmark-out; parameters (and, for the 'auto' strategy, the
model family) chosen only on the 5 training benchmarks; base [level, Opus] and +ADeLe use the same grid and procedure.
usage: adele_tuned.py  ->  data/llm/adele/tuned_{predictions.csv,choices.json,checkpoint.json}"""
import hashlib, itertools, json, os, sys, time, numpy as np, pandas as pd, sklearn, xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
sys.path.insert(0, os.path.dirname(__file__)); import adele_common as A
G, F = A.load_rows(); BASE = ['level', 'opus']; FULL = BASE + F; KINDS = ('xgb', 'rf')
GRID = {'xgb': [dict(max_depth=d, learning_rate=lr, n_estimators=n) for d, lr, n in itertools.product([2, 3, 4], [0.05, 0.1], [100, 300])],
        'rf': [dict(min_samples_leaf=m, max_features=f) for m, f in itertools.product([20, 50, 200], ['sqrt', 0.5])]}
FP = hashlib.sha256(b''.join([open(__file__, 'rb').read(), open(A.__file__, 'rb').read(), pd.util.hash_pandas_object(G, index=False).values.tobytes(),
                              json.dumps(GRID, sort_keys=True).encode(), f'xgb {xgboost.__version__} sklearn {sklearn.__version__}'.encode()])).hexdigest()[:16]
CK = f'{A.D}/tuned_checkpoint.json'; DONE = json.load(open(CK)) if os.path.exists(CK) else {'fingerprint': FP, 'folds': {}}
assert DONE['fingerprint'] == FP, f'checkpoint {DONE["fingerprint"]} was made with other data, code, grid or versions than {FP}: remove it to start over'
def save():   # write to a temporary file, then replace: an interruption cannot leave half a checkpoint
    json.dump(DONE, open(CK + '.tmp', 'w')); os.replace(CK + '.tmp', CK)
def make(kind, p):
    if kind == 'xgb': return XGBClassifier(**p, subsample=1.0, n_jobs=4, random_state=0, eval_metric='logloss', verbosity=0)
    return RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=0, **p)
def wt(df, eq):   # eq: every training benchmark carries the same total weight; mean 1 per row either way
    if not eq: return np.ones(len(df))
    w = 1.0 / df.bench.map(df.bench.value_counts()).values; return w * len(w) / w.sum()
FITS = []
def fitpred(kind, p, cols, tr, te, eq):
    t = time.time(); m = make(kind, p).fit(tr[cols].values, tr.y.values, sample_weight=wt(tr, eq)); FITS.append(time.time() - t)
    return m.predict_proba(te[cols].values)[:, 1]
def brier(y, p): return float(np.mean((y - p) ** 2))
print('fingerprint', FP, '| xgboost', xgboost.__version__, 'sklearn', sklearn.__version__, '| rows', len(G))
for eq in (True, False):
    for hb in sorted(G.bench.unique()):
        tr, te = G[G.bench != hb], G[G.bench == hb]; inner = sorted(tr.bench.unique()); assert len(inner) == 5 and len(te) >= 100
        for kind in KINDS:
            ck = f'{eq}|{hb}|{kind}'
            if ck in DONE['folds']: continue
            rec = {}
            for cols, tag in ((BASE, 'base'), (FULL, 'adele')):
                scores = [[brier(tr[tr.bench == vb].y.values, fitpred(kind, p, cols, tr[tr.bench != vb], tr[tr.bench == vb], eq)) for vb in inner] for p in GRID[kind]]
                assert np.isfinite(scores).all(), f'{ck}|{tag}: non-finite inner score'
                j = int(np.argmin([np.mean(s) for s in scores])); pr = fitpred(kind, GRID[kind][j], cols, tr, te, eq); assert np.isfinite(pr).all()
                rec[tag] = dict(params=GRID[kind][j], inner_mean=float(np.mean(scores[j])), inner_scores=scores, pred=pr.tolist())
            DONE['folds'][ck] = rec; save()
            print(f'  done {ck} | inner base {rec["base"]["inner_mean"]:.4f} adele {rec["adele"]["inner_mean"]:.4f} | mean fit {np.mean(FITS):.2f}s over {len(FITS)} fits', flush=True)
OUT, rng = [], np.random.default_rng(0)
for eq in (True, False):
    print('== weights:', 'equal_bench (PRIMARY)' if eq else 'rows (secondary, declared in advance)')
    res = {k: [] for k in KINDS + ('auto',)}
    for hb in sorted(G.bench.unique()):
        tr, te = G[G.bench != hb], G[G.bench == hb]; y = te.y.values
        lin = LogisticRegression(C=1.0, max_iter=5000).fit(np.column_stack([tr.level, tr.opus, tr.level * tr.opus]), tr.y)
        pl = lin.predict_proba(np.column_stack([te.level, te.opus, te.level * te.opus]))[:, 1]
        recs = {k: DONE['folds'][f'{eq}|{hb}|{k}'] for k in KINDS}
        for k in KINDS + ('auto',):   # auto: per feature set, the family with the lower inner score (base and +ADeLe choose separately)
            pick = {t: (min(KINDS, key=lambda q: recs[q][t]['inner_mean']) if k == 'auto' else k) for t in ('base', 'adele')}
            p0, p1 = np.array(recs[pick['base']]['base']['pred']), np.array(recs[pick['adele']]['adele']['pred'])
            res[k].append((brier(y, p0), brier(y, p1), brier(y, pl)))
            OUT.append(pd.DataFrame(dict(weights='equal_bench' if eq else 'rows', strategy=k, base_family=pick['base'], adele_family=pick['adele'], bench=hb,
                                         subject_id=te.subject_id.values, item_id=te.item_id.values, y=y, level=te.level.values, p_linear=pl, p_base=p0, p_adele=p1)))
    for k in KINDS + ('auto',):
        X = np.array(res[k])
        for lab, d in (('+ADeLe - tuned base', X[:, 1] - X[:, 0]), ('+ADeLe - linear base', X[:, 1] - X[:, 2])):
            bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
            print(f'  {k:4s} {lab}: mean {d.mean():+.4f}, 90% by benchmark [{np.percentile(bs, 5):+.4f}, {np.percentile(bs, 95):+.4f}], better in {np.sum(d < 0)} of 6 '
                  f'| per benchmark {" ".join(f"{v:+.4f}" for v in d)}')
        print(f'  {k:4s} means: linear {X[:, 2].mean():.4f} | tuned base {X[:, 0].mean():.4f} | tuned +ADeLe {X[:, 1].mean():.4f}')
pd.concat(OUT).to_csv(f'{A.D}/tuned_predictions.csv', index=False)
json.dump({k: {t: dict(params=v[t]['params'], inner_mean=v[t]['inner_mean'], inner_scores=v[t]['inner_scores']) for t in v} for k, v in DONE['folds'].items()},
          open(f'{A.D}/tuned_choices.json', 'w'), indent=0)
print('fits this run', len(FITS), '| mean seconds per fit', round(float(np.mean(FITS)), 2) if FITS else None)
