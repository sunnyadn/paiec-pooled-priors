"""ADeLe pilot (exploratory): 18 DeLeAn demand levels + answer format per item (Opus via subscription CLI, one call per 5 items) on 6
development benchmarks x 150 items; leave-one-benchmark-out, does a mapping fitted on 5 benchmarks add to level + Opus rating?
usage: adele_pilot.py list | annotate (Mac) | analyse (win, needs data/long.parquet)"""
import json, os, random, re, sys, threading
sys.path.insert(0, os.path.dirname(__file__)); import adele_common as A
D = A.D; ITEMS = json.load(open('data/llm/items_to_rate.json'))
if sys.argv[1] == 'list':
    if os.path.exists(f'{D}/list.json'): sys.exit('list exists')
    rng = random.Random(0); benches = rng.sample(sorted(b for b, v in ITEMS.items() if len(v) == 300), 6)
    out = {b: sorted(str(x['item_id']) for x in rng.sample(ITEMS[b], 150)) for b in sorted(benches)}
    json.dump(out, open(f'{D}/list.json', 'w'), indent=0); print({b: len(v) for b, v in out.items()})
elif sys.argv[1] == 'annotate':
    from concurrent.futures import ThreadPoolExecutor
    L = json.load(open(f'{D}/list.json')); TX = {(b, str(x['item_id'])): x['text'] for b in L for x in ITEMS[b]}
    keys = [(b, i) for b in sorted(L) for i in L[b]]; jobs = [keys[j:j + 5] for j in range(0, len(keys), 5)]
    OUT = f'{D}/annotations.json'; RES = json.load(open(OUT)) if os.path.exists(OUT) else {}; LOCK = threading.Lock()
    def ask(job):
        tag = '|'.join(f'{b}:{i}' for b, i in job)
        if tag in RES: return   # never repeat a recorded call (failures included)
        user = A.batch_user([TX[k] for k in job]); r = A.cli(A.BATCH_SYS, user); lv = None
        h = __import__('hashlib').sha256(A.BATCH_SYS.encode()).hexdigest()[:8]; os.makedirs(f'{D}/prompts', exist_ok=True)   # the prompt actually sent
        open(f'{D}/prompts/system_{h}.txt', 'w').write(A.BATCH_SYS); open(f"{D}/prompts/pilot_{job[0][0]}_{job[0][1]}.user.txt", 'w').write(user)
        if r['error'] is None:
            m = re.search(r'\{.*\}', r['text'], re.S)
            try: J = json.loads(m.group(0)) if m else {}
            except Exception: J = {}
            lv = {f'{b}:{i}': A.valid(J.get(str(n), {})) for n, (b, i) in enumerate(job)}   # None for an invalid instance
        with LOCK:
            RES[tag] = dict(levels=lv, served=r['served'], usage=r['usage'], error=r['error'], truncated=[len(TX[k]) > 3000 for k in job], system=h)
            json.dump(RES, open(OUT, 'w'))   # kept after every call
        print(len(RES), '/', len(jobs), r['error'] or 'ok', flush=True)
    with ThreadPoolExecutor(int(os.environ.get('AD_PAR', '3'))) as ex: list(ex.map(ask, jobs[:int(os.environ.get('AD_JOBS', '100000'))]))
else:
    import glob, numpy as np, pandas as pd
    from sklearn.linear_model import LogisticRegression
    L = json.load(open(f'{D}/list.json')); A_ = {}; ANN = json.load(open(f'{D}/annotations.json'))
    keys = [(b, i) for b in sorted(L) for i in L[b]]; frozen = {'|'.join(f'{b}:{i}' for b, i in keys[j:j + 5]) for j in range(0, len(keys), 5)}
    assert set(ANN) == frozen and len(frozen) == 180, f'annotation records {len(ANN)} differ from the 180 frozen tasks'   # failures count as records
    print('annotation calls', len(ANN), '| failed calls', sum(not r.get('levels') for r in ANN.values()),
          '| invalid items', sum(v is None for r in ANN.values() if r.get('levels') for v in r['levels'].values()))
    for r in ANN.values():
        A_.update({k: v for k, v in (r.get('levels') or {}).items() if v})
    rat = {}
    for f in glob.glob(os.environ.get('RATING_DIR', 'data/llm/batches_opus') + '/*.json'):   # RATING_DIR: e.g. the fence-free copy
        try:
            for x in json.load(open(f))['ratings']: rat[str(x['id'])] = float(x['p'])
        except Exception: pass
    R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'trial', 'y'])
    R = R.sort_values('trial').drop_duplicates(['bench', 'subject_id', 'item_id']); R['item_id'] = R.item_id.astype(str)
    lg = lambda p: np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98)))
    F = [f'f{j}' for j in range(19)]; rows = []; EXCL = {}
    for b, ids in L.items():
        other = {str(x['item_id']) for x in ITEMS[b]} - set(ids)   # the level comes from the bench's other 150 items: disjoint from the scored ones
        lev = R[(R.bench == b) & R.item_id.isin(other)].groupby('subject_id').y.agg(['sum', 'count'])
        g = R[(R.bench == b) & R.item_id.isin(ids)].join(lev, on='subject_id'); g = g[g['count'] >= 20].copy()
        n_all = R[(R.bench == b) & R.item_id.isin(ids)].subject_id.nunique(); EXCL[b] = n_all - g.subject_id.nunique()   # subjects with < 20 level items
        g['level'] = lg((g['sum'] + 1) / (g['count'] + 2)); g['has_opus'] = g.item_id.isin(set(rat))
        r = g.item_id.map(rat).fillna(50.0).clip(2.0, 98.0); g['opus'] = np.log(r / (100.0 - r))   # cached ratings are percents: the deployed _rating rule
        g.loc[~g.has_opus, 'opus'] = 0.0; g['miss'] = (~g.has_opus).astype(float)   # pre-set fallback for unrated items, flagged
        g['lo'] = g.level * g.opus; g['ann'] = g.item_id.map(lambda i: f'{b}:{i}' in A_)
        X = np.array([A_.get(f'{b}:{i}', [np.nan] * 19) for i in g.item_id]); g[F] = X; g['bench'] = b; rows.append(g)
    G0 = pd.concat(rows); print('subjects excluded (< 20 level items):', EXCL)
    print('rating check (percent -> logit):', [(v, round(float(np.log(min(max(v, 2), 98) / (100 - min(max(v, 2), 98)))), 3)) for v in (1, 25, 50, 90, 99)])
    for name, G, BASE in (('PRIMARY: items with a cached Opus rating', G0[G0.has_opus], ['level', 'opus', 'lo']),
                          ('ALL items, missing rating -> 0 logit + flag', G0, ['level', 'opus', 'lo', 'miss'])):
      assert sorted(G.bench.unique()) == sorted(L), f'{name}: benchmarks {sorted(G.bench.unique())}'
      print(f'== {name}: rows {len(G)} | items {G.groupby("bench").item_id.nunique().to_dict()} | annotated share {G.ann.mean():.3f}')
      res = []
      for hb in sorted(G.bench.unique()):
          tr, te = G[(G.bench != hb) & G.ann], G[G.bench == hb]; assert len(tr) >= 1000 and len(te) >= 100, f'{hb}: train {len(tr)} test {len(te)}'
          mu, sd = tr[F].values.mean(0), tr[F].values.std(0) + 1e-9
          Z = lambda df: (df[F].values - mu) / sd; feats = lambda df: np.column_stack([df[BASE].values, Z(df), Z(df) * df[['level']].values])
          def wt(df):   # AD_BALANCE=1: every training benchmark carries the same total weight, rescaled to mean 1 per row
              if os.environ.get('AD_BALANCE') != '1': return None
              c = df.bench.map(df.bench.value_counts()); w = 1.0 / c.values; return w * len(w) / w.sum()
          t0 = G[G.bench != hb]
          m0 = LogisticRegression(C=1.0, max_iter=5000).fit(t0[BASE].values, t0.y, sample_weight=wt(t0))
          m1 = LogisticRegression(C=1.0, max_iter=5000).fit(feats(tr), tr.y, sample_weight=wt(tr))
          p0 = m0.predict_proba(te[BASE].values)[:, 1]; p1 = p0.copy(); a = te.ann.values
          if a.any(): p1[a] = m1.predict_proba(feats(te[a]))[:, 1]   # unannotated items fall back to the baseline
          d = (te.y.values - p1) ** 2 - (te.y.values - p0) ** 2; per_pair = pd.Series(d).groupby(te.subject_id.values).mean()
          assert np.isfinite(p0).all() and np.isfinite(p1).all(); res.append(d.mean())
          print(f'{hb:12s} rows {len(te):6d} annotated {a.mean():.0%} Brier base {((te.y.values - p0) ** 2).mean():.4f} +ADeLe {((te.y.values - p1) ** 2).mean():.4f} '
                f'diff {d.mean():+.4f} | subjects better {np.mean(per_pair < 0):.0%} of {len(per_pair)}')
      dm = np.array(res); rng = np.random.default_rng(0); bs = [rng.choice(dm, len(dm)).mean() for _ in range(5000)]
      print(f'mean over {len(dm)} held-out benchmarks {dm.mean():+.4f}, 90% by benchmark [{np.percentile(bs, 5):+.4f}, {np.percentile(bs, 95):+.4f}], better in {np.sum(dm < 0)} of {len(dm)}')
