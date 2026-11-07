"""Headline robustness check (zero calls): prior RMSE and answers-equivalent with two truths, the all-responses accuracy (the platform
scores every response) and the first-trial-per-item accuracy (the simulator's approximation); main 277, new 400, pooled; benchmark bootstrap."""
import csv, json, numpy as np, pandas as pd
laplace_mse = lambda a, n: np.mean((n * a * (1 - a) + (1 - 2 * a) ** 2) / (n + 2) ** 2)
labels = lambda a, mse: next((n for n in range(500) if laplace_mse(a, n) <= mse), 500)
R = [r for r in csv.DictReader(open('data/forecasts/forecasts.csv')) if r['benchmark_has_calibration_example'] == 'False']
ALL = {('main', e['bench'], e['subject_id']): e['acc'] for e in json.load(open('data/llm/prior_eval_all.json'))}
ALL.update({('new', e['bench'], e['subject_id']): e['acc'] for e in json.load(open('data/llm/extra/eligible.json'))})
L = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'trial', 'y']); L['subject_id'] = L.subject_id.astype(str)
first = L.sort_values('trial').drop_duplicates(['bench', 'subject_id', 'item_id']).groupby(['bench', 'subject_id']).y.mean()
multi = (L.groupby(['bench', 'subject_id', 'item_id']).size() > 1).groupby(['bench', 'subject_id']).mean()
FIRST, MULTI = {}, {}
for r in R:
    if r['set'] == 'main':
        k = (r['bench'], r['subject_id']); FIRST[('main',) + k] = first.get(k); MULTI[('main',) + k] = multi.get(k)
for b in sorted({r['bench'] for r in R if r['set'] == 'new'}):
    X = pd.read_parquet(f'data/mdb_extra/{b}/formatted_tables/responses.parquet', columns=['subject_id', 'item_id', 'trial', 'response'])
    X['subject_id'] = X.subject_id.astype(str); X['y'] = pd.to_numeric(X.response, errors='coerce'); X = X.dropna(subset=['y'])
    f = X.sort_values(['item_id', 'trial']).drop_duplicates(['subject_id', 'item_id']).groupby('subject_id').y.mean()
    m = (X.groupby(['subject_id', 'item_id']).size() > 1).groupby('subject_id').mean()
    for r in R:
        if r['set'] == 'new' and r['bench'] == b: FIRST[('new', b, r['subject_id'])] = f.get(r['subject_id']); MULTI[('new', b, r['subject_id'])] = m.get(r['subject_id'])
rng = np.random.default_rng(0)
for name, keep in (('main 277', {'main'}), ('new 400', {'new'}), ('pooled 677', {'main', 'new'})):
    x = [r for r in R if r['set'] in keep]; key = lambda r: (r['set'], r['bench'], r['subject_id'])
    assert all(FIRST[key(r)] is not None for r in x), 'missing first-trial accuracy'
    q = np.array([float(r['estimate']) for r in x]); p = 0.5 + 0.75 * (q - 0.5); B = np.array([r['bench'] for r in x]); ub = np.unique(B)
    mshare = np.mean([MULTI[key(r)] > 0 for r in x])
    for tname, T in (('all trials', ALL), ('first trial', FIRST)):
        a = np.array([T[key(r)] for r in x]); Ls = []
        for _ in range(2000):
            ix = np.concatenate([np.where(B == u)[0] for u in rng.choice(ub, len(ub))]); Ls.append(labels(a[ix], np.mean((p[ix] - a[ix]) ** 2)))
        print(f'{name:10s} truth={tname:11s} | RMSE prior {np.sqrt(np.mean((p - a) ** 2)):.3f} constant {np.sqrt(np.mean((0.5 - a) ** 2)):.3f} | answers {labels(a, np.mean((p - a) ** 2))} '
              f'[{np.percentile(Ls, 5):.0f}, {np.percentile(Ls, 95):.0f}] (by benchmark) | pairs with any repeated trial {mshare:.0%}')
