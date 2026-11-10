"""New-benchmark check, step 1 (runs on win): eligible (benchmark, named subject) pairs from measurement-db benchmarks outside the
main evaluation, the frozen run order (at most 400), and the data-only predictor for every eligible pair.
usage: prior_extra_build.py  ->  data/llm/extra/{eligible,run_order,data_pred}.json"""
import glob, hashlib, json, os, random
import numpy as np, pandas as pd
OUT = 'data/llm/extra'; os.makedirs(OUT, exist_ok=True)
MAIN = sorted({x['bench'] for x in json.load(open('data/llm/prior_eval_all.json'))})
tables = lambda root: {p.split('/')[-3]: os.path.dirname(p) for p in glob.glob(f'{root}/*/formatted_tables/responses.parquet')}
SEM = json.load(open('data/llm/extra/semantics.json'))   # score meaning, declared before any estimate
LONG = set(pd.read_parquet('data/long.parquet', columns=['bench']).bench.unique())   # the 0/1 benchmarks behind Table 1
NEW = {b: d for b, d in tables('data/mdb_extra').items() if b in SEM['keep']}   # only the benchmarks whose score meaning was checked
OLD = {}   # the Table 1 benchmarks enter the data-only predictor from long.parquet, exactly as in Table 1 (below)

def hashes(d):
    try: return set(pd.read_parquet(d + '/items.parquet', columns=['content_hash']).content_hash.astype(str))
    except Exception: return set()

MAIN_DIRS = {b: f'data/mdb/{b}/formatted_tables' for b in MAIN}   # item tables of the 32 main benchmarks (synced to win)
MAIN_HAS = sorted(b for b, d in MAIN_DIRS.items() if os.path.exists(d + '/items.parquet'))
assert len(MAIN_HAS) == len(MAIN), 'missing main item tables: ' + str(sorted(set(MAIN) - set(MAIN_HAS)))
MAIN_HASH = set().union(*[hashes(MAIN_DIRS[b]) for b in MAIN])

def cells(b, d):
    """Accuracy per named configuration: trials averaged per item, then items; only 0/1-scored benchmarks."""
    S = pd.read_parquet(d + '/subjects.parquet'); R = pd.read_parquet(d + '/responses.parquet', columns=['subject_id', 'item_id', 'response'])
    y = pd.to_numeric(R.response, errors='coerce'); R = R.assign(y=y).dropna(subset=['y'])
    if not len(R) or not R.y.between(0, 1).all() or R.y.isin([0, 1]).mean() < 0.95: return None, None
    S = S[S.normalized_name.notna()] if 'normalized_name' in S else S.iloc[:0]
    it = R[R.subject_id.isin(S.subject_id)].groupby(['subject_id', 'item_id']).y.mean().reset_index()
    g = it.groupby('subject_id').y.agg(['mean', 'size']).join(S.set_index('subject_id'))
    return g, it

log, elig, allcells = [], [], []
NEWHASH = {}
for b, d in sorted({**OLD, **NEW}.items()):
    g, it = cells(b, d)
    if g is None: continue
    if b in NEW: NEWHASH[b] = hashes(d)
    for n, r in g.groupby('normalized_name'):   # every named benchmark feeds the data-only predictor
        top = r.sort_values('size', ascending=False).index[0]
        allcells.append(dict(bench=b, name=n, acc=float(r.loc[top, 'mean'])))
    if b not in NEW: continue
    h = hashes(d); overlap = len(h & MAIN_HASH) / max(1, len(h))
    if overlap >= 0.10: log.append(dict(bench=b, excluded='shares items with the main evaluation', overlap=round(overlap, 3))); continue
    if not os.path.exists(d + '/items.parquet'): log.append(dict(bench=b, excluded='no item table')); continue
    I = pd.read_parquet(d + '/items.parquet', columns=['item_id', 'content', 'item_features'])
    I = I[I.item_id.isin(set(it.item_id))]
    samp = I.sample(min(10, len(I)), random_state=0)   # as in the main evaluation
    texts = [(str(c)[:600] + (' ' + str(f) if f is not None and str(f) not in ('', 'None', '{}', 'nan') else '')) for c, f in zip(samp.content, samp.item_features)]
    for n, r in g[g['size'] >= 50].groupby('normalized_name'):
        s = sorted(r.index, key=lambda i: (-r.loc[i, 'size'], str(i)))[0]   # most items, then smallest id: no use of accuracy
        subj = {k: ('' if pd.isna(r.loc[s].get(k)) else str(r.loc[s].get(k))) for k in
                ['normalized_name', 'provider', 'release_date', 'access_date', 'harness', 'harness_version', 'reasoning_effort', 'subject_features_extra']}
        elig.append(dict(bench=b, subject_id=str(s), subject=subj, acc=float(r.loc[s, 'mean']), n=int(r.loc[s, 'size']), texts=texts,
                         configs_for_name=len(r), item_overlap_with_main=round(overlap, 3)))

main_names = {x['subject']['normalized_name'] for x in json.load(open('data/llm/prior_eval_all.json'))}
for e in elig: e['name_in_main_eval'] = e['subject']['normalized_name'] in main_names
order = sorted(range(len(elig)), key=lambda i: (elig[i]['bench'], elig[i]['subject_id'])); random.Random(0).shuffle(order)
run = [dict(bench=elig[i]['bench'], subject_id=elig[i]['subject_id']) for i in order[:400]]

# data-only predictor (Table 1, alternating ridge): subject level from every other 0/1 benchmark, target benchmark left out
LC = pd.read_parquet('data/long.parquet', columns=['bench', 'normalized_name', 'y']).groupby(['normalized_name', 'bench']).y.mean().reset_index()
allcells += [dict(bench=b, name=n, acc=float(y)) for n, b, y in zip(LC.normalized_name, LC.bench, LC.y) if b not in NEW]
C = pd.DataFrame(allcells); nid = {n: k for k, n in enumerate(C.name.unique())}; bid = {b: k for k, b in enumerate(C.bench.unique())}
ni, bj = C.name.map(nid).values, C.bench.map(bid).values; z = np.log(np.clip(C.acc.values, .02, .98) / (1 - np.clip(C.acc.values, .02, .98)))
def fit(keep, lam=1.0):
    a, bb, mu = np.zeros(len(nid)), np.zeros(len(bid)), z[keep].mean()
    for _ in range(50):
        a = np.bincount(ni[keep], (z - mu - bb[bj])[keep], len(nid)) / (np.bincount(ni[keep], None, len(nid)) + lam)
        bb = np.bincount(bj[keep], (z - mu - a[ni])[keep], len(bid)) / (np.bincount(bj[keep], None, len(bid)) + lam)
    return mu, a, np.bincount(ni[keep], None, len(nid))
def related(b):   # the target and its variants: declared families plus new benchmarks sharing >= 10% of items
    fam = set().union(*[set(f) for f in SEM['variants'] if b in f]) | {b}
    h = NEWHASH.get(b, set())
    return fam | {o for o, ho in NEWHASH.items() if h and len(h & ho) / max(1, len(h)) >= 0.10}
pred, REL = {}, {}
for b in sorted({e['bench'] for e in elig}):
    REL[b] = sorted(related(b)); mu, a, cnt = fit(~C.bench.isin(REL[b]).values)
    for e in [e for e in elig if e['bench'] == b]:
        n = e['subject']['normalized_name']; k = nid.get(n)
        pred[f"{b}|{e['subject_id']}"] = dict(p=float(1 / (1 + np.exp(-(mu + (a[k] if k is not None else 0.0))))), other_scores=int(cnt[k]) if k is not None else 0)
json.dump(elig, open(f'{OUT}/eligible.json', 'w')); json.dump(run, open(f'{OUT}/run_order.json', 'w')); json.dump(pred, open(f'{OUT}/data_pred.json', 'w'))
json.dump(log + [dict(bench=b, excluded=r) for b, r in SEM['exclude'].items()], open(f'{OUT}/excluded.json', 'w'), indent=0)
json.dump(REL, open(f'{OUT}/left_out_of_training.json', 'w'), indent=0)
json.dump(sorted(C.bench.unique()), open(f'{OUT}/training_benchmarks.json', 'w'), indent=0)
json.dump(dict(main_benchmarks_checked=MAIN_HAS, main_hashes=len(MAIN_HASH), overlap={e['bench']: e['item_overlap_with_main'] for e in elig}), open(f'{OUT}/overlap_check.json', 'w'), indent=0)
print('eligible pairs', len(elig), 'benchmarks', len({e['bench'] for e in elig}), '| run order', len(run), '| excluded benchmarks', len(log),
      '| names also in main eval', sum(e['name_in_main_eval'] for e in elig))
