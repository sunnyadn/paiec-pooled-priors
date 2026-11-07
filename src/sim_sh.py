"""Online-regime simulator: each draw samples 7 benchmarks with NSUBS subjects each (default 2,2,1,1,1,1,1) and NITEM items,
keeping a benchmark only if enough subjects answered 70% of its sampled items; one labeled list is pooled across all pairs.
usage: sim_sh.py MODEL_PY random|cover [RDIR]
env: DRAWS, SEED, NITEM, NSUBS (subjects per benchmark), ITEMS (item file), SUBJ=1 (full subject attributes),
USE_LLM=0, NOPRIOR=1, NORATE=1 (switch terms off), PRIOR_CACHE_ONLY=1 (replay cached priors, no API), STRICT=1
(fail on a cache miss); diagnostics: FEAT, WORKERS, PRIORLOG, MAXCALLS, DISSIM, LATE_COLD, PREDLOG, SETATTR, LABELING, ACQ31, COLD_DISK, RATESTAT"""
import threading, sys, json, os, zlib, functools, glob, hashlib, importlib.util, random, tempfile, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location('model', sys.argv[1]); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
for kv in filter(None, os.environ.get('SETATTR', '').split(',')): setattr(M, kv.split('=')[0], type(getattr(M, kv.split('=')[0]))(kv.split('=')[1]))   # diagnostic: override module constants, e.g. RIDGE=10
POL = sys.argv[2]; rdir = sys.argv[3] if len(sys.argv) > 3 else 'data/llm/api_haiku'
B = [0, 1, 3, 7, 15, 31]; W = [.1, .2, .2, .2, .2, .1]
rat = {}
for f in glob.glob(rdir + '/*.json'):
    try:
        for x in json.load(open(f))['ratings']: rat[str(x['id'])] = float(x['p'])
    except Exception: pass
items_all = json.load(open(os.environ.get('ITEMS', 'data/llm/items_to_rate.json')))
txt2id = {}
for b in items_all:
    for x in items_all[b]:
        txt2id[(x['text'][:4000] + ' ').lower().strip()] = x['item_id']; txt2id[(x['text'][:4000] + ' ').strip()] = x['item_id']; txt2id[x['text'].strip()] = x['item_id']
use_llm = os.environ.get('USE_LLM', '1') == '1'
M._cfg = lambda: {'provider': 'cache', 'model': 'x', 'key': 'x'} if use_llm else {}
RMISS = [0]
def _rate_cached(text):   # cached Opus rating; counts misses
    v = rat.get(txt2id.get(text.strip()))
    RMISS[0] += v is None; return v
M._call = _rate_cached
M._disk_path = lambda: '/dev/null'
MISS = [0]; CALLS = [0]
if hasattr(M, '_call_prior'):   # subject prior: real API calls through the submission key, cached on disk
    try: _REAL = dict(json.load(open('submission/llm_config.example.json')), key=os.environ['ANTHROPIC_API_KEY'])
    except Exception: _REAL = None; os.environ['PRIOR_CACHE_ONLY'] = '1'   # no key: replay cached priors only
    _orig_cp = M._call_prior; _PC = 'data/llm/prior_cache.jsonl'
    _pc = {}
    if os.path.exists(_PC):
        for l in open(_PC):
            k, v = json.loads(l); _pc[k] = v
    def _cp(name, texts):
        k = hashlib.md5((getattr(M, '_ANCHOR_TXT', '') * getattr(M, 'USE_ANCHORS', 0) + name + '\n' + '\n'.join(texts)).encode()).hexdigest()
        if k not in _pc and (os.environ.get('PRIOR_CACHE_ONLY') == '1' or CALLS[0] >= int(os.environ.get('MAXCALLS', '1000000'))):
            MISS[0] += 1; return None   # replay only: never call the API (or the call budget is spent)
        if k not in _pc:
            CALLS[0] += 1; keep = M._cfg; M._cfg = lambda: _REAL
            try: v = _orig_cp(name, texts)
            except Exception: v = None
            M._cfg = keep
            _pc[k] = v
            with open(_PC, 'a') as f: f.write(json.dumps([k, v]) + '\n')
        return _pc[k]
    M._call_prior = _cp
if os.environ.get('NOPRIOR') == '1' and hasattr(M, '_call_prior'): M._call_prior = lambda *a, **k: None
if os.environ.get('NORATE') == '1': M._call = lambda text: None
RS = {}   # RATESTAT=1 (diagnostic): every _rating request classified by where its answer came from; unique (draw, text) counted apart
if os.environ.get('RATESTAT') == '1' and hasattr(M, '_rating'):
    _orig_rating = M._rating; _orig_call = M._call; _RS_LOCK = threading.Lock(); _RS_U = set(); _RS_TL = threading.local()
    def _w_call(text):   # per-thread flag: this request passed the cap and reached the (cached) rating call
        _RS_TL.called = True; return _orig_call(text)
    M._call = _w_call
    def _w_rating(text):
        was_mem = text in M._RATE; was_none = was_mem and M._RATE[text] is None; _RS_TL.called = False
        v = _orig_rating(text); called = _RS_TL.called
        kind = ('mem_none' if was_none else 'mem') if was_mem else (('called_hit' if v is not None else 'called_miss') if called else ('disk' if v is not None else 'capped'))
        with _RS_LOCK:
            RS[kind] = RS.get(kind, 0) + 1; RS['requests'] = RS.get('requests', 0) + 1
            if (globals().get('draw'), text) not in _RS_U: _RS_U.add((globals().get('draw'), text)); RS['unique_draw_text'] = RS.get('unique_draw_text', 0) + 1; RS['unique_' + kind] = RS.get('unique_' + kind, 0) + 1
        return v
    M._rating = _w_rating
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'normalized_name', 'trial', 'y'])
R = R.sort_values('trial').drop_duplicates(['bench', 'subject_id', 'item_id'])
R = R[R.bench.isin(list(items_all))]
DRAWS = int(os.environ.get('DRAWS', '30')); rng = np.random.default_rng(int(os.environ.get('SEED', '5')))
NITEM = int(os.environ.get('NITEM', '60'))
byb = {b: g for b, g in R.groupby('bench')}
SUBJ = {}
if os.environ.get('SUBJ', '0') == '1':   # full subject attributes as in the online inputs
    for b in items_all:
        try:
            S = pd.read_parquet(f'data/mdb/{b}/formatted_tables/subjects.parquet')
        except Exception: continue
        for _, r in S.iterrows():
            SUBJ[(b, str(r.subject_id))] = {k: ('' if r.get(k) is None or str(r.get(k)) in ('None', 'nan', 'NaT') else str(r.get(k)))
                for k in ['normalized_name', 'provider', 'release_date', 'access_date', 'harness', 'harness_version', 'reasoning_effort', 'subject_features_extra']}
FEAT = {}
if os.environ.get('FEAT', '0') == '1':   # pass the benchmark's item_features through, as the online inputs do
    for b in items_all:
        try:
            I = pd.read_parquet(f'data/mdb/{b}/formatted_tables/items.parquet', columns=['item_id', 'item_features'])
            FEAT.update({str(i): ('' if f is None else str(f)) for i, f in zip(I.item_id, I.item_features)})
        except Exception: pass
    for b in items_all:
        for x in items_all[b]:
            k = (x['text'][:4000] + ' ' + FEAT.get(str(x['item_id']), '')).strip()
            txt2id[k] = x['item_id']; txt2id[k.lower()] = x['item_id']
benches = sorted(byb)
def make_item(tx, b, i):
    return {'item_content': tx[i], 'item_features': FEAT.get(str(i), ''), 'interactors': '', 'benchmark_id': 'b_' + b}
NSUB = [int(x) for x in os.environ.get('NSUBS', '2,2,1,1,1,1,1').split(',')]
TR = {'sp': [], 'single': 0, 'calls': 0}
if os.environ.get('PREDLOG'):   # diagnostic: record the prior each prediction actually received and whether it went through _single
    _o_sp, _o_single, _o_predict = M._subject_prior, M._single, M.predict
    def _w_sp(*a, **k):
        v = _o_sp(*a, **k); TR['sp'].append(None if v is None else round(float(v), 6)); return v
    def _w_single(*a, **k):
        TR['single'] += 1; return _o_single(*a, **k)
    def _w_predict(*a, **k):
        TR['calls'] += 1; return _o_predict(*a, **k)
    M._subject_prior, M._single, M.predict = _w_sp, _w_single, _w_predict
def _prior_peek(desc, bench):   # the prior _prior_value would return, read without writing it into the model's cache
    v, _ = M._read_prior(M._prior_file(desc, bench)); return v if v is not None else M._SPRIOR.get((desc, bench))
alcs = []; PB = {}; FL = []; SH = {}; BYB = {}; PBB = {}; PL = []; ACQ = {}; SHORT = [0, 0]
# DRAWS_OUT writes each draw as identifiers only (benchmark, subject, and per item its id and its row in data/long.parquet,
# in order); DRAWS_IN replays such a file, so the draws do not depend on how pandas breaks sort ties on another platform.
# The rows refer to data/long.parquet built by src/assemble.py from the measurement-db revision pinned in dl.py.
D_IN = [json.loads(l) for l in open(os.environ['DRAWS_IN'])] if os.environ.get('DRAWS_IN') else None
if D_IN:
    _L = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'normalized_name', 'trial', 'y'])
    _want = open('data/draws/long_parquet.sha256').read().split()[0]   # digest of the ordered table the row ids refer to
    if hashlib.sha256(pd.util.hash_pandas_object(_L[['bench', 'subject_id', 'item_id', 'trial', 'y']], index=True).values.tobytes()).hexdigest() != _want:
        sys.exit('data/long.parquet differs from the table the draw files refer to: rebuild it from the revision pinned in dl.py')
    def _row(b, s, i, r):
        r = _L.index.get_loc(r)   # row label as written by src/assemble.py
        if (_L.bench.iat[r], str(_L.subject_id.iat[r]), str(_L.item_id.iat[r])) != (b, s, i):
            sys.exit('data/long.parquet does not match the draw file: rebuild it from the pinned measurement-db revision')
        return int(_L.y.iat[r]), _L.normalized_name.iat[r]
for draw in range(len(D_IN) if D_IN else DRAWS):
    pairs = []
    if D_IN:
        for p_ in D_IN[draw]:
            b = p_['b']; tx = {str(x['item_id']): x['text'] for x in items_all[b]}; idict = functools.partial(make_item, tx, b)
            yy = lambda lst: [(i, _row(b, p_['s'], i, r)[0]) for i, r in lst]
            name = _row(b, p_['s'], *p_['ev'][0])[1]
            subj = SUBJ.get((b, p_['s'])) or {'normalized_name': str(name), 'subject_features_extra': p_['s']}
            pairs.append(dict(b=b, subj=subj, idict=idict, stream=yy(p_['stream']), ev=[(idict(i), y) for i, y in yy(p_['ev'])], pos=0, got=[]))
    bs = [] if D_IN else list(rng.choice(benches, 7, replace=False))
    for b, k in zip(bs, NSUB):
        ids = [x['item_id'] for x in items_all[b]]; tx = {x['item_id']: x['text'] for x in items_all[b]}
        ids = list(rng.choice(ids, min(NITEM, len(ids)), replace=False))
        d = byb[b][byb[b].item_id.isin(ids)]
        cnt = d.groupby('subject_id').item_id.nunique(); subs = cnt[cnt >= int(0.7 * len(ids))].index
        if len(subs) < k: continue
        subs = rng.choice(subs, k, replace=False)
        if os.environ.get('DISSIM') == '1' and k == 2:   # stress test: the strongest and the weakest subject on this benchmark
            acc_s = d.groupby('subject_id').y.mean(); cand = acc_s[acc_s.index.isin(cnt[cnt >= int(0.7 * len(ids))].index)]
            subs = np.array([cand.idxmax(), cand.idxmin()])
        for s in subs:
            g = d[d.subject_id == s].sample(frac=1, random_state=int(rng.integers(1e9)))
            h = len(g) // 2
            idict = functools.partial(make_item, tx, b)
            subj = SUBJ.get((b, str(s))) or {'normalized_name': str(g.normalized_name.iloc[0]), 'subject_features_extra': str(s)}
            pairs.append(dict(b=b, subj=subj, idict=idict, stream=list(zip(g.item_id.iloc[:h], g.y.iloc[:h])), ev=[(idict(i), y) for i, y in zip(g.item_id.iloc[h:], g.y.iloc[h:])], pos=0, got=[],
                              ids=dict(b=b, s=str(s), stream=[[str(i), int(r)] for i, r in zip(g.item_id.iloc[:h], g.index[:h])],
                                       ev=[[str(i), int(r)] for i, r in zip(g.item_id.iloc[h:], g.index[h:])])))
    if not pairs: continue
    if os.environ.get('DRAWS_OUT'):
        with open(os.environ['DRAWS_OUT'], 'a') as fo: fo.write(json.dumps([p['ids'] for p in pairs]) + '\n')
        if os.environ.get('DRAWS_ONLY') == '1': continue
    if hasattr(M, '_cache'): M._cache.clear()
    for attr in ('_SPRIOR', '_SEEN_TXT', '_RATE', '_SK2DESC'):   # a fresh evaluation run: no state from earlier draws
        if hasattr(M, attr): getattr(M, attr).clear()
    for attr in ('_PROC_CALLS', '_RATE_CALLS'):
        if hasattr(M, attr): getattr(M, attr)[0] = 0
    if os.environ.get('COLD_DISK') == '1' and hasattr(M, '_DISK'): M._DISK = None   # each draw starts from an empty rating disk cache (reused across its budgets)
    if hasattr(M, '_PDIRS'): M._PDIRS = [tempfile.mkdtemp(prefix='pp_prior_')]
    lab_mod = None
    if POL == 'cover':
        spec2 = importlib.util.spec_from_file_location('lab', os.environ.get('LABELING') or os.path.join(os.path.dirname(sys.argv[1]), 'labeling.py'))   # LABELING: diagnostic override
        lab_mod = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(lab_mod)
    labeled = []; alc = np.zeros(len(pairs)); racq = np.random.default_rng(zlib.crc32(str(draw).encode()))
    for n, w in zip(B, W):
        if n >= int(os.environ.get('LATE_COLD', '99')):   # diagnostic: workers at this budget share no prior cache with earlier ones
            for attr in ('_SPRIOR', '_SEEN_TXT', '_SK2DESC', '_cache'):
                if hasattr(M, attr): getattr(M, attr).clear()
            if hasattr(M, '_PROC_CALLS'): M._PROC_CALLS[0] = 0
            if hasattr(M, '_PDIRS'): M._PDIRS = [tempfile.mkdtemp(prefix='pp_prior_')]
        for p in pairs:
            while len(p['got']) < n and p['pos'] < len(p['stream']):
                i, y = p['stream'][p['pos']]; p['pos'] += 1
                rem_items = len(p['stream']) - p['pos'] + 1; rem_lab = 31 - len(p['got'])
                if lab_mod is None: take = racq.random() < min(1, rem_lab / rem_items)
                else:
                    ctx = {'labels_remaining': (31 if os.environ.get('ACQ31') == '1' else n) - len(p['got']), 'items_remaining': rem_items}   # ACQ31: allowance out of 31, as the platform
                    kw = {}; a = ACQ.setdefault(n, dict(calls=0, taken=0, forced=0, predict=0))
                    if 'prediction' in __import__('inspect').signature(lab_mod.acquisition_function).parameters:   # as the platform: predict before the label is revealed
                        kw['prediction'] = M.predict([p['subj'], p['idict'](i)], labeled); a['predict'] += 1
                    take = lab_mod.acquisition_function([p['subj'], p['idict'](i)], labeled=labeled, context=ctx, **kw)
                    a['calls'] += 1; a['forced'] += bool(take and rem_items <= ctx['labels_remaining']); a['taken'] += bool(take)
                if take:
                    p['got'].append(i); labeled.append([[p['subj'], p['idict'](i)], int(y)])
        WK = int(os.environ.get('WORKERS', '0'))
        if WK:   # emulate the platform: fresh worker state at every budget, eval calls spread over WK workers
            calls = [(pi, k) for pi, p in enumerate(pairs) for k in range(len(p['ev']))]
            random.Random(draw * 100 + n).shuffle(calls)
            PRED = {}
            for w_ in range(WK):
                if hasattr(M, '_cache'): M._cache.clear()
                for pi, k in calls[w_::WK]:
                    PRED[(pi, k)] = M.predict([pairs[pi]['subj'], pairs[pi]['ev'][k][0]], labeled)
        for pi, p in enumerate(pairs):
            TR.update(sp=[], single=0, calls=0)
            pr = np.array([PRED[(pi, k)] for k in range(len(p['ev']))]) if WK else np.array([M.predict([p['subj'], it], labeled) for it, _ in p['ev']]); y = np.array([y for _, y in p['ev']])
            if os.environ.get('PREDLOG'):   # diagnostic: the prior in force at this budget, the revealed labels, how many pairs share the benchmark
                _d = M._subject_desc(p['subj']) if M._subject_name(p['subj']) else None; _yy = dict(p['stream'])
                PL.append(dict(draw=draw, b=p['b'], pair=pi, n=n, got=len(p['got']), p=[round(float(v), 5) for v in pr], y=[int(v) for v in y],
                               prior=(_prior_peek(_d, 'b_' + p['b']) if _d else None), k=int(sum(_yy[i] for i in p['got'])),
                               nsub=sum(q['b'] == p['b'] for q in pairs), sp=list(TR['sp']),
                               single=TR['single'], calls=TR['calls'], revealed=[str(i) for i in p['got']],
                                                       acc=float(np.mean([v for _, v in p['stream']] + [v for _, v in p['ev']]))))   # diagnostic: predictions per pair and budget
            if n == 31: SHORT[0] += len(p['got']) < 31; SHORT[1] += 1   # pairs that ended with fewer than 31 labels
            e = np.mean((y - pr) ** 2); alc[pi] += w * e; PB.setdefault(n, []).append(e); PBB.setdefault((p['b'], n), []).append(e); SH.setdefault((n, sum(q["b"] == p["b"] for q in pairs)), []).append(e)
    for pi, p in enumerate(pairs): BYB.setdefault(p['b'], []).append(float(alc[pi]))
    if os.environ.get('PRIORLOG') and hasattr(M, '_prior_value'):   # the prior each pair ended with vs its true accuracy
        with open(os.environ['PRIORLOG'], 'a') as fl:
            for p in pairs:
                d = M._subject_desc(p['subj']) if M._subject_name(p['subj']) else None
                v = M._prior_value(d, p['idict'](p['stream'][0][0])['benchmark_id']) if d else None
                ys = [y for _, y in p['stream']] + [y for _, y in p['ev']]
                fl.write(json.dumps({'draw': draw, 'b': p['b'], 'named': d is not None, 'prior': v, 'acc': float(np.mean(ys))}) + '\n')
    alcs.append(float(np.mean(alc))); FL.extend([np.mean((np.array([y for _,y in p["ev"]])-np.mean([y for _,y in p["ev"]]))**2) for p in pairs])
if os.environ.get('PREDLOG'): json.dump(PL, open(os.environ['PREDLOG'], 'w'))
print('DRAWS', len(alcs), 'MEAN', round(float(np.mean(alcs)), 4), 'SE', round(float(np.std(alcs) / np.sqrt(len(alcs))), 4))
print('PER_BUDGET', {n: round(float(np.mean(v)), 4) for n, v in sorted(PB.items())})
print('FLOOR', round(float(np.mean(FL)),4))
print('PRIOR_CACHE_MISSES', MISS[0], 'API_CALLS', CALLS[0])
print('ACQ', ACQ)
print('RATING_CACHE_MISSES', RMISS[0])
print('PAIRS_SHORT_AT_31', SHORT[0], 'of', SHORT[1])
if RS: print('RATESTAT', dict(sorted(RS.items())))   # unique_<kind>: how each (draw, text) was first answered
if hasattr(M, '_H6STAT'): print('H6STAT fits, revealed, unrated, fits_without_rating', M._H6STAT)   # v27 diagnostic
if os.environ.get('STRICT') == '1' and MISS[0]: sys.exit(f'{MISS[0]} subject priors missing from the cache: the replay would not match the reported run')
print('BY_BENCH', {b: (round(float(np.mean(v)), 4), len(v)) for b, v in sorted(BYB.items())})
print('ALCS', [round(a,5) for a in alcs])
print('BY_BENCH_PB', {f'{b}|{n}': round(float(np.mean(v)), 4) for (b, n), v in sorted(PBB.items())})
print('BY_NSUB', {k: (round(float(np.mean(v)),4), round(float(np.max(v)),3)) for k, v in sorted(SH.items())})
