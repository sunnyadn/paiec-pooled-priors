"""Online-regime simulator matched to Codabench formative feedback: 9 pairs over 7 benchmarks
(5 with one subject), ~60 items each, one labeled list pooled across all pairs. Mean ALC over DRAWS draws.
usage: sim_online.py MODEL_PY random|cover [RDIR]; env DRAWS, SEED, USE_LLM, NITEM"""
import sys, json, os, zlib, importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location('model', sys.argv[1]); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
POL = sys.argv[2]; rdir = sys.argv[3] if len(sys.argv) > 3 else 'data/llm/api_haiku'
B = [0, 1, 3, 7, 15, 31]; W = [.1, .2, .2, .2, .2, .1]
import glob
rat = {}
for f in glob.glob(rdir + '/*.json'):
    try:
        for x in json.load(open(f))['ratings']: rat[str(x['id'])] = float(x['p'])
    except Exception: pass
items_all = json.load(open('data/llm/items_to_rate.json'))
txt2id = {}
for b in items_all:
    for x in items_all[b]:
        txt2id[(x['text'][:4000] + ' ').lower().strip()] = x['item_id']; txt2id[(x['text'][:4000] + ' ').strip()] = x['item_id']; txt2id[x['text'].strip()] = x['item_id']
use_llm = os.environ.get('USE_LLM', '1') == '1'
M._cfg = lambda: {'provider': 'cache', 'model': 'x', 'key': 'x'} if use_llm else {}
M._call = lambda text: rat.get(txt2id.get(text.strip()))
M._disk_path = lambda: '/dev/null'
if hasattr(M, '_call_prior'):   # subject prior: real API calls through the submission key, cached on disk
    import hashlib
    _REAL = json.load(open('submission/llm_config.json')); _orig_cp = M._call_prior; _PC = 'data/llm/prior_cache.jsonl'
    _pc = {}
    if os.path.exists(_PC):
        for l in open(_PC):
            k, v = json.loads(l); _pc[k] = v
    def _cp(name, texts):
        k = hashlib.md5((getattr(M, '_ANCHOR_TXT', '') * getattr(M, 'USE_ANCHORS', 0) + name + '\n' + '\n'.join(texts)).encode()).hexdigest()
        if k not in _pc:
            keep = M._cfg; M._cfg = lambda: _REAL
            try: v = _orig_cp(name, texts)
            except Exception: v = None
            M._cfg = keep
            _pc[k] = v
            with open(_PC, 'a') as f: f.write(json.dumps([k, v]) + '\n')
        return _pc[k]
    M._call_prior = _cp
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
alcs = []; PB = {}; FL = []
for draw in range(DRAWS):
    bs = list(rng.choice(benches, 7, replace=False)); nsub = [2, 2, 1, 1, 1, 1, 1]
    pairs = []
    for b, k in zip(bs, nsub):
        ids = [x['item_id'] for x in items_all[b]]; tx = {x['item_id']: x['text'] for x in items_all[b]}
        ids = list(rng.choice(ids, min(NITEM, len(ids)), replace=False))
        d = byb[b][byb[b].item_id.isin(ids)]
        cnt = d.groupby('subject_id').item_id.nunique(); subs = cnt[cnt >= int(0.7 * len(ids))].index
        if len(subs) < k: continue
        subs = rng.choice(subs, k, replace=False)
        for s in subs:
            g = d[d.subject_id == s].sample(frac=1, random_state=int(rng.integers(1e9)))
            h = len(g) // 2
            idict = (lambda tx, b: (lambda i: {'item_content': tx[i], 'item_features': FEAT.get(str(i), ''), 'interactors': '', 'benchmark_id': 'b_' + b}))(tx, b)
            subj = SUBJ.get((b, str(s))) or {'normalized_name': str(g.normalized_name.iloc[0]), 'subject_features_extra': str(s)}
            pairs.append(dict(b=b, subj=subj, idict=idict, stream=list(zip(g.item_id.iloc[:h], g.y.iloc[:h])), ev=[(idict(i), y) for i, y in zip(g.item_id.iloc[h:], g.y.iloc[h:])], pos=0, got=[]))
    if not pairs: continue
    if hasattr(M, '_cache'): M._cache.clear()
    lab_mod = None
    if POL == 'cover':
        spec2 = importlib.util.spec_from_file_location('lab', os.path.join(os.path.dirname(sys.argv[1]), 'labeling.py'))
        lab_mod = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(lab_mod)
    labeled = []; alc = np.zeros(len(pairs)); racq = np.random.default_rng(zlib.crc32(str(draw).encode()))
    for n, w in zip(B, W):
        for p in pairs:
            while len(p['got']) < n and p['pos'] < len(p['stream']):
                i, y = p['stream'][p['pos']]; p['pos'] += 1
                rem_items = len(p['stream']) - p['pos'] + 1; rem_lab = 31 - len(p['got'])
                if lab_mod is None: take = racq.random() < min(1, rem_lab / rem_items)
                else: take = lab_mod.acquisition_function([p['subj'], p['idict'](i)], labeled=labeled, context={'labels_remaining': n - len(p['got']), 'items_remaining': rem_items})
                if take:
                    p['got'].append(i); labeled.append([[p['subj'], p['idict'](i)], int(y)])
        for pi, p in enumerate(pairs):
            pr = np.array([M.predict([p['subj'], it], labeled) for it, _ in p['ev']]); y = np.array([y for _, y in p['ev']])
            e = np.mean((y - pr) ** 2); alc[pi] += w * e; PB.setdefault(n, []).append(e)
    alcs.append(float(np.mean(alc))); FL.extend([np.mean((np.array([y for _,y in p["ev"]])-np.mean([y for _,y in p["ev"]]))**2) for p in pairs])
print('DRAWS', len(alcs), 'MEAN', round(float(np.mean(alcs)), 4), 'SE', round(float(np.std(alcs) / np.sqrt(len(alcs))), 4))
print('PER_BUDGET', {n: round(float(np.mean(v)), 4) for n, v in sorted(PB.items())})
print('FLOOR', round(float(np.mean(FL)),4))
print('ALCS', [round(a,5) for a in alcs])
