"""Reliability pre-check at larger scale with a mocked API ($0): fresh worker processes per budget, random latency,
429s and hangs; reports latency (limit 240 s), exceptions, API calls and prior/rating coverage per budget.
usage: stress_platform.py MODEL_DIR OUT_JSON; env BENCHES SUBJ_PER_BENCH ITEMS_PER_PAIR WORKERS CACHE LAT_LO LAT_HI FAIL HANG SEED SLOW_S TOP_SLOW
     DET=1 (answers depend only on the request, so two model versions can be compared), PRED_OUT=1 (keep every prediction),
     PATCH='NAME=value,...' (override model constants), BUDGET_S (time budget to check request start times against)"""
import contextvars, hashlib, importlib.util, io, json, multiprocessing as mp, os, random, shutil, sys, tempfile, time, urllib.error, urllib.request
import numpy as np, pandas as pd

ENV = lambda k, d: type(d)(os.environ.get(k, d))
BUDGETS = [0, 1, 3, 7, 15, 31]


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


_OWNER = contextvars.ContextVar('owner', default=None)   # index of the prediction whose work this is (pool threads inherit it at submit)
_POOLED = contextvars.ContextVar('pooled', default=False)   # inside a model thread pool, i.e. not one request after another


def _mock_urlopen(log, rng, lat, fail, hang, T0):
    def urlopen(req, timeout=None, *a, **k):
        body = json.loads(req.data.decode()) if getattr(req, 'data', None) else {}
        msgs = body.get('messages', [{}]); content = msgs[-1].get('content', '')
        text = content if isinstance(content, str) else ''.join(c.get('text', '') for c in content)
        kind = 'prior' if 'percentage of this benchmark' in text else 'rating'
        h = hashlib.md5((body.get('model', '') + '\n' + text).encode()).hexdigest()[:12]   # identifies repeated requests, no text kept
        t0 = time.time(); u = rng.random()
        # [kind, seconds, outcome, request hash, start, end (s from worker start), owning prediction, issued from a thread pool]
        entry = [kind, None, 'in_flight', h, t0 - T0, None, _OWNER.get(), _POOLED.get()]
        log.append(entry)   # counted when issued
        def done(out): entry[1], entry[2], entry[5] = time.time() - t0, out, time.time() - T0
        if u < hang:
            time.sleep(min(timeout or 60, 61)); done('timeout')
            raise urllib.error.URLError('timed out')
        if u < hang + fail:
            time.sleep(rng.uniform(*lat)); done('429')
            raise urllib.error.HTTPError(req.full_url, 429, 'rate limited', {'retry-after': '2'}, None)
        time.sleep(rng.uniform(*lat)); done('ok')
        ans = int(h, 16) % 91 + 5 if os.environ.get('DET') == '1' else rng.randint(5, 95)
        return _Resp(json.dumps({'content': [{'type': 'text', 'text': str(ans)}],
                                 'usage': {'input_tokens': 1000, 'output_tokens': 5}}).encode())
    return urlopen


def _worker(args):
    model_dir, tmp, inputs, labeled, seed, lat, fail, hang = args
    os.environ['TMPDIR'] = tmp; tempfile.tempdir = tmp; os.chdir(tmp); os.environ['HOME'] = tmp
    T0 = time.time(); log = []; urllib.request.urlopen = _mock_urlopen(log, random.Random(seed), lat, fail, hang, T0)
    spec = importlib.util.spec_from_file_location('model', os.path.join(model_dir, 'model.py'))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    class _Pool(M.ThreadPoolExecutor):   # rating threads keep the index of the prediction that submitted them
        def submit(self, fn, *a, **k):
            ctx = contextvars.copy_context(); ctx.run(_POOLED.set, True)
            return super().submit(ctx.run, fn, *a, **k)
    M.ThreadPoolExecutor = _Pool
    if hasattr(M, '_submit'):   # models that copy the caller's context into pool threads: mark the copy as pooled too
        def _sub(ex, fn, *a):
            ctx = contextvars.copy_context(); ctx.run(_POOLED.set, True)
            return ex.submit(ctx.run, fn, *a)
        M._submit = _sub
    for kv in filter(None, os.environ.get('PATCH', '').split(',')):
        k, v = kv.split('='); setattr(M, k, type(getattr(M, k))(v))
    ev_texts = {M._text(i) for _, i in inputs}; RQ = []   # every _rating request: (target item?, cached?, outcome)
    _orig_rating = M._rating
    def _rating(text):
        cached = text in M._RATE
        at_cap = M._RATE_CALLS[0] >= M.PROC_RATE_MAX
        v = _orig_rating(text)
        RQ.append((text in ev_texts, cached, 'ok' if v is not None else ('cap' if at_cap and not cached else 'none')))
        return v
    M._rating = _rating
    T = {}   # (owning prediction, part) -> seconds spent inside it, to break down slow predictions
    def timed(name, fn):
        def w(*a, **k):
            t = time.time()
            try:
                return fn(*a, **k)
            finally:
                key = (_OWNER.get(), name); T[key] = T.get(key, 0.0) + time.time() - t
        return w
    PQ = {}   # subject-prior asks: answered with a number, empty, refused at the per-process cap, or not asked (lock held / run cap)
    _orig_cp = M._call_prior
    def _cp(*a, **k):
        v = _orig_cp(*a, **k)
        key = 'number' if v is not None else 'empty'; PQ[key] = PQ.get(key, 0) + 1
        return v
    M._call_prior = _cp
    _orig_ask = M._ask_once
    def _ask(base, suf, desc, texts):
        c0, n0, t = M._PROC_CALLS[0], PQ.get('number', 0) + PQ.get('empty', 0), time.time()
        v = _orig_ask(base, suf, desc, texts)
        if PQ.get('number', 0) + PQ.get('empty', 0) == n0:   # no request: which branch of _ask_once returned?
            lock = base + suf + '.lock'
            took_lock = os.path.exists(lock) and os.path.getmtime(lock) >= t - 0.01
            key = 'cap' if took_lock and c0 >= M.PROC_PRIOR_MAX else 'not_asked'; PQ[key] = PQ.get(key, 0) + 1
        return v
    M._ask_once = _ask
    for name in ('_subject_prior', '_global_llm', '_bench_z', '_rating'):
        if hasattr(M, name): setattr(M, name, timed(name, getattr(M, name)))
    slow = []
    times, errs, with_prior, named_calls, preds, win = [], 0, 0, 0, [], []
    for idx, inp in enumerate(inputs):
        _OWNER.set(idx)
        t0 = time.time(); p = None
        try:
            p = M.predict(inp, labeled)
            if not (0 <= p <= 1): errs += 1
        except Exception:
            errs += 1
        times.append(time.time() - t0); preds.append(p); win.append((t0 - T0, t0 - T0 + times[-1]))
        if times[-1] > ENV('SLOW_S', 60.0):   # slow call: requests this prediction started, and leftovers of earlier ones running meanwhile
            s0, s1 = t0 - T0, t0 - T0 + times[-1]
            own = [e for e in log if e[6] == idx]; main = [e for e in own if not e[7]]
            slow.append(dict(start=round(s0, 1), total=round(times[-1], 1),
                             parts={n: round(v, 1) for (o, n), v in T.items() if o == idx},   # parts nest: _rating also runs inside _global_llm and in threads
                             own_requests=len(own), own_serial=len(main), own_serial_s=round(sum(e[1] or 0 for e in main), 1),
                             own_by_kind={kd: sum(e[0] == kd for e in own) for kd in ('prior', 'rating')},
                             own_429=sum(e[2] == '429' for e in own), own_timeouts=sum(e[2] == 'timeout' for e in own),
                             own_running_at_return=sum(e[5] is None or e[5] > s1 for e in own), own_started_after_return=sum(e[4] > s1 for e in own),
                             earlier_running_meanwhile=sum(e[6] != idx and e[4] < s1 and (e[5] is None or e[5] > s0) for e in log)))
        if M._subject_name(inp[0]):   # was a prior cached right after this prediction (not proof the prediction used it)
            named_calls += 1
            with_prior += M._prior_value(M._subject_desc(inp[0]), inp[1]['benchmark_id']) is not None
    pairs = {(json.dumps(s, sort_keys=True), i['benchmark_id']) for s, i in inputs}
    got = named = 0
    for s, b in pairs:   # does this worker hold a prior for the pair?
        named += bool(M._subject_name(json.loads(s)))
        try:
            got += M._prior_value(M._subject_desc(json.loads(s)), b) is not None   # cached value only, no new call
        except Exception:
            pass
    texts = {M._text(i) for _, i in inputs}   # eval items this worker predicted: did each get a difficulty rating?
    rated = sum(M._RATE.get(t) is not None for t in texts)
    _OWNER.set(None)
    B = ENV('BUDGET_S', 150.0)   # requests a prediction started after it returned, or after its time budget
    after = dict(after_return=sum(e[6] is not None and e[4] > win[e[6]][1] for e in log),
                 after_budget=sum(e[6] is not None and e[4] > win[e[6]][0] + B for e in log),
                 unowned=sum(e[6] is None for e in log))
    log = [tuple(e) for e in log]
    rq = {}
    for tgt, cached, out in RQ:
        k = ('target' if tgt else 'calibration') + '_' + out; rq[k] = rq.get(k, 0) + 1   # entries still 'in_flight' were issued but had not returned when the worker finished
    return dict(after=after, preds=preds if os.environ.get('PRED_OUT') == '1' else None, pq=PQ, slow=slow, rq=rq, times=times, errs=errs, log=log, named_calls=named_calls, with_prior=with_prior, pairs=len(pairs), named=named, prior_pairs=got, rated=rated, rate_seen=max(1, len(texts)),
                proc_rate_calls=M._RATE_CALLS[0], proc_prior_calls=M._PROC_CALLS[0], rate_cap_hit=M._RATE_CALLS[0] >= M.PROC_RATE_MAX,
                prior_cap_hit=M._PROC_CALLS[0] >= M.PROC_PRIOR_MAX)


def main():
    model_dir, out_json = os.path.abspath(sys.argv[1]), sys.argv[2]
    rng = np.random.default_rng(ENV('SEED', 5)); NB, NS, NI = ENV('BENCHES', 30), ENV('SUBJ_PER_BENCH', 6), ENV('ITEMS_PER_PAIR', 60)
    K, CACHE = ENV('WORKERS', 8), os.environ.get('CACHE', 'shared')
    lat = (ENV('LAT_LO', 1.0), ENV('LAT_HI', 4.0)); fail, hang = ENV('FAIL', 0.05), ENV('HANG', 0.01)
    items = json.load(open('data/llm/items_to_rate.json')); items.update(json.load(open('data/llm/items_disjoint.json')))
    R = pd.read_parquet('data/long_dedup_mac.parquet') if os.path.exists('data/long_dedup_mac.parquet') else pd.read_parquet('data/long.parquet')
    R = R[R.bench.isin(list(items))]
    benches = list(rng.choice(sorted(items), min(NB, len(items)), replace=False))
    pairs = []
    for b in benches:
        tx = {x['item_id']: x['text'] for x in items[b]}; d = R[R.bench == b]
        cnt = d[d.item_id.isin(list(tx))].groupby("subject_id").item_id.nunique(); subs = list(cnt[cnt >= NI].index)   # every pair gets its full ITEMS_PER_PAIR items
        for s in rng.choice(subs, min(NS, len(subs)), replace=False):
            g = d[(d.subject_id == s) & d.item_id.isin(list(tx))].sample(frac=1, random_state=int(rng.integers(1e9))).head(NI)
            if len(g) < 10:
                continue
            subj = {'normalized_name': str(g.normalized_name.iloc[0]), 'provider': '', 'subject_features_extra': str(s)}
            mk = lambda i, b=b, tx=tx: {'item_content': tx[i], 'item_features': '', 'interactors': '', 'benchmark_id': 'b_' + b}
            h = len(g) // 2
            pairs.append(dict(subj=subj, stream=list(zip(g.item_id.iloc[:h], g.y.iloc[:h])), ev=[mk(i) for i in g.item_id.iloc[h:]], mk=mk, pos=0, got=0))
    spec = importlib.util.spec_from_file_location('lab', os.path.join(model_dir, 'labeling.py'))
    L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
    base = tempfile.mkdtemp(prefix='stress_'); clean = os.path.join(base, 'model')
    shutil.copytree(model_dir, clean, ignore=shutil.ignore_patterns('pooled_priors_*', '__pycache__')); model_dir = clean; labeled = []; report = dict(pairs=len(pairs), benchmarks=len(benches), workers=K, cache=CACHE,
                                                                            latency=lat, fail=fail, hang=hang, budgets={})
    ctx = mp.get_context('spawn')
    for n in BUDGETS:
        for p in pairs:   # acquisition in the coordinator
            while p['got'] < n and p['pos'] < len(p['stream']):
                i, y = p['stream'][p['pos']]; p['pos'] += 1
                c = {'labels_acquired': p['got'], 'labels_remaining': 31 - p['got'], 'max_labels': 31,   # allowance out of 31, as on the platform
                     'items_remaining': len(p['stream']) - p['pos'] + 1, 'benchmark_id': p['mk'](i)['benchmark_id'], 'subject_id': p['subj']['subject_features_extra']}
                if L.acquisition_function([p['subj'], p['mk'](i)], labeled=labeled, context=c):
                    p['got'] += 1; labeled.append([[p['subj'], p['mk'](i)], int(y)])
        inputs = [[p['subj'], it] for p in pairs for it in p['ev']]
        random.Random(n).shuffle(inputs)
        jobs = []
        for w in range(K):
            if CACHE == 'shared':
                tmp = os.path.join(base, 'shared'); md = model_dir
            else:   # nothing shared between workers: own temp dir, home, cwd and model copy
                tmp = os.path.join(base, f'b{n}w{w}'); md = os.path.join(tmp, 'sub'); shutil.copytree(model_dir, md, dirs_exist_ok=True)
            os.makedirs(tmp, exist_ok=True)
            jobs.append((md, tmp, inputs[w::K], labeled, 1000 * n + w, lat, fail, hang))
        t0 = time.time()
        with ctx.Pool(K) as pool:
            res = pool.map(_worker, jobs)
        times = np.concatenate([r['times'] for r in res]); log = [x for r in res for x in r['log']]
        report['budgets'][n] = dict(wall_s=round(time.time() - t0, 1), predict_calls=int(len(times)), max_s=round(float(times.max()), 2),
                                    p99_s=round(float(np.percentile(times, 99)), 2), over_240s=int((times > 240).sum()),
                                    exceptions=int(sum(r['errs'] for r in res)),
                                    api=dict(prior=sum(e[0] == 'prior' for e in log), rating=sum(e[0] == 'rating' for e in log),
                                             ok=sum(e[2] == 'ok' for e in log), errors=sum(e[2] in ('429', 'timeout') for e in log),
                                             in_flight=sum(e[2] == 'in_flight' for e in log)),
                                    reveals_min=min(p['got'] for p in pairs), reveals_mean=round(float(np.mean([p['got'] for p in pairs])), 1),
                                    pairs_short_of_budget=sum(p['got'] < n for p in pairs),
                                    prior_cached_after_prediction=round(sum(r['with_prior'] for r in res) / max(1, sum(r['named_calls'] for r in res)), 3),
                                    prior_coverage_named=round(sum(r['prior_pairs'] for r in res) / max(1, sum(r['named'] for r in res)), 3),
                                    rating_coverage=round(sum(r['rated'] for r in res) / max(1, sum(r['rate_seen'] for r in res)), 3),
                                    max_rate_calls_one_process=max(r['proc_rate_calls'] for r in res), max_prior_calls_one_process=max(r['proc_prior_calls'] for r in res),
                                    processes_at_rate_cap=sum(r['rate_cap_hit'] for r in res), processes_at_prior_cap=sum(r['prior_cap_hit'] for r in res),
                                    rating_requests={k: sum(r['rq'].get(k, 0) for r in res) for k in sorted({k for r in res for k in r['rq']})},
                                    prior_requests={k: sum(r['pq'].get(k, 0) for r in res) for k in ('number', 'empty', 'cap', 'not_asked')},
                                    slowest=sorted([x for r in res for x in r['slow']], key=lambda x: -x['total'])[:ENV('TOP_SLOW', 3)],
                                    repeated_api_requests=len(log) - len({e[3] for e in log}),
                                    requests_started={k: sum(r['after'][k] for r in res) for k in ('after_return', 'after_budget', 'unowned')},
                                    stream_len_min=min(len(p['stream']) for p in pairs), stream_len_mean=round(float(np.mean([len(p['stream']) for p in pairs])), 1))
        report['budgets'][n]['run_total_api_calls_so_far'] = sum(v['api']['prior'] + v['api']['rating'] for v in report['budgets'].values())
        print(n, report['budgets'][n], flush=True)
        if os.environ.get('PRED_OUT') == '1':
            report['budgets'][n]['preds'] = [r['preds'] for r in res]
    json.dump(report, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
