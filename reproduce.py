"""Recompute every table in the technical report from cached model outputs, without an API key.
usage: python reproduce.py [--rerun]   (--rerun reruns every simulation first, into data/rerun/; see README)"""
import ast, glob, json, os, re, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, 'src')
RERUN = '--rerun' in sys.argv
LOG = 'data/rerun' if RERUN else 'data'
SEEDS = (5, 6, 7)
# The simulated draws are replayed from data/draws/ (identifiers only), so they do not depend on the platform, and the
# subject priors from data/llm/prior_cache.jsonl: no API call is made, and STRICT stops on any cache miss.
sim = lambda env, model, draws: (f'{env}STRICT=1 PRIOR_CACHE_ONLY=1 DRAWS_IN=data/draws/{draws}.jsonl SUBJ=1 '
                                 f'{sys.executable} src/sim_sh.py {model} cover data/llm/batches_opus')
ARMS = [('Neither term', 'NOPRIOR=1 NORATE=1 ', 'none'), ('Item ratings only', 'NOPRIOR=1 ', 'noprior'),
        ('Subject prior only', 'NORATE=1 ', 'norate'), ('Both (submitted)', '', 'full')]


def field(path, key):
    """A summary line of a sim_sh.py log (ALCS, PER_BUDGET, BY_BENCH), parsed as a Python literal."""
    v = re.search(rf'^{key} (.*)$', open(path).read(), re.M).group(1)
    return ast.literal_eval(re.sub(r"np\.str_\(('[^']*')\)", r'\1', v))


def alcs(fmt):
    """Per-draw ALCs over the three seeds; fmt has a {s} slot for the seed."""
    return np.concatenate([field(fmt.format(s=s), 'ALCS') for s in SEEDS])


def paired(dd):
    return f'{dd.mean():+.4f} ± {dd.std(ddof=1) / np.sqrt(len(dd)):.4f}  better in {np.mean(dd < 0) * 100:.0f}%'


def mean_of(path):
    """ALC averaged equally over subject-benchmark pairs, as the competition scores it."""
    rows = re.findall(r'^(\S+) (\d+) ([\d.]+) distinct', open(path).read(), re.M)
    n = np.array([int(r[1]) for r in rows]); v = np.array([float(r[2]) for r in rows])
    return float((n * v).sum() / n.sum()) if len(rows) else float('nan')


def table_prior():
    """Tables 1 and 3: subject prior on held-out training pairs, in labels a Laplace estimate would need to match it."""
    if not (os.path.exists('data/llm/prior_eval.json') and os.path.exists('submission/anchors.json')):
        print('Tables 1 and 3: run src/prior_eval_build.py and src/make_anchors.py first (they need measurement-db)'); return
    E = json.load(open('data/llm/prior_eval.json')); A = json.load(open('submission/anchors.json'))
    acc = np.array([r['acc'] for r in E]); bench = np.array([r['bench'] for r in E])
    anchor_pair = {i for a in A for i, r in enumerate(E)   # pairs that serve as calibration examples
                   if r['subject']['normalized_name'] == a['model'] and any(a['items'][0][:150] in t for t in r['texts'])}
    anchor_bench = {E[i]['bench'] for i in anchor_pair}
    held = np.array([i not in anchor_pair for i in range(len(E))])
    est = lambda tag: np.array([x[0] if isinstance(x, list) and x and isinstance(x[0], (int, float)) else np.nan
                                for x in json.load(open(f'data/llm/prior_eval_{tag}.json'))], float)
    H, S, O = est('hd5low'), est('sd5low'), est('of5_600')   # deployed prompt, five sample items each
    chain = np.where(~np.isnan(O), O, np.where(~np.isnan(S), S, H))   # deployed: Opus, then Sonnet, then Haiku

    def laplace_mse(a, n):   # E[((k+1)/(n+2) - a)^2], k ~ Binomial(n, a)
        return np.mean((n * a * (1 - a) + (1 - 2 * a) ** 2) / (n + 2) ** 2)

    labels = lambda a, mse: next(n for n in range(500) if laplace_mse(a, n) <= mse)

    def row(name, e, m, shrink=0.75):
        p = np.full(m.sum(), 0.5) if e is None else 0.5 + shrink * (e[m] - 0.5)
        mse = np.mean((p - acc[m]) ** 2); c = np.corrcoef(e[m], acc[m])[0, 1] if e is not None else float('nan')
        print(f'  {name:44s} pairs {m.sum():3d}  RMSE {np.sqrt(mse):.3f}  corr {c:5.2f}  labels {labels(acc[m], mse)}')

    common = held & ~np.isnan(H) & ~np.isnan(S) & ~np.isnan(O)
    print('Tables 1 and 3. Subject prior on held-out training pairs (calibration-example pairs excluded)')
    row('Constant 0.5', None, common)
    for name, e in (('Claude Haiku 5.5', H), ('Claude Sonnet 5.5', S), ('Claude Opus 5.5', O)):
        row(name, e, common)
    # deployed chain on every held-out pair, shrinkage chosen leave-one-benchmark-out, bootstrap over benchmarks
    m = held & ~np.isnan(chain); a, e, b = acc[m], chain[m], bench[m]; grid = (0.5, 0.6, 0.75, 0.9, 1.0); p = np.empty_like(e)
    for bb in set(b):
        tr = b != bb; w = min(grid, key=lambda w: np.mean((0.5 + w * (e[tr] - 0.5) - a[tr]) ** 2)); p[b == bb] = 0.5 + w * (e[b == bb] - 0.5)
    mse = np.mean((p - a) ** 2); rng = np.random.default_rng(0); ub = np.array(sorted(set(b))); R, L = [], []
    for _ in range(2000):
        idx = np.concatenate([np.where(b == x)[0] for x in rng.choice(ub, len(ub))]); ms = np.mean((p[idx] - a[idx]) ** 2)
        R.append(np.sqrt(ms)); L.append(labels(a[idx], ms))
    print(f'  {"Deployed: Opus, then Sonnet, then Haiku":44s} pairs {m.sum():3d}  RMSE {np.sqrt(mse):.3f} '
          f'[{np.percentile(R, 5):.3f}, {np.percentile(R, 95):.3f}]  corr {np.corrcoef(e, a)[0, 1]:5.2f}  '
          f'labels {labels(a, mse)} [{np.percentile(L, 5):.0f}, {np.percentile(L, 95):.0f}]')
    row('Opus, calibration benchmarks removed', O, common & ~np.isin(bench, list(anchor_bench)))
    sim = set(json.load(open('data/llm/items_to_rate.json'))) if os.path.exists('data/llm/items_to_rate.json') else set()
    row('Opus, simulation benchmarks removed', O, held & ~np.isnan(O) & ~np.isin(bench, list(sim)))
    # recall tests: the same pairs with the subject's identity hidden, or with the benchmark named
    R_, B_ = est('oredact'), est('obench'); m2 = held & ~np.isnan(O) & ~np.isnan(R_) & ~np.isnan(B_)
    print('Recall tests (Opus, same pairs)')
    for name, e in (('as deployed', O), ('subject identity hidden', R_), ('benchmark named', B_)):
        row(name, e, m2)
    table_prior_all(labels, sim)


def table_prior_all(labels, dev_sim):
    """Table 1 (top rows) and Appendix A: the deployed chain on every named public pair, shrinkage fixed at 0.75."""
    if not os.path.exists('data/llm/prior_eval_all.json'):
        print('Table 1, top rows: run src/prior_eval_build_all.py first (needs measurement-db)'); return
    E = json.load(open('data/llm/prior_eval_all.json')); O = json.load(open('data/llm/prior_eval_oall.json'))
    num = lambda x: x[0] if isinstance(x[0], (int, float)) else np.nan
    est = np.array([num(o) for o in O], float); skip = np.array([bool(o[1].get('skip')) for o in O])
    miss = [i for i in range(len(O)) if np.isnan(est[i]) and not skip[i]]   # Opus gave no number: Sonnet, then Haiku
    S, H = json.load(open('data/llm/prior_eval_smiss.json')), json.load(open('data/llm/prior_eval_hmiss.json'))
    for j, i in enumerate(miss):
        est[i] = num(S[j]) if not np.isnan(num(S[j])) else num(H[j])
    acc = np.array([e['acc'] for e in E]); bench = np.array([e['bench'] for e in E])
    calib = set(bench[skip]); dev = dev_sim | {r['bench'] for r in json.load(open('data/llm/prior_eval.json'))}
    # data-only priors from the public response tables, target pair left out: benchmark mean, and an additive logit model
    # of subject level plus benchmark level fitted by alternating ridge regression
    import pandas as pd
    R = pd.read_parquet('data/long.parquet', columns=['bench', 'normalized_name', 'y'])
    bm = R.groupby('bench').y.mean(); C = R.groupby(['normalized_name', 'bench']).y.mean().reset_index()
    nid = {n: k for k, n in enumerate(C.normalized_name.unique())}; bid = {b: k for k, b in enumerate(C.bench.unique())}
    ni, bj = C.normalized_name.map(nid).values, C.bench.map(bid).values
    y = np.clip(C.y.values, .02, .98); z = np.log(y / (1 - y)); cell = {(n, b): k for k, (n, b) in enumerate(zip(C.normalized_name, C.bench))}

    def two_factor(keep, lam=1.0):
        a, b, mu = np.zeros(len(nid)), np.zeros(len(bid)), z[keep].mean()
        for _ in range(50):
            a = np.bincount(ni[keep], (z - mu - b[bj])[keep], len(nid)) / (np.bincount(ni[keep], None, len(nid)) + lam)
            b = np.bincount(bj[keep], (z - mu - a[ni])[keep], len(bid)) / (np.bincount(bj[keep], None, len(bid)) + lam)
        return mu, a, b
    m = ~skip & ~np.isin(bench, list(calib)); a_ = acc[m]; sig = lambda v: 1 / (1 + np.exp(-v)); P1, P2 = [], []
    for i in np.where(m)[0]:
        n, b_ = E[i]['subject']['normalized_name'], E[i]['bench']; keep = np.ones(len(C), bool)
        if (n, b_) in cell: keep[cell[(n, b_)]] = False
        mu, a, b = two_factor(keep); lv = a[nid[n]] if n in nid else 0.0
        P1.append(sig(mu + lv)); P2.append(sig(mu + lv + b[bid[b_]]))
    print('Table 1 (top rows). Data-only priors on the pairs without a calibration example')
    for name, p in (('constant 0.5', np.full(m.sum(), 0.5)), ('mean accuracy, this benchmark left out', np.array([bm.drop(b).mean() for b in bench[m]])),
                    ("subject's level from other benchmarks", np.array(P1)), ("subject's + this benchmark's level (other subjects)", np.array(P2))):
        mse = np.mean((p - a_) ** 2); print(f'  {name:52s} pairs {m.sum():3d}  RMSE {np.sqrt(mse):.3f}  labels {labels(a_, mse)}')
    rng = np.random.default_rng(0)
    print('Tables 1 (top rows) and 3. Deployed chain on every named public pair, shrinkage 0.75 fixed in advance')
    for name, m in (('all benchmarks', ~skip), ('benchmarks with a calibration example', ~skip & np.isin(bench, list(calib))),
                    ('benchmarks without one', ~skip & ~np.isin(bench, list(calib))),
                    ('  of which never used in development', ~skip & ~np.isin(bench, list(calib)) & ~np.isin(bench, list(dev)))):
        a, b = acc[m], bench[m]; p = 0.5 + 0.75 * (est[m] - 0.5); mse = np.mean((p - a) ** 2); ub = np.unique(b); R, L = [], []
        for _ in range(2000):
            ix = np.concatenate([np.where(b == x)[0] for x in rng.choice(ub, len(ub))]); ms = np.mean((p[ix] - a[ix]) ** 2)
            R.append(np.sqrt(ms)); L.append(labels(a[ix], ms))
        print(f'  {name:40s} pairs {m.sum():3d}  benchmarks {len(ub):2d}  constant {np.sqrt(np.mean((0.5 - a) ** 2)):.3f}  '
              f'RMSE {np.sqrt(mse):.3f} [{np.percentile(R, 5):.3f}, {np.percentile(R, 95):.3f}]  corr {np.corrcoef(p, a)[0, 1]:.2f}  '
              f'labels {labels(a, mse)} [{np.percentile(L, 5):.0f}, {np.percentile(L, 95):.0f}]')


def table_online():
    """Figure 1, Section 6 and Table 7: online formative runs, draw-adjusted, with per-budget excess."""
    if not glob.glob('data/online/scoring_*.txt'):
        print('Table 7: Codabench scoring logs not found in data/online/ (run IDs in src/online_table.py)'); return
    from online_pairs import load
    import online_table as T
    print('Table 7, weighted excess. Online formative runs by version'); T.main()
    P = load(); print('Table 7. Online excess Brier per budget (b=0,1,3,7,15)')
    for name, runs in T.GROUPS:
        pb = np.array([p['pb'] for p in P if p['run'] in runs])
        print(f'  {name:38s}', ' '.join(f'{v:6.3f}' for v in (pb - pb[:, 5:6]).mean(0)[:5]))
    print('Section 6. Budget 0, paired: the baseline predicts one half, so it scores 0.250 on every pair')
    rng = np.random.default_rng(0)
    for name, runs in T.GROUPS[1:]:
        G = [p for p in P if p['run'] in runs]; by = {}
        for p in G: by.setdefault((p['subj'], p['bench']), []).append(p['pb'][0])   # a pair that recurs across runs counts once (mean of its runs)
        d = np.array([np.mean(v) for v in by.values()]) - 0.25; b = np.array([k[1] for k in by]); ub = np.unique(b)
        bs = [d[np.concatenate([np.where(b == x)[0] for x in rng.choice(ub, len(ub))])].mean() for _ in range(4000)]
        print(f'  {name:38s} pair-runs {len(G)}  distinct pairs {len(by)}  B0 {0.25 + d.mean():.3f} [{0.25 + np.percentile(bs, 5):.3f}, {0.25 + np.percentile(bs, 95):.3f}]  lower on {(d < 0).sum()}')


def table_ablate():
    """Table 2 and Figure 1: the submitted predictor on identical simulated draws, language-model terms switched off in turn."""
    d = f'{LOG}/ablate'
    base = alcs(f'{d}/v26_none_s{{s}}.log')
    print('Table 2. Submitted predictor on identical simulated draws (90): ALC, B0, paired difference vs neither term')
    for label, _, stem in [('Laplace on own labels', None, 'laplace')] + ARMS:
        log = f'{d}/{stem}_s{{s}}.log' if stem == 'laplace' else f'{d}/v26_{stem}_s{{s}}.log'
        x = alcs(log); b0 = np.mean([field(log.format(s=s), 'PER_BUDGET')[0] for s in SEEDS])
        print(f'  {label:20s} {x.mean():.4f}  B0 {b0:.3f}  {paired(x - base)} of draws')


def table_regimes():
    """Section 5 and Appendix B: the subject prior with four subjects per benchmark, and on benchmarks held out from all development."""
    print('Section 5 and Appendix B. Subject prior across regimes (paired over identical simulated draws)')
    M_, D_ = f'{LOG}/multi', f'{LOG}/disjoint'
    if os.path.exists(f'{M_}/full_s5.log'):
        dd = alcs(f'{M_}/full_s{{s}}.log') - alcs(f'{M_}/noprior_s{{s}}.log')
        print(f'  four subjects per benchmark  {paired(dd)} of {len(dd)} draws')
    if os.path.exists(f'{D_}/bb_prior_s5.log'):
        named = json.load(open('data/llm/disjoint_named.json')) if os.path.exists('data/llm/disjoint_named.json') else {}
        agg = {}
        for s in SEEDS:
            P, N = field(f'{D_}/bb_prior_s{s}.log', 'BY_BENCH'), field(f'{D_}/bb_none_s{s}.log', 'BY_BENCH')
            for b in P:
                a = agg.setdefault(b, [0.0, 0]); a[0] += (P[b][0] - N[b][0]) * P[b][1]; a[1] += P[b][1]
        for grp in (True, False):
            bs = [b for b in agg if named.get(b, 0) > 0.5] if grp else [b for b in agg if named.get(b, 0) <= 0.5]
            d = sum(agg[b][0] for b in bs) / max(1, sum(agg[b][1] for b in bs)); helped = sum(agg[b][0] < 0 for b in bs)
            print(f'  held-out, mostly {"named" if grp else "unnamed"} subjects ({len(bs)} benchmarks)  {d:+.4f}  helped on {helped} of {len(bs)}')
    H_ = f'{LOG}/disjoint' if RERUN else 'data/disjoint_pb'   # the same runs, with per-benchmark, per-budget output
    if os.path.exists(f'{H_}/bb_prior_s5.log'):
        print('Table 4. Prior minus no prior on the named held-out benchmarks: pairs, B0, B1, B3, B7, ALC')
        agg = {}
        for s in SEEDS:
            P, N = field(f'{H_}/bb_prior_s{s}.log', 'BY_BENCH'), field(f'{H_}/bb_none_s{s}.log', 'BY_BENCH')
            PP, NP = field(f'{H_}/bb_prior_s{s}.log', 'BY_BENCH_PB'), field(f'{H_}/bb_none_s{s}.log', 'BY_BENCH_PB')
            for b in P:
                a = agg.setdefault(b, [0, np.zeros(5)]); n = P[b][1]; a[0] += n
                a[1] += n * np.array([PP[f'{b}|{k}'] - NP[f'{b}|{k}'] for k in (0, 1, 3, 7)] + [P[b][0] - N[b][0]])
        for b, (n, v) in sorted(agg.items(), key=lambda kv: kv[1][1][4] / kv[1][0]):
            if named.get(b, 0) > 0.5:
                print(f'  {b:18s} {n:3d} ' + ' '.join(f'{x:+.4f}' for x in v / n))


SIMS = {   # report cell -> (command, log)
    'base300': ('USE_LLM=0 {py} src/sim_pp.py src/baselines/empirical_mean.py random', 'pp_baseline_mean_random.log'),
    'v5_300': ('{py} src/sim_pp.py submission/model_v5.py random', 'pp_v5_random.log'),
    'v6r_300': ('{py} src/sim_pp.py submission/model_v6.py random', 'pp_v6_random.log'),
    'v6c_300': ('{py} src/sim_pp.py submission/model_v6.py cover', 'pp_v6_cover.log'),
    'v6c_nollm': ('USE_LLM=0 {py} src/sim_pp.py submission/model_v6.py cover', 'pp_v6_cover_nollm.log'),
    'oracle': ('ORACLE_THETA=1 {py} src/sim_pp.py submission/model_v6.py cover', 'oracle1_300.log'),
    'base_full': ('FULL=1 EVALCAP=150 USE_LLM=0 {py} src/sim_pp.py src/baselines/empirical_mean.py random', 'ppfull_baseline_mean.log'),
    'v5_full': ('FULL=1 EVALCAP=150 USE_LLM=0 {py} src/sim_pp.py submission/model_v5.py random', 'ppfull_v5_random.log'),
    'v6r_full': ('FULL=1 EVALCAP=150 USE_LLM=0 {py} src/sim_pp.py submission/model_v6.py random', 'ppfull_v6_random.log'),
    'v6c_full': ('FULL=1 EVALCAP=150 USE_LLM=0 {py} src/sim_pp.py submission/model_v6.py cover', 'ppfull_v6_cover.log'),
}
SMALL = {'Baseline': ('USE_LLM=0 ', 'src/baselines/empirical_mean.py random', 'baseline_mean'),
         'v5': ('', 'submission/model_v5.py random', 'v5_random'),
         'v6, random reveals': ('', 'submission/model_v6.py random', 'v6_random'),
         'v6, coverage-first': ('', 'submission/model_v6.py cover', 'v6_cover')}


def rerun():
    if not os.path.exists('data/llm/items_to_rate.json'):
        sys.exit('--rerun needs the item files: run dl.py, src/assemble.py and src/rate_prep.py first (measurement-db is gated)')
    for d in ('small', 'ablate', 'multi', 'disjoint', 'lap'):
        os.makedirs(f'data/rerun/{d}', exist_ok=True)
    # smoothed baseline on the same draws (needs a labeling.py next to it, as on the platform)
    shutil.copy('src/baselines/laplace.py', 'data/rerun/lap/model.py'); shutil.copy('submission/labeling.py', 'data/rerun/lap/')
    jobs = [(c.format(py=sys.executable), f'data/rerun/{log}') for c, log in SIMS.values()]
    for s in SEEDS:
        for _, env, stem in ARMS:
            jobs.append((sim(env, 'submission/model.py', f'ablate_s{s}'), f'data/rerun/ablate/v26_{stem}_s{s}.log'))
        jobs.append((sim('', 'data/rerun/lap/model.py', f'ablate_s{s}'), f'data/rerun/ablate/laplace_s{s}.log'))
        for arm, env in (('full', ''), ('noprior', 'NOPRIOR=1 ')):   # four subjects per benchmark
            jobs.append((sim(env, 'submission/model.py', f'multi_s{s}'), f'data/rerun/multi/{arm}_s{s}.log'))
        for arm, env in (('prior', 'NORATE=1 '), ('none', 'NOPRIOR=1 NORATE=1 ')):   # benchmarks used in no development step
            jobs.append((sim(env + 'ITEMS=data/llm/items_disjoint.json ', 'submission/model.py', f'disjoint_s{s}'), f'data/rerun/disjoint/bb_{arm}_s{s}.log'))
    for env, args, stem in SMALL.values():
        for n in (2, 5, 10):
            for s in SEEDS:
                jobs.append((f'{env}NITEM=100 NSUB={n} SEED={s} {sys.executable} src/sim_pp.py {args}',
                             f'data/rerun/small/{stem}_n{n}_s{s}.log'))
    from concurrent.futures import ThreadPoolExecutor
    def run(job):
        print('running:', job[0], '>', job[1], flush=True)
        r = subprocess.run(f'{job[0]} > {job[1]} 2>&1', shell=True)
        if r.returncode:
            sys.exit(f'{job[1]} failed: see the log. Draw replay needs data/long.parquet built from the revision pinned in dl.py.')
    with ThreadPoolExecutor(int(os.environ.get('JOBS', '4'))) as ex:
        list(ex.map(run, jobs))


def table_new():
    """Section 5, new benchmarks: the subject prior on the 44 new benchmarks, item ratings off (src/sim_extra_build.py, sim_extra.py)."""
    S = 'data/llm/extra/sim'; arms = ('prior', 'noprior', 'laplace', 'empirical_mean')
    missing = [f'out_{a}_s{s}.json' for a in arms for s in SEEDS if not os.path.exists(f'{S}/out_{a}_s{s}.json')]
    if missing:
        print(f'New-benchmark table: {len(missing)} simulation outputs missing from {S}/. They hold per-pair scores on the gated measurement-db, so they are '
              'not shipped: run src/sim_extra_build.py, then src/sim_extra.py ARM SEED for each arm and seed 5, 6, 7 (no API call)'); return
    print('New-benchmark table (Section 5). With and without the subject prior on the new benchmarks, 90 draws, item ratings off')
    import runpy; runpy.run_path('src/sim_extra_analysis.py', run_name='__main__')


def table_sim():
    """Tables 5 and 6: pooling in simulation, overall and per benchmark."""
    g = lambda k: mean_of(f'{LOG}/{SIMS[k][1]}')
    sm = lambda stem, n: np.mean([mean_of(f'{LOG}/small/{stem}_n{n}_s{s}.log') for s in SEEDS])
    print('Table 5. Brier ALC in simulation: Baseline, v5, v6 random reveals, v6 coverage-first')
    print('  300 items, 12 benchmarks   ', ' '.join(f'{g(k):.4f}' for k in ('base300', 'v5_300', 'v6r_300', 'v6c_300')))
    for n in (2, 5, 10):
        print(f'  Online scale, {n:2d} subjects  ', ' '.join(f'{sm(stem, n):.4f}' for _, _, stem in SMALL.values()))
    print('  Full pools, no rating term ', ' '.join(f'{g(k):.4f}' for k in ('base_full', 'v5_full', 'v6r_full', 'v6c_full')))
    print(f'  v6 coverage-first without the rating term: {g("v6c_nollm"):.4f}   with true ability (oracle): {g("oracle"):.4f}')
    print('Table 6. ALC per benchmark (300 items): benchmark, subjects, baseline, v5, v6, v6 without rating term')
    per = {}
    for k in ('base300', 'v5_300', 'v6c_300', 'v6c_nollm'):
        for b, n, v in re.findall(r'^(\S+) (\d+) ([\d.]+)', open(f'{LOG}/{SIMS[k][1]}').read(), re.M):
            per.setdefault(b, [n]).append(float(v))
    for b, v in per.items():
        print(f'  {b:14s} {v[0]:>3s}', ' '.join(f'{x:.4f}' for x in v[1:]))


if __name__ == '__main__':
    if RERUN:
        rerun()
    table_prior(); print(); table_online(); print(); table_ablate(); print(); table_regimes(); print(); table_new(); print(); table_sim()
