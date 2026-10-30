"""Recompute every table in the technical report from cached model outputs, without an API key.
usage: python reproduce.py [--rerun]   (--rerun reruns every simulation first, into data/rerun/; see README)"""
import glob, json, os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, 'src')
RERUN = '--rerun' in sys.argv
LOG = 'data/rerun' if RERUN else 'data'


def mean_of(path):
    """ALC averaged equally over subject-benchmark pairs, as the competition scores it."""
    rows = re.findall(r'^(\S+) (\d+) ([\d.]+) distinct', open(path).read(), re.M)
    n = np.array([int(r[1]) for r in rows]); v = np.array([float(r[2]) for r in rows])
    return float((n * v).sum() / n.sum()) if len(rows) else float('nan')


def table_prior():
    """Table 1: subject prior on held-out training pairs, in labels a Laplace estimate would need to match it."""
    if not (os.path.exists('data/llm/prior_eval.json') and os.path.exists('submission/anchors.json')):
        print('Table 1: run src/prior_eval_build.py and src/make_anchors.py first (they need measurement-db)'); return
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
    print('Table 1. Subject prior on held-out training pairs (calibration-example pairs excluded)')
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


def table_online():
    """Tables 2 and 3: online formative runs, draw-adjusted, with per-budget excess."""
    if not glob.glob('data/online/scoring_*.txt'):
        print('Tables 2-3: Codabench scoring logs not found in data/online/ (run IDs in src/online_table.py)'); return
    print('Table 2. Online formative runs'); subprocess.run([sys.executable, 'src/online_table.py'], check=True)
    from online_pairs import load
    import online_table as T
    P = load(); print('Table 3. Online excess Brier per budget (b=0,1,3,7,15)')
    for name, runs in T.GROUPS:
        pb = np.array([p['pb'] for p in P if p['run'] in runs])
        print(f'  {name:38s}', ' '.join(f'{v:6.3f}' for v in (pb - pb[:, 5:6]).mean(0)[:5]))


ABL = {'Neither term': 'NOPRIOR=1 NORATE=1 ', 'Item ratings only': 'NOPRIOR=1 ', 'Subject prior only': 'NORATE=1 ', 'Both (submitted)': ''}


def table_ablate():
    """Table 4: the submitted predictor on identical simulated draws, language-model terms switched off in turn."""
    d = f'{LOG}/ablate' if RERUN else 'data/ablate'
    stem = {'Neither term': 'none', 'Item ratings only': 'noprior', 'Subject prior only': 'norate', 'Both (submitted)': 'full'}
    get = lambda k: np.concatenate([np.array(eval(re.search(r'ALCS (\[.*\])', open(f'{d}/v26_{stem[k]}_s{s}.log').read()).group(1)))
                                    for s in (5, 6, 7)])
    b0 = lambda k: np.mean([eval(re.search(r'PER_BUDGET (\{.*\})', open(f'{d}/v26_{stem[k]}_s{s}.log').read()).group(1))[0] for s in (5, 6, 7)])
    base = get('Neither term')
    print('Table 4. Submitted predictor on identical simulated draws (90): ALC, B0, paired difference vs neither term')
    lap = np.concatenate([np.array(eval(re.search(r'ALCS (\[.*\])', open(f'{d}/laplace_s{s}.log').read()).group(1))) for s in (5, 6, 7)])
    dd = lap - base
    print(f'  {"Laplace on own labels":20s} {lap.mean():.4f}  B0 0.250  {dd.mean():+.4f} ± {dd.std(ddof=1) / np.sqrt(len(dd)):.4f}  better in {np.mean(dd < 0) * 100:.0f}% of draws')
    for k in ABL:
        x = get(k); dd = x - base
        print(f'  {k:20s} {x.mean():.4f}  B0 {b0(k):.3f}  {dd.mean():+.4f} ± {dd.std(ddof=1) / np.sqrt(len(dd)):.4f}  better in {np.mean(dd < 0) * 100:.0f}% of draws')


def table_regimes():
    """Table 5: the subject prior with four subjects per benchmark, and on benchmarks held out from all development."""
    get = lambda d, a: np.concatenate([np.array(eval(re.search(r'ALCS (\[.*\])', open(f'{d}/{a}_s{s}.log').read()).group(1))) for s in (5, 6, 7)])
    print('Table 5. Subject prior across regimes (paired over identical simulated draws)')
    M_, D_ = (f'{LOG}/multi', f'{LOG}/disjoint') if RERUN else ('data/multi', 'data/disjoint')
    if os.path.exists(f'{M_}/full_s5.log'):
        dd = get(M_, 'full') - get(M_, 'noprior')
        print(f'  four subjects per benchmark  {dd.mean():+.4f} ± {dd.std(ddof=1) / np.sqrt(len(dd)):.4f}  better in {np.mean(dd < 0) * 100:.0f}% of {len(dd)} draws')
    if os.path.exists(f'{D_}/bb_prior_s5.log'):
        named = json.load(open('data/llm/disjoint_named.json')) if os.path.exists('data/llm/disjoint_named.json') else {}
        agg = {}
        for s in (5, 6, 7):
            P = eval(re.search(r'BY_BENCH (\{.*\})', open(f'{D_}/bb_prior_s{s}.log').read()).group(1))
            N = eval(re.search(r'BY_BENCH (\{.*\})', open(f'{D_}/bb_none_s{s}.log').read()).group(1))
            for b in P:
                a = agg.setdefault(b, [0.0, 0]); a[0] += (P[b][0] - N[b][0]) * P[b][1]; a[1] += P[b][1]
        for grp in (True, False):
            bs = [b for b in agg if named.get(b, 0) > 0.5] if grp else [b for b in agg if named.get(b, 0) <= 0.5]
            d = sum(agg[b][0] for b in bs) / max(1, sum(agg[b][1] for b in bs)); helped = sum(agg[b][0] < 0 for b in bs)
            print(f'  held-out, mostly {"named" if grp else "unnamed"} subjects ({len(bs)} benchmarks)  {d:+.4f}  helped on {helped} of {len(bs)}')


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
    os.makedirs('data/rerun/small', exist_ok=True)
    jobs = [(c.format(py=sys.executable), f'data/rerun/{log}') for c, log in SIMS.values()]
    os.makedirs('data/rerun/ablate', exist_ok=True)
    for k, env in ABL.items():   # subject priors replayed from data/llm/prior_cache.jsonl, never the API
        stem = {'Neither term': 'none', 'Item ratings only': 'noprior', 'Subject prior only': 'norate', 'Both (submitted)': 'full'}[k]
        for s in (5, 6, 7):
            jobs.append((f'{env}STRICT=1 PRIOR_CACHE_ONLY=1 SEED={s} DRAWS=30 NITEM=120 SUBJ=1 {sys.executable} src/sim_sh.py submission/model.py cover data/llm/batches_opus',
                         f'data/rerun/ablate/v26_{stem}_s{s}.log'))
    for s in (5, 6, 7):   # smoothed baseline on the same draws (needs a labeling.py next to it, as on the platform)
        jobs.append((f'mkdir -p data/rerun/lap && cp src/baselines/laplace.py data/rerun/lap/model.py && cp submission/labeling.py data/rerun/lap/ && '
                     f'PRIOR_CACHE_ONLY=1 SEED={s} DRAWS=30 NITEM=120 SUBJ=1 {sys.executable} src/sim_sh.py data/rerun/lap/model.py cover data/llm/batches_opus',
                     f'data/rerun/ablate/laplace_s{s}.log'))
    for d in ('multi', 'disjoint'):
        os.makedirs(f'data/rerun/{d}', exist_ok=True)
    for s in (5, 6, 7):   # four subjects per benchmark, and benchmarks used in no development step (priors replayed)
        for arm, env in (('full', ''), ('norate', 'NORATE=1 '), ('noprior', 'NOPRIOR=1 '), ('none', 'NOPRIOR=1 NORATE=1 ')):
            jobs.append((f'{env}STRICT=1 PRIOR_CACHE_ONLY=1 NSUBS=4,4,4,4,4,4,4 SEED={s} DRAWS=10 NITEM=120 SUBJ=1 {sys.executable} src/sim_sh.py submission/model.py cover data/llm/batches_opus',
                         f'data/rerun/multi/{arm}_s{s}.log'))
        for arm, env in (('prior', 'NORATE=1 '), ('none', 'NOPRIOR=1 NORATE=1 ')):
            jobs.append((f'{env}STRICT=1 PRIOR_CACHE_ONLY=1 ITEMS=data/llm/items_disjoint.json SEED={s} DRAWS=30 NITEM=120 SUBJ=1 {sys.executable} src/sim_sh.py submission/model.py cover data/llm/batches_opus',
                         f'data/rerun/disjoint/bb_{arm}_s{s}.log'))
    for env, args, stem in SMALL.values():
        for n in (2, 5, 10):
            for s in (5, 6, 7):
                jobs.append((f'{env}NITEM=100 NSUB={n} SEED={s} {sys.executable} src/sim_pp.py {args}',
                             f'data/rerun/small/{stem}_n{n}_s{s}.log'))
    from concurrent.futures import ThreadPoolExecutor
    def run(job):
        print('running:', job[0], '>', job[1], flush=True)
        subprocess.run(f'{job[0]} > {job[1]} 2>&1', shell=True, check=True)
    with ThreadPoolExecutor(int(os.environ.get('JOBS', '4'))) as ex:
        list(ex.map(run, jobs))


def table_sim():
    """Table 4 (simulation, pooling) and Table A1 (per benchmark)."""
    g = lambda k: mean_of(f'{LOG}/{SIMS[k][1]}')
    sm = lambda stem, n: np.mean([mean_of(f'{LOG}/small/{stem}_n{n}_s{s}.log') for s in (5, 6, 7)])
    print('Table A2. Brier ALC in simulation: Baseline, v5, v6 random reveals, v6 coverage-first')
    print('  300 items, 12 benchmarks   ', ' '.join(f'{g(k):.4f}' for k in ('base300', 'v5_300', 'v6r_300', 'v6c_300')))
    for n in (2, 5, 10):
        print(f'  Online scale, {n:2d} subjects  ', ' '.join(f'{sm(stem, n):.4f}' for _, _, stem in SMALL.values()))
    print('  Full pools, no rating term ', ' '.join(f'{g(k):.4f}' for k in ('base_full', 'v5_full', 'v6r_full', 'v6c_full')))
    print(f'  v6 coverage-first without the rating term: {g("v6c_nollm"):.4f}   with true ability (oracle): {g("oracle"):.4f}')
    print('Table A1. ALC per benchmark (300 items): benchmark, subjects, baseline, v5, v6, v6 without rating term')
    per = {}
    for k in ('base300', 'v5_300', 'v6c_300', 'v6c_nollm'):
        for b, n, v in re.findall(r'^(\S+) (\d+) ([\d.]+)', open(f'{LOG}/{SIMS[k][1]}').read(), re.M):
            per.setdefault(b, [n]).append(float(v))
    for b, v in per.items():
        print(f'  {b:14s} {v[0]:>3s}', ' '.join(f'{x:.4f}' for x in v[1:]))


if __name__ == '__main__':
    if RERUN:
        rerun()
    table_prior(); print(); table_online(); print(); table_ablate(); print(); table_regimes(); print(); table_sim()
