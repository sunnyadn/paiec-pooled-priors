"""Free screen (10-30): how well do two signals rank two subjects on the same benchmark?
LLM: difference of the deployed subject priors (shrunk 0.75), as used by _prior_shift in the submission.
Data: difference of subject levels from an additive logit model (subject + benchmark) fitted on all public
response tables WITH THE TARGET BENCHMARK LEFT OUT. Truth: difference of the two subjects' logit accuracies.
Pairs: all 765 named public pairs (src/prior_eval_build_all.py), calibration pairs skipped; every pair of subjects
sharing a benchmark. Reports correlation and sign agreement, overall and on benchmarks with no calibration example."""
import json, itertools, numpy as np, pandas as pd
D = 'data/llm/'
E = json.load(open(D + 'prior_eval_all.json')); O = json.load(open(D + 'prior_eval_oall.json'))
S, H = json.load(open(D + 'prior_eval_smiss.json')), json.load(open(D + 'prior_eval_hmiss.json'))
num = lambda x: x[0] if isinstance(x[0], (int, float)) else np.nan
est = np.array([num(o) for o in O]); skip = np.array([bool(o[1].get('skip')) for o in O])
miss = [i for i in range(len(O)) if np.isnan(est[i]) and not skip[i]]
for j, i in enumerate(miss): est[i] = num(S[j]) if not np.isnan(num(S[j])) else num(H[j])
acc = np.array([e['acc'] for e in E]); bench = np.array([e['bench'] for e in E]); name = np.array([e['subject']['normalized_name'] for e in E])
calib = set(bench[skip]); lg = lambda p: np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98)))
R = pd.read_parquet('data/long.parquet', columns=['bench', 'normalized_name', 'y'])
C = R.groupby(['normalized_name', 'bench']).y.mean().reset_index()
nid = {n: k for k, n in enumerate(C.normalized_name.unique())}; bid = {b: k for k, b in enumerate(C.bench.unique())}
ni, bj, z = C.normalized_name.map(nid).values, C.bench.map(bid).values, lg(C.y.values)
def fit(keep, lam=1.0):
    a, b, mu = np.zeros(len(nid)), np.zeros(len(bid)), z[keep].mean()
    for _ in range(50):
        a = np.bincount(ni[keep], (z - mu - b[bj])[keep], len(nid)) / (np.bincount(ni[keep], None, len(nid)) + lam)
        b = np.bincount(bj[keep], (z - mu - a[ni])[keep], len(bid)) / (np.bincount(bj[keep], None, len(bid)) + lam)
    return a
rows = []
for bb in sorted(set(bench[~skip])):
    idx = list(np.where((bench == bb) & ~skip)[0])
    if len(idx) < 2: continue
    keep = C.bench.values != bb; a = fit(keep); seen = set(C.normalized_name[keep])
    for i, j in itertools.combinations(idx, 2):
        dat = a[nid[name[i]]] - a[nid[name[j]]] if name[i] in seen and name[j] in seen else np.nan
        rows.append((bb, bb in calib, lg(acc[i]) - lg(acc[j]), lg(0.5 + 0.75 * (est[i] - 0.5)) - lg(0.5 + 0.75 * (est[j] - 0.5)), dat))
df = pd.DataFrame(rows, columns=['b', 'calib', 'true', 'llm', 'dat'])
for nm, d in (('all benchmarks', df), ('no calibration example', df[~df.calib])):
    d = d[d.dat.notna()]; print(f'{nm}: {len(d)} subject pairs on {d.b.nunique()} benchmarks')
    for col, v in (('llm', d.llm), ('data', d.dat), ('mean of both', (d.llm + d.dat) / 2)):
        print(f'  {col:13s} corr {np.corrcoef(v, d.true)[0, 1]:.2f}  sign agreement {np.mean(np.sign(v) == np.sign(d.true)):.2f}')
