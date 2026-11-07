"""Control analysis, fixed before the control run: Opus 4.5 named estimate vs constant 0.5 on earlier-released subjects
(before 2025-06) and on subjects released after it, both on benchmarks without a calibration example; bootstrap over models."""
import json, numpy as np
CAL = set(json.load(open('data/llm/unseen_manifest/calibration_benches.json')))
U = [r for r in json.load(open('data/llm/prior_eval_unseen_claude-opus-4-5-20251101_cli.json')) if not r['redact'] and r['bench'] not in CAL]
P = json.load(open('data/llm/prior_eval_unseen_claude-opus-4-5-20251101_cli_pre.json'))
shrink = lambda v: 0.5 if v is None else 0.5 + 0.75 * (v - 0.5)
rng = np.random.default_rng(0)

def boot(rows):
    a = np.array([r['acc'] for r in rows]); p = np.array([shrink(r['estimate']) for r in rows]); d = (p - a) ** 2 - (0.5 - a) ** 2
    m = np.array([r['name'] for r in rows]); um = np.unique(m)
    B, S = [], []
    for _ in range(5000):
        i = np.concatenate([np.where(m == x)[0] for x in rng.choice(um, len(um))])
        B.append(d[i].mean()); S.append(1 - np.mean((p[i] - a[i]) ** 2) / np.mean((0.5 - a[i]) ** 2))
    return a, p, d, um, np.array(B), np.array(S)

print('failed or unparsed in control:', sum(r['estimate'] is None for r in P))
res, obs = {}, {}
for label, rows in (('earlier-released (before 2025-06)', P), ('unseen, all (2025-09 or later)', U),
                    ('unseen, strict (after 2025-11-24)', [r for r in U if r['released'] > '2025-11-24'])):
    a, p, d, um, B, S = boot(rows); res[label] = B; res[label + ' skill'] = S
    skill = 1 - np.mean((p - a) ** 2) / np.mean((0.5 - a) ** 2)   # share of the constant's MSE removed (normalised to the baseline only)
    obs[label] = (d.mean(), skill)
    print(f'{label}: pairs {len(rows)} models {len(um)} | SD of accuracy {a.std():.3f} | RMSE named {np.sqrt(np.mean((p - a) ** 2)):.3f} constant {np.sqrt(np.mean((0.5 - a) ** 2)):.3f}'
          f' | MSE diff named - constant {d.mean():+.4f} [{np.percentile(B, 5):+.4f}, {np.percentile(B, 95):+.4f}] | skill {skill:+.2f} [{np.percentile(S, 5):+.2f}, {np.percentile(S, 95):+.2f}]')
for lab in ('unseen, all (2025-09 or later)', 'unseen, strict (after 2025-11-24)'):
    D = res[lab] - res['earlier-released (before 2025-06)']   # independent bootstraps of the two groups
    G = res['earlier-released (before 2025-06) skill'] - res[lab + ' skill']
    e = 'earlier-released (before 2025-06)'   # point estimates from the observed statistics; bootstrap only for the intervals
    print(f'gap {lab} - earlier: MSE diff {obs[lab][0] - obs[e][0]:+.4f} [{np.percentile(D, 5):+.4f}, {np.percentile(D, 95):+.4f}]  (positive = worse on unseen)'
          f' | skill earlier - unseen {obs[e][1] - obs[lab][1]:+.2f} [{np.percentile(G, 5):+.2f}, {np.percentile(G, 95):+.2f}]  (descriptive: task mix differs)')
