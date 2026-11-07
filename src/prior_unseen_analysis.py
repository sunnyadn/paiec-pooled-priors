"""Analysis fixed before the run of prior_eval_unseen.py: paired MSE differences named - constant and named - hidden,
bootstrap over models, on all 34 pairs and on the 20 whose subject was released after Opus 4.5 (2025-11-24).
Failed or unparsed estimates are counted and fall back to 0.5 (the predictor's behaviour without a prior)."""
import json, sys, numpy as np
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'data/llm/prior_eval_unseen_claude-opus-4-5-20251101.json'))
key = lambda r: (r['bench'], r['subject_id'])
named = {key(r): r for r in R if not r['redact']}; hidden = {key(r): r for r in R if r['redact']}
shrink = lambda v: 0.5 if v is None else 0.5 + 0.75 * (v - 0.5)
print('failed or unparsed: named', sum(r['estimate'] is None for r in named.values()), 'hidden', sum(r['estimate'] is None for r in hidden.values()),
      '| stop reasons', sorted({str(r['usage'].get('stop_reason', r['usage'].get('err'))) for r in R}))
rng = np.random.default_rng(0)
for label, sel in (('all 34 (released 2025-09 or later)', lambda r: True), ('strict 20 (released after 2025-11-24)', lambda r: r['released'] > '2025-11-24')):
    ks = [k for k in named if sel(named[k])]
    a = np.array([named[k]['acc'] for k in ks]); pn = np.array([shrink(named[k]['estimate']) for k in ks]); ph = np.array([shrink(hidden[k]['estimate']) for k in ks])
    models = np.array([named[k]['name'] for k in ks]); um = np.unique(models)
    d1 = (pn - a) ** 2 - (0.5 - a) ** 2; d2 = (pn - a) ** 2 - (ph - a) ** 2
    B = []
    for _ in range(5000):
        idx = np.concatenate([np.where(models == m)[0] for m in rng.choice(um, len(um))])
        B.append((d1[idx].mean(), d2[idx].mean()))
    B = np.array(B)
    print(f'{label}: pairs {len(ks)} models {len(um)} | RMSE named {np.sqrt(np.mean((pn - a) ** 2)):.3f} hidden {np.sqrt(np.mean((ph - a) ** 2)):.3f} constant {np.sqrt(np.mean((0.5 - a) ** 2)):.3f}')
    for j, nm in enumerate(('named - constant', 'named - hidden')):
        d = (d1, d2)[j]
        print(f'   MSE diff {nm}: {d.mean():+.4f}  90% interval [{np.percentile(B[:, j], 5):+.4f}, {np.percentile(B[:, j], 95):+.4f}]  (negative = named better)')
