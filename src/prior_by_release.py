"""Diagnostic (10-30): deployed-chain estimate (Opus, shrink 0.75) vs constant 0.5 by subject release date.
Release date = subject['release_date'] from measurement-db subjects tables (data/llm/prior_eval_all.json). All 678 pairs with
an estimate, development benchmarks included (a trend, not a held-out test). Pairs without a release date are reported separately."""
import json, csv, numpy as np, collections
d = json.load(open('data/llm/prior_eval_all.json'))
est = {(r['benchmark'], r['subject_id']): r['opus_estimate'] for r in csv.DictReader(open('data/llm/prior_estimates_all.csv'))}
B = collections.defaultdict(list)
for x in d:
    e = est.get((x['bench'], x['subject_id']))
    if e in (None, '', 'nan'): continue
    p = 0.5 + 0.75 * (float(e) - 0.5); rd = x['subject'].get('release_date') or ''
    k = 'unknown' if not rd else ('<=2024' if rd < '2025' else ('2025H1' if rd < '2025-07' else ('2025H2' if rd < '2026' else '2026')))
    B[k].append((x['acc'], p))
for k in ['<=2024', '2025H1', '2025H2', '2026', 'unknown']:
    v = np.array(B[k]); a, p = v[:, 0], v[:, 1]
    print(k, len(a), 'RMSE estimate %.3f  constant 0.5 %.3f' % (np.sqrt(np.mean((a - p) ** 2)), np.sqrt(np.mean((a - .5) ** 2))))
