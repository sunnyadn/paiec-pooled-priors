"""CHI-Bench check: the deployed prior (submission/model_v26.py _call_prior, Opus 5.5 chain, 8 examples) on the 28 named
configurations that ran the full 75-task set; groups by release date (before 2026-06, June, from 2026-07: after Opus 5.5's documented training cutoff). Descriptive only.
usage: ANTHROPIC_API_KEY=... prior_eval_chibench.py [run|analyse]"""
import hashlib, json, os, sys, importlib.util
import numpy as np, pandas as pd
D = 'data/mdb/chi_bench/formatted_tables/'
OUT = 'data/llm/prior_eval_chibench.json'
MP = 'submission/model_v26.py' if os.path.exists('submission/model_v26.py') else 'submission/model.py'   # the release ships the same predictor as model.py
spec = importlib.util.spec_from_file_location('m', MP); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
assert len(M.ANCHORS) == 8, 'build the 8 calibration examples first (src/make_anchors.py, see README)'
I = pd.read_parquet(D + 'items.parquet'); S = pd.read_parquet(D + 'subjects.parquet').set_index('subject_id')
R = pd.read_parquet(D + 'responses.parquet').merge(I[['item_id', 'raw_item_id']], on='item_id')
R = R.join(S[['normalized_name']], on='subject_id'); R = R[R.normalized_name.notna()]
full = frozenset(I.raw_item_id.unique()); sets = R.groupby('subject_id').raw_item_id.apply(frozenset)
CONF = sorted(s for s in sets.index if sets[s] == full)
ACC = R.groupby(['subject_id', 'raw_item_id']).response.mean().groupby('subject_id').mean()   # trials averaged per task, then tasks
first = I.drop_duplicates('raw_item_id').set_index('raw_item_id').content
fam = I.drop_duplicates('raw_item_id').set_index('raw_item_id').item_features.astype(str).str.extract(r'family=([a-z_]+)')[0]
md5 = lambda r: hashlib.md5(str(r).encode()).hexdigest()
PICK = [r for f, k in (('care_management', 2), ('prior_auth_provider', 2), ('prior_auth_um', 1))   # all three task families
        for r in sorted((x for x in full if fam[x] == f), key=md5)[:k]]
SAMPLE = [str(first[r]) for r in PICK]   # same 5 tasks for every configuration
group = lambda d: 'earlier' if d < '2026-06' else ('june' if d < '2026-07' else 'after')
ORDER = {'after': 0, 'earlier': 1, 'june': 2}   # call order: the 3 main targets first, then the earlier group, then June
CONF = sorted(CONF, key=lambda c: (ORDER[group(str(S.loc[c, 'release_date']))], c))

MAXREQ = 90   # hard cap on API requests (fallback chain and retries included)
VIA_CLI = os.environ.get('VIA') == 'cli'   # same prompt and model chain through the Claude Code CLI on the subscription (not the deployed API path)
if VIA_CLI:
    OUT = OUT.replace('.json', '_cli.json')
    import subprocess, re
    DROP_ENV = ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX',
                'CLAUDE_CODE_USE_FOUNDRY', 'ANTHROPIC_BEDROCK_BASE_URL', 'ANTHROPIC_VERTEX_BASE_URL', 'AWS_BEARER_TOKEN_BEDROCK')
    SUB_ENV = dict({k: v for k, v in os.environ.items() if k not in DROP_ENV and k != 'MAX_THINKING_TOKENS'}, CLAUDE_CODE_MAX_OUTPUT_TOKENS='2048',
                   CLAUDE_CODE_EFFORT_LEVEL='low', CLAUDE_CODE_MAX_RETRIES='0')   # subscription login only; settings of the deployed API call (Opus 5.5 always thinks, so thinking stays on)
    def cli_prior(desc, texts, log):
        body = "\n\n".join("<item>\n" + t[:600] + "\n</item>" for t in texts)
        user = M._ANCHOR_TXT + "\n\nNow the benchmark to estimate:\n\n" + body + "\n\n" + desc + "\n\nReply with only one number from 0 to 100: the percentage of this benchmark's items this system answers correctly."
        for m in M.PRIOR_MODELS:
            if len(log) >= MAXREQ: raise SystemExit('request cap reached')
            e = dict(model=m, status='sent'); log.append(e)
            try:
                r = subprocess.run(['claude', '-p', '--safe-mode', '--no-session-persistence', '--model', m, '--system-prompt', M.PRIOR_PROMPT,
                                    '--tools', '', '--effort', 'low', '--output-format', 'json'], input=user, capture_output=True, text=True,
                                   timeout=300, cwd='/tmp', env=SUB_ENV)
                d = json.loads(r.stdout); nums = re.findall(r"\d+(?:\.\d+)?", d.get('result') or '')
                e.update(status='ok', exit=r.returncode, stop_reason=d.get('stop_reason'), served=list(d.get('modelUsage', {})), is_error=d.get('is_error'), usage=d.get('usage'))
                if nums and r.returncode == 0 and not d.get('is_error') and d.get('stop_reason') == 'end_turn' and e['served'] == [m]:
                    return min(100.0, max(0.0, float(nums[-1]))) / 100.0
            except Exception as x:
                e.update(status='error', err=str(x)[:120])
            finally:
                json.dump(log, open(OUT + '.requests', 'w'))
        return None
BUDGET = 3.0   # USD, conservative: (UTF-8 bytes of the request + 1000) input tokens and 2,048 output tokens, at Opus 5.5's $4 / $20 per MTok for every model in the chain
SPENT = [0.0]
if sys.argv[1:2] == ['run']:
    if any(os.path.exists(OUT + x) for x in ('', '.partial', '.requests')): sys.exit('results already exist: refusing to run again')
    assert not any(k in os.environ for k in ('PP_ANCHORS', 'PP_PMODELS', 'PP_SPS')), 'deployed settings overridden'
if sys.argv[1:2] == ['run'] and VIA_CLI:
    LOG, rows = [], []
    try:
        for s in CONF:
            subj = {k: (None if pd.isna(v) else v) for k, v in S.loc[s].to_dict().items()}
            n0 = len(LOG); p = cli_prior(M._subject_desc(subj), SAMPLE, LOG)
            rows.append(dict(subject_id=s, name=subj['normalized_name'], harness=subj['harness'], released=str(subj['release_date']),
                             acc=float(ACC[s]), estimate=p, requests=LOG[n0:])); print(rows[-1]['name'], p, [x.get('served', x.get('err')) for x in LOG[n0:]], flush=True)
            json.dump(rows, open(OUT + '.partial', 'w'), indent=0)
    except SystemExit as x:
        print('stopped:', x, '| configurations done', len(rows), 'of', len(CONF)); sys.exit(1)
    json.dump(rows, open(OUT, 'w'), indent=0); print('requests', len(LOG))
elif sys.argv[1:2] == ['run']:
    import urllib.request
    LOG, _open = [], urllib.request.urlopen
    def logged(req, timeout=None, *a, **k):   # every request: model asked, outcome, usage
        if len(LOG) >= MAXREQ: raise SystemExit('request cap reached')
        cost = ((len(req.data) + 1000) * 4 + 2048 * 20) / 1e6
        if SPENT[0] + cost > BUDGET: raise SystemExit('budget cap reached')
        SPENT[0] += cost
        e = dict(model=json.loads(req.data)['model'], reserved_usd=round(cost, 6), status='sent'); LOG.append(e)
        json.dump(LOG, open(OUT + '.requests', 'w'))   # saved before the request and again with its outcome
        try:
            r = _open(req, timeout=timeout, *a, **k); body = r.read(); d = json.loads(body)
            e.update(status='ok', stop_reason=d.get('stop_reason'), served=d.get('model'), usage=d.get('usage'))
            import io; return io.BytesIO(body)
        except Exception as x:
            e.update(status='error', err=str(x)[:120]); raise
        finally:
            json.dump(LOG, open(OUT + '.requests', 'w'))
    urllib.request.urlopen = logged
    M._cfg = lambda: {'provider': 'anthropic', 'key': os.environ['ANTHROPIC_API_KEY']}
    rows = []
    try:
        for s in CONF:
            subj = {k: (None if pd.isna(v) else v) for k, v in S.loc[s].to_dict().items()}
            n0 = len(LOG); p = M._call_prior(M._subject_desc(subj), SAMPLE)
            rows.append(dict(subject_id=s, name=subj['normalized_name'], harness=subj['harness'], released=str(subj['release_date']),
                             acc=float(ACC[s]), estimate=p, requests=LOG[n0:])); print(rows[-1]['name'], p, [x.get('served', x.get('err')) for x in LOG[n0:]], flush=True)
            json.dump(rows, open(OUT + '.partial', 'w'), indent=0)   # saved after every configuration
    except SystemExit as x:
        print('stopped:', x, '| configurations done', len(rows), 'of', len(CONF)); sys.exit(1)
    json.dump(rows, open(OUT, 'w'), indent=0)
    print('requests', len(LOG), 'reserved USD %.4f' % SPENT[0])

rows = json.load(open(OUT))
early = [r for r in rows if group(r['released']) == 'earlier']
def earlier_mean(name):   # earlier configurations, each model weighted equally, all configurations of the target's own model left out
    by = {}
    for e in early:
        if e['name'] != name: by.setdefault(e['name'], []).append(e['acc'])
    return float(np.mean([np.mean(v) for v in by.values()]))
print('earlier-mean baseline for new models: %.6f (%d models)' % (earlier_mean(None), len({e['name'] for e in early})))
print('missing estimates:', sum(r['estimate'] is None for r in rows))
for g in ('earlier', 'june', 'after'):
    rs = [r for r in rows if group(r['released']) == g]
    a = np.array([r['acc'] for r in rs]); p = np.array([0.5 if r['estimate'] is None else 0.5 + 0.75 * (r['estimate'] - 0.5) for r in rs])
    m = np.array([earlier_mean(r['name']) for r in rs])
    print(f'{g}: configs {len(rs)} models {len({r["name"] for r in rs})} | RMSE prior {np.sqrt(np.mean((p - a) ** 2)):.3f}  constant 0.5 {np.sqrt(np.mean((0.5 - a) ** 2)):.3f}  earlier mean {np.sqrt(np.mean((m - a) ** 2)):.3f}')
    for r, pp in zip(rs, p):
        print(f'   {r["released"]} {r["name"]:30s} {r["harness"]:14s} true {r["acc"]:.3f}  prior {pp:.3f}')
