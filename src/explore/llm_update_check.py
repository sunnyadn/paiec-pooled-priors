"""Offline check (subscription CLI, exploratory): does re-asking Opus with the revealed answers in the prompt estimate a pair's accuracy
better than the deployed mechanical update (k + 4 m) / (n + 4)? One pair per new benchmark (seed 0), revealed items 1, 3, 7 in a
fixed random order; target = accuracy on the pair's other items (after the first 7). usage: llm_update_check.py list|run|analyse
(test-only env: LU_DIR, LU_MODEL, LU_PAIRS)"""
import json, os, random, re, subprocess, sys, threading, importlib.util, numpy as np
from concurrent.futures import ThreadPoolExecutor
D = os.environ.get('LU_DIR', 'data/llm/llm_update'); os.makedirs(D, exist_ok=True); NS = (1, 3, 7); LIST = 'data/llm/llm_update/list.json'
MODEL = os.environ.get('LU_MODEL', 'claude-opus-5-5')
P = {p['key']: p for p in json.load(open('data/llm/extra/sim_local/pairs.json'))}
txt = lambda c, f: str(c)[:600] + (' ' + str(f) if f is not None and str(f) not in ('', 'None', '{}') else '')   # as the prior's sample items
if sys.argv[1] == 'list':
    if os.path.exists(LIST): sys.exit('list exists')
    rng = random.Random(0); by = {}
    for k in sorted(P): by.setdefault(P[k]['bench'], []).append(k)
    out = []
    for b in sorted(by):
        k = rng.choice(by[b]); ids = [i for i, _ in P[k]['outcomes']]; random.Random(k).shuffle(ids)
        out.append(dict(key=k, order=ids[:7]))
    json.dump(out, open(LIST, 'w'), indent=0); print('pairs', len(out))
elif sys.argv[1] == 'run':
    spec = importlib.util.spec_from_file_location('m', 'submission/model_v26.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    E = {f"{e['bench']}|{e['subject_id']}": e for e in json.load(open('data/llm/extra/eligible.json'))}
    I = json.load(open('data/llm/extra/sim_local/items.json'))
    DROP = ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX',
            'CLAUDE_CODE_USE_FOUNDRY', 'ANTHROPIC_BEDROCK_BASE_URL', 'ANTHROPIC_VERTEX_BASE_URL', 'AWS_BEARER_TOKEN_BEDROCK', 'MAX_THINKING_TOKENS')
    ENV = dict({k: v for k, v in os.environ.items() if k not in DROP}, CLAUDE_CODE_MAX_OUTPUT_TOKENS='2048', CLAUDE_CODE_EFFORT_LEVEL='low', CLAUDE_CODE_MAX_RETRIES='0')
    OUT = f'{D}/results.json'; RES = json.load(open(OUT)) if os.path.exists(OUT) else {}; LOCK = threading.Lock()
    def ask(job):
        x, n = job; tag = f"{x['key']}#{n}"
        if tag in RES: return   # never repeat a recorded call (failures included)
        e, p = E[x['key']], P[x['key']]; y = dict(p['outcomes'])
        body = "\n\n".join("<item>\n" + t[:600] + "\n</item>" for t in e['texts'][:5])
        seen = "\n\n".join("<item>\n" + txt(*I[p['bench']][i]) + "\n</item>\nResult: " + ('correct' if y[i] else 'incorrect') for i in x['order'][:n])
        user = M._ANCHOR_TXT + "\n\nNow the benchmark to estimate:\n\n" + body + "\n\n" + M._subject_desc(e['subject']) + \
            f"\n\nThis system has already been scored on {n} item{'s' if n > 1 else ''} of this benchmark:\n\n" + seen + \
            "\n\nReply with only one number from 0 to 100: the percentage of this benchmark's items this system answers correctly."
        rec = dict(estimate=None)
        try:
            r = subprocess.run(['claude', '-p', '--safe-mode', '--no-session-persistence', '--model', MODEL, '--system-prompt', M.PRIOR_PROMPT,
                                '--tools', '', '--effort', 'low', '--output-format', 'json'], input=user, capture_output=True, text=True, timeout=300, cwd='/tmp', env=ENV)
            d = json.loads(r.stdout) if r.stdout.strip().startswith('{') else {}
            nums = re.findall(r"\d+(?:\.\d+)?", d.get('result') or '')
            ok = bool(nums) and r.returncode == 0 and not d.get('is_error') and list(d.get('modelUsage', {})) == [MODEL]
            rec.update(estimate=min(100., max(0., float(nums[-1]))) / 100 if ok else None, served=list(d.get('modelUsage', {})), usage=d.get('usage'),
                       exit=r.returncode, error=None if ok else ((d.get('result') or r.stderr or '')[-200:] or 'unparsed'))
        except Exception as ex:
            rec.update(error=f'{type(ex).__name__}: {ex}'[:200])
        with LOCK:
            RES[tag] = rec; json.dump(RES, open(OUT, 'w'), indent=0)   # kept after every call
        print(len(RES), tag, rec['estimate'], rec.get('error') or '', flush=True)
    L = json.load(open(LIST))[:int(os.environ.get('LU_PAIRS', '1000'))]
    with ThreadPoolExecutor(4) as ex: list(ex.map(ask, [(x, n) for x in L for n in NS]))
else:
    RES = json.load(open(f'{D}/results.json')); L = json.load(open(LIST))[:int(os.environ.get('LU_PAIRS', '1000'))]; rng = np.random.default_rng(0)
    BY = {k: r.get('answered_by') for k, r in json.load(open('data/llm/extra/results.json'))['results'].items()}
    def table(n, keep, fallback):
        rows = []
        for x in L:
            if not keep(x): continue
            p = P[x['key']]; y = dict(p['outcomes']); seen = set(x['order'][:7]); r = RES.get(f"{x['key']}#{n}") or {}
            acc = np.mean([v for i, v in p['outcomes'] if i not in seen]); k = sum(y[i] for i in x['order'][:n]); q = p['estimate']; m = 0.5 + 0.75 * (q - 0.5)
            mech, mech_raw = (k + 4 * m) / (n + 4), (k + 4 * q) / (n + 4)
            est = r.get('estimate')
            if est is None and not fallback: continue
            llm = mech if est is None else est   # failed call -> the deployed mechanical update
            rows.append((p['bench'], est is None, *[(v - acc) ** 2 for v in (llm, mech, mech_raw, q, m)]))
        if not rows: print(f'  n={n}: no pairs'); return
        b = np.array([r[0] for r in rows]); ub = np.unique(b); A = np.array([r[2:] for r in rows]); d = A[:, 0] - A[:, 1]
        bs = [np.mean(np.concatenate([d[b == u] for u in rng.choice(ub, len(ub))])) for _ in range(2000)]
        print(f'  n={n}: pairs {len(rows)} (failed->fallback {sum(r[1] for r in rows)}) | MSE llm {A[:, 0].mean():.4f}  mech {A[:, 1].mean():.4f}  '
              f'mech-unshrunk {A[:, 2].mean():.4f}  raw prior {A[:, 3].mean():.4f}  shrunk prior {A[:, 4].mean():.4f} | llm - mech {d.mean():+.4f} '
              f'[{np.percentile(bs, 5):+.4f}, {np.percentile(bs, 95):+.4f}] by benchmark, llm better in {np.mean(d < 0):.0%}')
    fails = [t for t, r in RES.items() if r.get('estimate') is None]
    print('recorded calls', len(RES), '| failed', len(fails), fails[:5])
    for name, keep, fb in (('PRIMARY all pairs, failures -> mechanical', lambda x: True, True), ('successful calls only', lambda x: True, False),
                           ('excluding pairs whose prior came from Sonnet/Haiku', lambda x: BY.get(x['key']) == 'claude-opus-5-5', True)):
        print(name)
        for n in NS: table(n, keep, fb)
