"""Route check (subscription, no key): 20 pairs of the 277 main pairs, list fixed by seed 0 before any call, asked through the same
Claude Code CLI path as the new-benchmark estimates (prior_eval_extra.py: 5 items, 600 chars, Opus 5.5 only, effort low); compared
paired with the API estimates of prior_eval_oall.json. usage: prior_cli_api_check.py list|run|analyse -> data/llm/cli_api_check/"""
import json, os, random, re, subprocess, sys, importlib.util, numpy as np
D = 'data/llm/cli_api_check'; os.makedirs(D, exist_ok=True)
E = json.load(open('data/llm/prior_eval_all.json')); O = json.load(open('data/llm/prior_eval_oall.json'))
num = lambda x: x[0] if isinstance(x[0], (int, float)) else None
if sys.argv[1] == 'list':
    if os.path.exists(f'{D}/pairs.json'): sys.exit('list exists')
    skip = [bool(o[1].get('skip')) for o in O]; calib = {e['bench'] for e, s in zip(E, skip) if s}
    pool = [i for i, (e, o) in enumerate(zip(E, O)) if not skip[i] and e['bench'] not in calib and num(o) is not None]
    pick = sorted(random.Random(0).sample(pool, 20))
    json.dump([dict(i=i, bench=E[i]['bench'], subject_id=E[i]['subject_id']) for i in pick], open(f'{D}/pairs.json', 'w'), indent=0)
    print('pool', len(pool), 'picked', pick)
elif sys.argv[1] == 'run':
    if os.path.exists(f'{D}/cli.json'): sys.exit('results exist')
    spec = importlib.util.spec_from_file_location('m', 'submission/model_v26.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    DROP = ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX',
            'CLAUDE_CODE_USE_FOUNDRY', 'ANTHROPIC_BEDROCK_BASE_URL', 'ANTHROPIC_VERTEX_BASE_URL', 'AWS_BEARER_TOKEN_BEDROCK', 'MAX_THINKING_TOKENS')
    ENV = dict({k: v for k, v in os.environ.items() if k not in DROP}, CLAUDE_CODE_MAX_OUTPUT_TOKENS='2048', CLAUDE_CODE_EFFORT_LEVEL='low', CLAUDE_CODE_MAX_RETRIES='0')
    out = []
    for p in json.load(open(f'{D}/pairs.json')):
        e = E[p['i']]; assert (e['bench'], e['subject_id']) == (p['bench'], p['subject_id'])
        body = "\n\n".join("<item>\n" + t[:600] + "\n</item>" for t in e['texts'][:5])
        user = M._ANCHOR_TXT + "\n\nNow the benchmark to estimate:\n\n" + body + "\n\n" + M._subject_desc(e['subject']) + \
            "\n\nReply with only one number from 0 to 100: the percentage of this benchmark's items this system answers correctly."
        r = subprocess.run(['claude', '-p', '--safe-mode', '--no-session-persistence', '--model', 'claude-opus-5-5', '--system-prompt', M.PRIOR_PROMPT,
                            '--tools', '', '--effort', 'low', '--output-format', 'json'], input=user, capture_output=True, text=True, timeout=300, cwd='/tmp', env=ENV)
        d = json.loads(r.stdout) if r.stdout.strip().startswith('{') else {}
        nums = re.findall(r"\d+(?:\.\d+)?", d.get('result') or '')
        ok = bool(nums) and r.returncode == 0 and not d.get('is_error') and list(d.get('modelUsage', {})) == ['claude-opus-5-5']
        out.append(dict(p, estimate=min(100., max(0., float(nums[-1]))) / 100 if ok else None, served=list(d.get('modelUsage', {})), usage=d.get('usage'), exit=r.returncode))
        print(len(out), p['bench'], out[-1]['estimate'], flush=True); json.dump(out, open(f'{D}/cli.json.partial', 'w'), indent=0)   # kept after every call
    json.dump(out, open(f'{D}/cli.json', 'w'), indent=0)
else:
    R = [r for r in json.load(open(f'{D}/cli.json')) if r['estimate'] is not None]
    cli = np.array([r['estimate'] for r in R]); api = np.array([num(O[r['i']]) for r in R]); acc = np.array([E[r['i']]['acc'] for r in R])
    d = cli - api; rng = np.random.default_rng(0); bs = [np.mean(rng.choice(d, len(d))) for _ in range(2000)]
    print(f'pairs {len(R)}/20 | CLI - API mean {d.mean():+.3f} [{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}], median {np.median(d):+.3f}, '
          f'|diff| mean {np.abs(d).mean():.3f}, corr {np.corrcoef(cli, api)[0, 1]:.2f}')
    for n, q in (('API', api), ('CLI', cli)):
        print(f'{n}: mean est {q.mean():.3f} (acc {acc.mean():.3f}), RMSE raw {np.sqrt(np.mean((q - acc) ** 2)):.3f}, shrunk .75 {np.sqrt(np.mean((.5 + .75 * (q - .5) - acc) ** 2)):.3f}')
