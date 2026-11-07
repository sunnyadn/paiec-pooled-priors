"""New-benchmark check, step 2 (runs on the Mac, subscription): the deployed prior prompt (8 examples, 5 sample items, Opus 5.5
then Sonnet 5.5 then Haiku 5.5) through the Claude Code CLI, pairs in the frozen order, until done, quota runs out or the cutoff.
usage: prior_eval_extra.py  ->  data/llm/extra/results.json (+ .requests)"""
import json, os, re, subprocess, sys, threading, time, importlib.util
from concurrent.futures import ThreadPoolExecutor
OUT = 'data/llm/extra/results.json'
CUTOFF = os.environ.get('CUTOFF_UTC', '2026-10-31T09:00')   # no new pair starts after this (report cutoff 09:30Z)
if any(os.path.exists(OUT + x) for x in ('', '.partial', '.requests')): sys.exit('results already exist: refusing to run again')
MP = 'submission/model_v26.py' if os.path.exists('submission/model_v26.py') else 'submission/model.py'   # the release ships the same predictor as model.py
spec = importlib.util.spec_from_file_location('m', MP); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
assert len(M.ANCHORS) == 8, 'build the 8 calibration examples first (src/make_anchors.py, see README)'
E = {(e['bench'], e['subject_id']): e for e in json.load(open('data/llm/extra/eligible.json'))}
ORDER = json.load(open('data/llm/extra/run_order.json'))
DROP = ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX',
        'CLAUDE_CODE_USE_FOUNDRY', 'ANTHROPIC_BEDROCK_BASE_URL', 'ANTHROPIC_VERTEX_BASE_URL', 'AWS_BEARER_TOKEN_BEDROCK', 'MAX_THINKING_TOKENS')
ENV = dict({k: v for k, v in os.environ.items() if k not in DROP}, CLAUDE_CODE_MAX_OUTPUT_TOKENS='2048', CLAUDE_CODE_EFFORT_LEVEL='low',
           CLAUDE_CODE_MAX_RETRIES='0')
LOCK, LOG, RES = threading.Lock(), [], {}
STOP = [None]

def save():
    with LOCK:
        json.dump(LOG, open(OUT + '.requests', 'w')); json.dump(RES, open(OUT + '.partial', 'w'))

def ask(e):
    body = "\n\n".join("<item>\n" + t[:600] + "\n</item>" for t in e['texts'][:5])
    user = M._ANCHOR_TXT + "\n\nNow the benchmark to estimate:\n\n" + body + "\n\n" + M._subject_desc(e['subject']) + \
        "\n\nReply with only one number from 0 to 100: the percentage of this benchmark's items this system answers correctly."
    status = 'unparsed'
    for m in M.PRIOR_MODELS:
        rec = dict(pair=f"{e['bench']}|{e['subject_id']}", model=m, start=time.time())
        try:
            r = subprocess.run(['claude', '-p', '--safe-mode', '--no-session-persistence', '--model', m, '--system-prompt', M.PRIOR_PROMPT,
                                '--tools', '', '--effort', 'low', '--output-format', 'json'], input=user, capture_output=True, text=True,
                               timeout=300, cwd='/tmp', env=ENV)
            d = json.loads(r.stdout) if r.stdout.strip().startswith('{') else {}
            rec.update(exit=r.returncode, stop_reason=d.get('stop_reason'), served=list(d.get('modelUsage', {})), is_error=d.get('is_error'),
                       usage=d.get('usage'), api_error=d.get('api_error_status'), err=(r.stderr or '')[-200:] if not d else None)
            text = d.get('result') or ''
            msg = text + ' ' + (r.stdout if not d else '') + ' ' + (r.stderr or '')
            if (d.get('is_error') or not d or r.returncode != 0) and re.search(r'usage limit|quota', msg, re.I) and not re.search(r'rate', msg, re.I):
                STOP[0] = 'subscription limit'; status = 'failed'; return None, status
            nums = re.findall(r"\d+(?:\.\d+)?", text)
            if not d or r.returncode != 0 or d.get('is_error'): status = 'failed'; continue
            if nums and d.get('stop_reason') == 'end_turn' and rec['served'] == [m]:
                return min(100.0, max(0.0, float(nums[-1]))) / 100.0, 'ok:' + m   # a fallback model's answer counts, as when deployed
        except Exception as x:
            rec.update(err=str(x)[:200]); status = 'failed'
        finally:
            with LOCK: LOG.append(rec)
            save()
    return None, status

def job(k):
    if STOP[0] or time.strftime('%Y-%m-%dT%H:%M', time.gmtime()) >= CUTOFF:
        STOP[0] = STOP[0] or 'cutoff'; RES[k] = dict(status='untried'); return
    e = E[tuple(k.split('|', 1))]; p, s = ask(e)
    with LOCK: RES[k] = dict(status=s if p is None else 'ok', answered_by=s[3:] if p is not None else None, estimate=p)
    save(); print(len(RES), k, s, p, flush=True)

keys = [f"{o['bench']}|{o['subject_id']}" for o in ORDER]
with ThreadPoolExecutor(4) as ex: list(ex.map(job, keys))
for k in keys: RES.setdefault(k, dict(status='untried'))
json.dump(dict(stop=STOP[0], results=RES), open(OUT, 'w'))
print('done | stop', STOP[0], '|', {s: sum(r['status'] == s for r in RES.values()) for s in ('ok', 'unparsed', 'failed', 'untried')},
      '| answered by', {m: sum(r.get('answered_by') == m for r in RES.values()) for m in M.PRIOR_MODELS})
