"""Unseen-system check: deployed prior prompt run by Opus 4.5 (training data to 2025-08) on subjects released 2025-09+,
named vs identity hidden; the Gemini 3.1 Pro calibration example is dropped and its pairs left out.
usage: ANTHROPIC_API_KEY=... (or VIA=cli) prior_eval_unseen.py [MODEL] [EFFORT]   ->  data/llm/prior_eval_unseen_<model>.json"""
import json, sys, os, re, importlib.util, urllib.request
from concurrent.futures import ThreadPoolExecutor
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'claude-opus-4-5-20251101'
EFF = sys.argv[2] if len(sys.argv) > 2 else 'low'
MP = 'submission/model_v26.py' if os.path.exists('submission/model_v26.py') else 'submission/model.py'   # the release ships the same predictor as model.py
spec = importlib.util.spec_from_file_location('m', MP); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
assert len(M.ANCHORS) == 8, 'build the 8 calibration examples first (src/make_anchors.py, see README)'
KEY = os.environ.get('ANTHROPIC_API_KEY', '')
DROP = 'Google Gemini 3.1 Pro'
ANCH = "\n\n".join("<example>\n" + "\n".join("<item>\n" + t + "\n</item>" for t in a["items"]) + "\nModel: " + a["model"]
                   + "\nTrue percentage correct: " + str(a["pct"]) + "\n</example>" for a in M.ANCHORS if a["model"] != DROP)
PRE = os.environ.get('GROUP') == 'pre'   # control: the same predictor on subjects released before its reliable knowledge cutoff, named arm only
ALL = json.load(open('data/llm/prior_eval_all.json'))
if PRE:
    idx = {(x['bench'], x['subject_id']): x for x in ALL}
    E = [idx[(p['bench'], p['subject_id'])] for p in json.load(open('data/llm/unseen_manifest/pairs_pre.json'))]
else:
    E = [x for x in ALL if (x['subject'].get('release_date') or '') >= '2025-09' and x['subject']['normalized_name'] != DROP]

CLI = os.environ.get('VIA') == 'cli'   # VIA=cli: same prompt through the Claude Code CLI on the subscription, no API key

DROP_ENV = ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX',
            'CLAUDE_CODE_USE_FOUNDRY', 'ANTHROPIC_BEDROCK_BASE_URL', 'ANTHROPIC_VERTEX_BASE_URL', 'AWS_BEARER_TOKEN_BEDROCK')
SUB_ENV = dict({k: v for k, v in os.environ.items() if k not in DROP_ENV}, MAX_THINKING_TOKENS='0', CLAUDE_CODE_MAX_OUTPUT_TOKENS='512',
               CLAUDE_CODE_EFFORT_LEVEL=EFF)   # subscription login only; no thinking, as in the API call

def ask_cli(system, user):
    import subprocess
    r = subprocess.run(['claude', '-p', '--safe-mode', '--no-session-persistence', '--model', MODEL, '--system-prompt', system, '--tools', '', '--effort', EFF,
                        '--output-format', 'json'], input=user, capture_output=True, text=True, timeout=300, cwd='/tmp',
                       env=SUB_ENV)
    d = json.loads(r.stdout)
    return d.get('result') or '', d.get('stop_reason'), dict(d.get('usage', {}), is_error=d.get('is_error'), exit=r.returncode,
                                                         models=list(d.get('modelUsage', {})), api_error=d.get('api_error_status'))

def call(o, redact):
    body = "\n\n".join("<item>\n" + t[:600] + "\n</item>" for t in o['texts'][:5])
    desc = M._subject_desc({k: v for k, v in o['subject'].items() if k != 'subject_features_extra'})   # both arms omit 'Other settings', so only name/provider/date differ
    if redact:
        rest = M._subject_desc({k: v for k, v in o['subject'].items() if k in ('harness', 'harness_version', 'reasoning_effort')})
        desc = 'Model: an unnamed AI system' + ('\n' + rest if rest else '')   # same labels as the named arm
    user = [{"type": "text", "text": ANCH + "\n\nNow the benchmark to estimate:\n\n", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": body + "\n\n" + desc + "\n\nReply with only one number from 0 to 100: the percentage of this benchmark's items this system answers correctly."}]
    if CLI:
        try:
            text, stop, usage = ask_cli(M.PRIOR_PROMPT, user[0]['text'] + user[1]['text'])
            nums = re.findall(r"\d+(?:\.\d+)?", text)
            ok = nums and stop == 'end_turn' and not usage.get('is_error') and usage.get('exit') == 0 and usage.get('models') == [MODEL]
            return (min(100., max(0., float(nums[-1]))) / 100 if ok else None), dict(usage, stop_reason=stop)
        except Exception as e:
            return None, {'err': str(e)[:200]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps({"model": MODEL, "max_tokens": 512,
        "output_config": {"effort": EFF}, "system": M.PRIOR_PROMPT, "messages": [{"role": "user", "content": user}]}).encode(),
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=180))
        nums = re.findall(r"\d+(?:\.\d+)?", "".join(b.get("text", "") for b in out.get("content", [])))
        ok = nums and out.get('stop_reason') == 'end_turn'   # cut off at max_tokens or refused: no estimate
        return (min(100., max(0., float(nums[-1]))) / 100 if ok else None), dict(out.get('usage', {}), stop_reason=out.get('stop_reason'))
    except Exception as e:
        return None, {'err': str(e)[:200]}

want = [(p['bench'], p['subject_id']) for p in json.load(open('data/llm/unseen_manifest/pairs_pre.json' if PRE else 'data/llm/unseen_manifest/pairs.json'))]
if [(o['bench'], o['subject_id']) for o in E] != want: sys.exit('pairs differ from data/llm/unseen_manifest/pairs.json')
f = f'data/llm/prior_eval_unseen_{MODEL}' + ('_cli' if CLI else '') + ('_pre' if PRE else '') + '.json'
if not os.path.exists(f):
    jobs = [(o, r) for r in ((False,) if PRE else (False, True)) for o in E]
    first = call(*jobs[0]); print('first call', first, flush=True)   # writes the prompt cache, and stops early on an API error
    if first[0] is None: sys.exit('first call failed: ' + json.dumps(first[1]))
    with ThreadPoolExecutor(4 if CLI else 8) as ex: rest = list(ex.map(lambda j: call(*j), jobs[1:]))
    out = [first] + rest
    json.dump([dict(bench=o['bench'], subject_id=o['subject_id'], name=o['subject']['normalized_name'], released=o['subject'].get('release_date'),
                    acc=o['acc'], redact=r, estimate=v, usage=u) for (o, r), (v, u) in zip(jobs, out)], open(f, 'w'))
print(len(E), 'pairs; written', f)
