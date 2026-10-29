"""Score prompts/models for the subject-accuracy prior on data/llm/prior_eval.json.
usage: prior_eval.py MODEL NITEMS ATTRS(0|1) TAG  -> data/llm/prior_eval_TAG.json + summary"""
import json, re, sys, os, urllib.request, numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import binom
MODEL, NIT, ATTRS, TAG = sys.argv[1], int(sys.argv[2]), sys.argv[3] == '1', sys.argv[4]
KEY = json.load(open('submission/llm_config.json'))['key']
SYS = ("You will see sample items from an AI evaluation benchmark (text may be truncated; images are not shown).\n"
       "Estimate the percentage of this benchmark's items that the AI system described below answers or completes correctly "
       "in a single attempt. Use what you know about that model and how hard these items are. Reply with only the number.")
def desc(s):
    if not ATTRS: return 'Model: ' + s['normalized_name']
    lab = dict(normalized_name='Model', provider='Provider', release_date='Released', harness='Harness / scaffold',
               harness_version='Harness version', reasoning_effort='Reasoning effort', subject_features_extra='Other settings')
    return '\n'.join(f'{lab[k]}: {s[k]}' for k in lab if s.get(k))
def call(o):
    body = '\n\n'.join('<item>\n' + t[:600] + '\n</item>' for t in o['texts'][:NIT])
    user = body + '\n\n' + desc(o['subject']) + "\n\nReply with only one number from 0 to 100: the percentage of this benchmark's items this system answers correctly."
    req = urllib.request.Request('https://api.anthropic.com/v1/messages', data=json.dumps({'model': MODEL, 'max_tokens': 2048,
        'output_config': {'effort': 'low'}, 'system': SYS, 'messages': [{'role': 'user', 'content': user}]}).encode(),
        headers={'x-api-key': KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=120))
        nums = re.findall(r'\d+(?:\.\d+)?', ''.join(b.get('text', '') for b in out.get('content', [])))
        return (min(100., max(0., float(nums[-1]))) / 100 if nums else None), out.get('usage', {})
    except Exception as e:
        return None, {'err': str(e)[:100]}
E = json.load(open('data/llm/prior_eval.json'))
f = f'data/llm/prior_eval_{TAG}.json'
if os.path.exists(f): res = json.load(open(f))
else:
    with ThreadPoolExecutor(8) as ex: res = list(ex.map(call, E))
    json.dump(res, open(f, 'w'))
B = [0, 1, 3, 7, 15, 31]; W = [.1, .2, .2, .2, .2, .1]
def alc(m, a, K):
    t = 0
    for n, w in zip(B, W):
        ss = np.arange(n + 1); pr = binom.pmf(ss, n, a)
        t += w * sum(q * (a * (1 - a) + ((s + K * m) / (n + K) - a) ** 2) for s, q in zip(ss, pr))
    return t
ok = [(o['acc'], r[0]) for o, r in zip(E, res) if r[0] is not None]
a = np.array([x for x, _ in ok]); p = np.array([y for _, y in ok])
tin = sum(r[1].get('input_tokens', 0) for r in res); tout = sum(r[1].get('output_tokens', 0) for r in res)
print(f'{TAG}: {len(ok)}/{len(E)} ok  tokens in {tin} out {tout}  RMSE raw {np.sqrt(np.mean((p-a)**2)):.3f}  corr {np.corrcoef(p,a)[0,1]:.2f}  mean p {p.mean():.2f} vs acc {a.mean():.2f}')
for sh in [0.5, 0.75, 1.0]:
    m = 0.5 + sh * (p - 0.5)
    print(f'   shrink {sh}: RMSE {np.sqrt(np.mean((m-a)**2)):.3f}  ALC K2 {np.mean([alc(x,y,2) for x,y in zip(m,a)]):.4f}  K4 {np.mean([alc(x,y,4) for x,y in zip(m,a)]):.4f}')
print(f'   reference: laplace ALC {np.mean([alc(0.5,y,2) for y in a]):.4f}  oracle-prior K4 {np.mean([alc(y,y,4) for y in a]):.4f}')
