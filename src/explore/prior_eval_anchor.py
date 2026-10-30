"""Prior with in-context anchors: K solved examples from OTHER training benchmarks (model, 2 sample items, true accuracy).
usage: prior_eval_anchor.py MODEL K TAG"""
import json, re, sys, os, random, urllib.request, numpy as np
from concurrent.futures import ThreadPoolExecutor
MODEL, K, TAG = sys.argv[1], int(sys.argv[2]), sys.argv[3]
KEY = json.load(open('submission/llm_config.json'))['key']
sys.argv = [sys.argv[0], MODEL, '10', '1', TAG]
SYS = ("You will see sample items from an AI evaluation benchmark (text may be truncated; images are not shown).\n"
       "Estimate the percentage of this benchmark's items that the AI system described below answers or completes correctly "
       "in a single attempt. Use what you know about that model and how hard these items are. "
       "Solved examples from other benchmarks show the true percentages, for calibration. Reply with only the number.")
E = json.load(open('data/llm/prior_eval.json'))
def desc(s):
    lab = dict(normalized_name='Model', provider='Provider', release_date='Released', harness='Harness / scaffold',
               harness_version='Harness version', reasoning_effort='Reasoning effort', subject_features_extra='Other settings')
    return '\n'.join(f'{lab[k]}: {s[k]}' for k in lab if s.get(k))
def anchors(o):
    rng = random.Random(o['bench'] + o['subject']['normalized_name'])
    pool = [e for e in E if e['bench'] != o['bench']]
    out = []
    for e in rng.sample(pool, K):
        items = '\n'.join('<item>\n' + t[:300] + '\n</item>' for t in e['texts'][:2])
        out.append(f"<example>\n{items}\nModel: {e['subject']['normalized_name']}\nTrue percentage correct: {round(100*e['acc'])}\n</example>")
    return '\n\n'.join(out)
def call(o):
    body = '\n\n'.join('<item>\n' + t[:600] + '\n</item>' for t in o['texts'][:10])
    user = anchors(o) + '\n\nNow the benchmark to estimate:\n\n' + body + '\n\n' + desc(o['subject']) + "\n\nReply with only one number from 0 to 100: the percentage of this benchmark's items this system answers correctly."
    req = urllib.request.Request('https://api.anthropic.com/v1/messages', data=json.dumps({'model': MODEL, 'max_tokens': 2048,
        'output_config': {'effort': 'low'}, 'system': SYS, 'messages': [{'role': 'user', 'content': user}]}).encode(),
        headers={'x-api-key': KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=120))
        nums = re.findall(r'\d+(?:\.\d+)?', ''.join(b.get('text', '') for b in out.get('content', [])))
        return (min(100., max(0., float(nums[-1]))) / 100 if nums else None), out.get('usage', {})
    except Exception as e:
        return None, {'err': str(e)[:100]}
f = f'data/llm/prior_eval_{TAG}.json'
if not os.path.exists(f):
    with ThreadPoolExecutor(8) as ex: json.dump(list(ex.map(call, E)), open(f, 'w'))
