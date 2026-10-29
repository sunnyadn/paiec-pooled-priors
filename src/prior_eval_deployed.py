"""Score the deployed prior prompt (v18: fixed anchors, subject attributes) on data/llm/prior_eval.json.
usage: prior_eval_deployed.py MODEL NITEMS EFFORT TAG ; anchor pairs are excluded from scoring."""
import json, sys, os, importlib.util, urllib.request, re
from concurrent.futures import ThreadPoolExecutor
MODEL, NIT, EFF, TAG = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
spec = importlib.util.spec_from_file_location('m', 'submission/model_v26.py' if os.path.exists('submission/model_v26.py') else 'submission/model.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
KEY = open(os.path.expanduser('~/.config/paiec/key2')).read().strip()
TRUNC = int(os.environ.get('TRUNC', '600'))
E = json.load(open(os.environ.get('EVALSET','data/llm/prior_eval.json')))
anch = {(a['model'], a['pct']) for a in M.ANCHORS}
def call(o):
    if (o['subject']['normalized_name'], round(100 * o['acc'])) in anch: return None, {'skip': 1}
    body = "\n\n".join("<item>\n" + t[:TRUNC] + "\n</item>" for t in o['texts'][:NIT])
    user = M._ANCHOR_TXT + "\n\nNow the benchmark to estimate:\n\n" + body + "\n\n" + M._subject_desc(o['subject']) + "\n\nReply with only one number from 0 to 100: the percentage of this benchmark's items this system answers correctly."
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps({"model": MODEL, "max_tokens": 4096,
        "output_config": {"effort": EFF}, "system": M.PRIOR_PROMPT, "messages": [{"role": "user", "content": user}]}).encode(),
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=180))
        nums = re.findall(r"\d+(?:\.\d+)?", "".join(b.get("text", "") for b in out.get("content", [])))
        return (min(100., max(0., float(nums[-1]))) / 100 if nums else None), out.get('usage', {})
    except Exception as e:
        return None, {'err': str(e)[:80]}
f = f'data/llm/prior_eval_{TAG}.json'
if not os.path.exists(f):
    with ThreadPoolExecutor(8) as ex: json.dump(list(ex.map(call, E)), open(f, 'w'))
