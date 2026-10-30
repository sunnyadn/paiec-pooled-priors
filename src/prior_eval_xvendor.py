"""Score the deployed subject-prior prompt with another vendor's model on data/llm/prior_eval.json (anchor pairs skipped).
usage: prior_eval_xvendor.py openai|gemini MODEL TAG  -> data/llm/prior_eval_TAG.json, same format as prior_eval_deployed.py"""
import json, sys, os, re, importlib.util, urllib.request
from concurrent.futures import ThreadPoolExecutor
VENDOR, MODEL, TAG = sys.argv[1:4]
spec = importlib.util.spec_from_file_location('m', 'submission/model_v26.py' if os.path.exists('submission/model_v26.py') else 'submission/model.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
KEY = open(os.path.expanduser('~/.config/paiec/openai_key' if VENDOR == 'openai' else os.environ.get('GEMINI_KEYFILE', '~/.config/gemini/key-personal'))).read().strip()
E = json.load(open('data/llm/prior_eval.json'))
anch = {(a['model'], a['pct']) for a in M.ANCHORS}


def ask(user):
    if VENDOR == 'openai':
        req = urllib.request.Request('https://api.openai.com/v1/responses', data=json.dumps({
            'model': MODEL, 'instructions': M.PRIOR_PROMPT, 'input': user, 'reasoning': {'effort': 'low'}}).encode(),
            headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
        out = json.load(urllib.request.urlopen(req, timeout=300))
        text = ''.join(c.get('text', '') for o in out.get('output', []) if o.get('type') == 'message' for c in o.get('content', []))
        return text, out.get('usage', {})
    req = urllib.request.Request(f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}', data=json.dumps({
        'systemInstruction': {'parts': [{'text': M.PRIOR_PROMPT}]}, 'contents': [{'role': 'user', 'parts': [{'text': user}]}]}).encode(),
        headers={'Content-Type': 'application/json'})
    out = json.load(urllib.request.urlopen(req, timeout=300))
    text = ''.join(p.get('text', '') for c in out.get('candidates', []) for p in c.get('content', {}).get('parts', []))
    return text, out.get('usageMetadata', {})


def call(o):
    if (o['subject']['normalized_name'], round(100 * o['acc'])) in anch:
        return None, {'skip': 1}
    body = "\n\n".join("<item>\n" + t[:600] + "\n</item>" for t in o['texts'][:5])
    user = (M._ANCHOR_TXT + "\n\nNow the benchmark to estimate:\n\n" + body + "\n\n" + M._subject_desc(o['subject'])
            + "\n\nReply with only one number from 0 to 100: the percentage of this benchmark's items this system answers correctly.")
    try:
        text, usage = ask(user)
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        return (min(100., max(0., float(nums[-1]))) / 100 if nums else None), usage
    except Exception as e:
        return None, {'err': str(e)[:120]}


f = f'data/llm/prior_eval_{TAG}.json'
if not os.path.exists(f):
    with ThreadPoolExecutor(int(os.environ.get("THREADS", "6"))) as ex:
        json.dump(list(ex.map(call, E)), open(f, 'w'))
R = json.load(open(f))
if os.environ.get('RETRY') == '1':   # free-tier rate limits: redo failed calls one at a time
    import time
    for i, r in enumerate(R):
        if 'err' in r[1]:
            R[i] = list(call(E[i])); time.sleep(float(os.environ.get('SLEEP', '5')))
    json.dump(R, open(f, 'w'))
print(TAG, 'answered', sum(1 for r in R if r[0] is not None), 'errors', sum(1 for r in R if 'err' in r[1]), [r[1]['err'] for r in R if 'err' in r[1]][:2])
