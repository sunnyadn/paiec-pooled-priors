"""Prior with anchors chosen by text similarity to the target benchmark's sample items (pool = other training benchmarks)."""
import json, sys, os, importlib.util, urllib.request, re
from concurrent.futures import ThreadPoolExecutor
TAG = sys.argv[1]
spec = importlib.util.spec_from_file_location('m', 'submission/model_v26.py' if os.path.exists('submission/model_v26.py') else 'submission/model.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
KEY = open(os.path.expanduser('~/.config/paiec/key2')).read().strip()
E = json.load(open('data/llm/prior_eval.json'))
bad = {'alignment_faking', 'emoji_attack', 'harmbench'}
def vec(texts): return M._vec(' '.join(t.lower() for t in texts))
V = [vec(e['texts'][:5]) for e in E]
def anchors(i):
    o = E[i]; cand = [j for j, e in enumerate(E) if e['bench'] != o['bench'] and e['bench'] not in bad]
    cand.sort(key=lambda j: -M._dot(V[i], V[j]))
    pick, seenb = [], set()
    for j in cand:
        if E[j]['bench'] in seenb: continue
        pick.append(j); seenb.add(E[j]['bench'])
        if len(pick) == 8: break
    return "\n\n".join("<example>\n" + "\n".join("<item>\n" + t[:300] + "\n</item>" for t in E[j]['texts'][:2]) + "\nModel: " + E[j]['subject']['normalized_name'] + "\nTrue percentage correct: " + str(round(100 * E[j]['acc'])) + "\n</example>" for j in pick)
def call(i):
    o = E[i]
    body = "\n\n".join("<item>\n" + t[:600] + "\n</item>" for t in o['texts'][:5])
    user = anchors(i) + "\n\nNow the benchmark to estimate:\n\n" + body + "\n\n" + M._subject_desc(o['subject']) + "\n\nReply with only one number from 0 to 100: the percentage of this benchmark's items this system answers correctly."
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps({"model": "claude-opus-5-5", "max_tokens": 4096,
        "output_config": {"effort": "low"}, "system": M.PRIOR_PROMPT, "messages": [{"role": "user", "content": user}]}).encode(),
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for att in range(4):
        try:
            out = json.load(urllib.request.urlopen(req, timeout=180))
            nums = re.findall(r"\d+(?:\.\d+)?", "".join(b.get("text", "") for b in out.get("content", [])))
            return (min(100., float(nums[-1])) / 100 if nums else None), out.get('usage', {})
        except urllib.error.HTTPError as e:
            if e.code != 429: return None, {'err': e.code}
            import time; time.sleep(5 * (att + 1))
        except Exception as e: return None, {'err': str(e)[:60]}
    return None, {}
f = f'data/llm/prior_eval_{TAG}.json'
with ThreadPoolExecutor(4) as ex: json.dump(list(ex.map(call, range(len(E)))), open(f, 'w'))
