"""Subject-aware LLM ratings ('item type x model' test): P(model NAME answers item correctly), Haiku via the submission key.
Writes data/llm/subj_opus/<bench>.json with rows [item_id, subject_id, name, p, y]."""
import json, os, re, sys, glob, urllib.request, random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
MODEL = "claude-opus-5-5"
KEY = open(__import__("os").path.expanduser("~/.config/paiec/key2")).read().strip()
SYS = ("Below is one test item from an AI evaluation benchmark (text may be truncated; images are not shown).\n"
       "Estimate the probability (0-100) that the AI model \"{name}\" answers or completes it correctly in a single attempt. "
       "Use what you know about this model's strengths and weaknesses on this kind of item. Judge only from the item itself.\n"
       "Reply with only the number.")
TAIL = "Do not answer the item. Reply with only one number from 0 to 100: the probability that {name} answers it correctly."
def call(name, text):
    user = "<item>\n" + text[:1500] + "\n</item>\n\n" + TAIL.format(name=name)
    req = urllib.request.Request("https://api.anthropic.com/v1/messages",
        data=json.dumps({"model": MODEL, "max_tokens": 1024, "output_config": {"effort": "low"},
                         "system": SYS.format(name=name), "messages": [{"role": "user", "content": user}]}).encode(),
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for att in range(5):
      try:
        out = json.load(urllib.request.urlopen(req, timeout=90))
        break
      except urllib.error.HTTPError as e:
        if e.code != 429 or att == 4: return None
        time.sleep(4 * (att + 1))
      except Exception:
        return None
    try:
        nums = re.findall(r"\d+(?:\.\d+)?", "".join(b.get("text", "") for b in out.get("content", [])))
        return min(100.0, max(0.0, float(nums[-1]))) if nums else None
    except Exception:
        return None
items = json.load(open('data/llm/items_to_rate.json'))
rated = set()
for f in glob.glob('data/llm/api_haiku/*.json'):
    rated |= {str(x['id']) for x in json.load(open(f))['ratings']}
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'normalized_name', 'y'])
NB, NS, NI = int(os.environ.get('NB', 6)), int(os.environ.get('NS', 5)), int(os.environ.get('NI', 50))
done = 0
for b in items:
    if done >= NB: break
    out_f = f'data/llm/subj_opus/{b}.json'
    tx = {str(x['item_id']): x['text'] for x in items[b]}
    d = R[(R.bench == b) & R.item_id.astype(str).isin(set(tx) & rated)].drop_duplicates(['subject_id', 'item_id'])
    d = d[d.normalized_name.notna() & (d.normalized_name.astype(str).str.len() > 0)]
    cnt = d.groupby('subject_id').item_id.nunique(); subs = cnt[cnt >= NI].index
    names = d[d.subject_id.isin(subs)].drop_duplicates('subject_id').set_index('subject_id').normalized_name
    names = names[~names.duplicated()]
    if len(names) < NS: continue
    acc = d[d.subject_id.isin(names.index)].groupby('subject_id').y.mean().sort_values()
    pick = [acc.index[int(round(i * (len(acc) - 1) / (NS - 1)))] for i in range(NS)]   # spread over ability
    common = set.intersection(*[set(d[d.subject_id == s].item_id.astype(str)) for s in pick])
    if len(common) < NI: continue
    its = sorted(random.Random(0).sample(sorted(common), NI))
    done += 1
    if os.path.exists(out_f): continue
    jobs = [(s, names[s], i) for s in pick for i in its]
    with ThreadPoolExecutor(4) as ex: ps = list(ex.map(lambda j: call(j[1], tx[j[2]]), jobs))
    yv = d.set_index(['subject_id', d.item_id.astype(str)]).y
    rows = [[i, s, n, p, int(yv[(s, i)])] for (s, n, i), p in zip(jobs, ps)]
    json.dump(rows, open(out_f, 'w'))
    print(b, list(names[pick]), sum(p is not None for p in ps), '/', len(ps), flush=True)
