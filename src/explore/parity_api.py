# parity: deployed per-item API rating (Opus, effort low) vs cached subscription batch rating
import json, glob, sys, random
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, 'submission'); import model; model._cfg()['model'] = 'claude-haiku-5-5'
from scipy.stats import spearmanr
items = json.load(open('data/llm/items_to_rate.json'))
old = {}
for f in glob.glob('data/llm/batches_haiku/*.json'):
    try:
        for r in json.load(open(f))['ratings']: old[r['id']] = r['p']
    except Exception: pass
random.seed(0); out = {}
for b in ['cruxeval', 'cmmlu', 'afrimedqa', 'harmbench']:
    pool = [it for it in items.get(b, []) if it['item_id'] in old]
    pick = random.sample(pool, min(40, len(pool)))
    with ThreadPoolExecutor(8) as ex: new = list(ex.map(lambda it: model._call(it['text']), pick))
    pairs = [(old[it['item_id']], n) for it, n in zip(pick, new) if n is not None]
    rho = spearmanr([a for a, _ in pairs], [c for _, c in pairs]).correlation
    print(b, 'n', len(pairs), 'of', len(pick), 'spearman', round(rho, 3))
