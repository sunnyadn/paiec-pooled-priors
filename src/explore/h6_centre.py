"""H6 centre anchor: mean Opus rating (logit) of the rated simulation items, overall and with each benchmark left out (for the cached
simulation, so no benchmark's own items set its anchor). usage: h6_centre.py  ->  data/kdiag/h6_centre.json"""
import glob, json, numpy as np
lg = lambda p: np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98)))
rat = {}
for f in glob.glob('data/llm/batches_opus_fixed/*.json'):
    for x in json.load(open(f))['ratings']: rat[str(x['id'])] = float(x['p'])
ITB = {b: [lg(rat[str(x['item_id'])] / 100.0) for x in v if str(x['item_id']) in rat] for b, v in json.load(open('data/llm/items_to_rate.json')).items()}
out = {'_all': float(np.mean([z for v in ITB.values() for z in v]))}
out.update({'b_' + b: float(np.mean([z for bb, v in ITB.items() if bb != b for z in v])) for b in ITB})
json.dump(out, open('data/kdiag/h6_centre.json', 'w'), indent=1); print(out)
