"""Rebuild the Opus rating cache without code fences: 17 batch files in data/llm/batches_opus hold their JSON inside ```json fences, which
every loader skipped silently. Writes data/llm/batches_opus_fixed (one plain JSON per readable batch) and lists the unreadable ones."""
import glob, json, os, re
SRC, DST = 'data/llm/batches_opus', 'data/llm/batches_opus_fixed'; os.makedirs(DST, exist_ok=True); bad = []
for f in sorted(glob.glob(f'{SRC}/*.json')):
    t = open(f).read(); m = re.fullmatch(r'\s*```(?:json)?\s*(\{.*\})\s*```\s*', t, re.S)
    try: d = json.loads(m.group(1) if m else t); assert isinstance(d.get('ratings'), list)
    except Exception: bad.append(os.path.basename(f)); continue
    json.dump(d, open(f'{DST}/{os.path.basename(f)}', 'w'))
n = sum(len(json.load(open(f))['ratings']) for f in glob.glob(f'{DST}/*.json'))
print('batches written', len(glob.glob(f'{DST}/*.json')), '| ratings', n, '| unreadable', bad)
