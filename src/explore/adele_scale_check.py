"""ADeLe scale check (subscription CLI): 10 pilot items (seed 0) on QLq, KNf, AT, MCr, one dimension per call with only that rubric (paper
style) vs the batched 18-demand prompt of adele_pilot.py. Gate: >= 80% of pairs within one level and |mean(batch - single)| <= 0.5 for
every dimension. usage: adele_scale_check.py run | analyse  ->  data/llm/adele/scale_check.json, prompts in data/llm/adele/prompts/"""
import hashlib, json, os, random, re, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); import adele_common as A
OUT = f'{A.D}/scale_check.json'; PD = f'{A.D}/prompts'; os.makedirs(PD, exist_ok=True); CHECK = ['QLq', 'KNf', 'AT', 'MCr']
L = json.load(open(f'{A.D}/list.json')); ITEMS = json.load(open('data/llm/items_to_rate.json'))
TX = {(b, str(x['item_id'])): x['text'] for b in L for x in ITEMS[b]}
keys = sorted(random.Random(0).sample([(b, i) for b in sorted(L) for i in L[b]], 10))
def keep(name, system, user):   # the prompt actually sent
    h = hashlib.sha256(system.encode()).hexdigest()[:8]; open(f'{PD}/system_{h}.txt', 'w').write(system); open(f'{PD}/{name}.user.txt', 'w').write(user); return h
if sys.argv[1] == 'run':
    RES = json.load(open(OUT)) if os.path.exists(OUT) else {'single': {}, 'batch': {}}
    for dim in CHECK:
        sysd = "You annotate the demands of AI-evaluation task instances. The following rubric describes six distinct levels of this demand:\n\n" + A.rubric(dim)
        for k in keys:
            tag = f'{dim}|{k[0]}:{k[1]}'
            if tag in RES['single']: continue   # never repeat a recorded call (failures included)
            user = (f'TASK INSTANCE:\n{TX[k][:3000]}\n\nScore the level of {dim} demanded by this task instance on a discrete value from 0 to 5. '
                    f'Reason step by step, then end with: "Thus, the level of {dim} demanded by the given TASK INSTANCE is: SCORE".')
            r = A.cli(sysd, user); m = re.search(r'is:\s*\**\s*([0-5])\s*\**\s*\.?\s*$', r['text'].strip())   # the closing statement, a whole integer
            lvl = int(m.group(1)) if (m and r['error'] is None) else None
            RES['single'][tag] = dict(level=lvl, served=r['served'], usage=r['usage'], error=r['error'] or (None if lvl is not None else 'no level'),
                                      system=keep(f'single_{dim}_{k[0]}_{k[1]}', sysd, user), truncated=len(TX[k]) > 3000)
            json.dump(RES, open(OUT, 'w')); print(tag, lvl, RES['single'][tag]['error'] or '', flush=True)
    for j in range(0, 10, 5):   # the exact prompt of adele_pilot.py annotate
        job = keys[j:j + 5]; tag = '|'.join(f'{b}:{i}' for b, i in job)
        if tag in RES['batch']: continue
        user = A.batch_user([TX[k] for k in job]); r = A.cli(A.BATCH_SYS, user); lv = None
        if r['error'] is None:
            m = re.search(r'\{.*\}', r['text'], re.S)
            try: J = json.loads(m.group(0)) if m else {}
            except Exception: J = {}
            lv = {f'{b}:{i}': A.valid(J.get(str(n), {})) for n, (b, i) in enumerate(job)}
        RES['batch'][tag] = dict(levels=lv, served=r['served'], usage=r['usage'], error=r['error'], system=keep(f'batch_{j}', A.BATCH_SYS, user))
        json.dump(RES, open(OUT, 'w')); print('batch', j, r['error'] or 'ok', flush=True)
else:
    RES = json.load(open(OUT)); B = {}
    for r in RES['batch'].values(): B.update({k: v for k, v in (r['levels'] or {}).items() if v})
    fails = [t for t, r in RES['single'].items() if r['level'] is None] + [t for t, r in RES['batch'].items() if not r['levels']]
    print('single calls', len(RES['single']), 'batch calls', len(RES['batch']), '| failed', len(fails), fails[:4])
    rows = [(t.split('|')[0], r['level'], B[t.split('|', 1)[1]][A.DIMS.index(t.split('|')[0])])
            for t, r in RES['single'].items() if r['level'] is not None and t.split('|', 1)[1] in B]
    passed = len(rows) == 40
    for dim in CHECK:
        x = np.array([(s, b) for d, s, b in rows if d == dim]).reshape(-1, 2)
        if not len(x): passed = False; print(f'{dim}: no comparable pairs'); continue
        w = np.mean(np.abs(x[:, 0] - x[:, 1]) <= 1); bias = np.mean(x[:, 1] - x[:, 0])
        passed &= abs(bias) <= 0.5; print(f'{dim}: n {len(x)} within 1 level {w:.0%} mean (batch - single) {bias:+.2f}')
    x = np.array([(s, b) for _, s, b in rows]).reshape(-1, 2); w = np.mean(np.abs(x[:, 0] - x[:, 1]) <= 1) if len(x) else 0.0; passed &= w >= 0.8
    print(f'ALL: compared {len(rows)}/40, within 1 level {w:.0%} | GATE {"PASS" if passed else "FAIL"}')
