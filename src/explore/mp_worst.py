import multiprocessing as mp, os, sys, random, importlib.util, tempfile
PKG = sys.argv[1]; W = int(sys.argv[2]); SHARE = sys.argv[3] == '1'; NB = int(sys.argv[5]) if len(sys.argv) > 5 else 7
def worker(args):
    wid, budget, items, labeled = args
    spec = importlib.util.spec_from_file_location('m%d_%d' % (wid, budget), PKG + '/model.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    pc = []; rc = []
    M._call_prior = lambda desc, texts: (pc.append(1) or 0.7)
    M._call = lambda text: (rc.append(1) or 50.0)
    if not SHARE:
        d = tempfile.mkdtemp(); M._PDIRS = [d]; M._disk_path = lambda: os.path.join(d, 'r.jsonl')
    for inp in items: M.predict(inp, labeled)
    return len(pc), len(rc)
if __name__ == '__main__':
    pairs = [({'normalized_name': f'Model {i}', 'provider': 'X'}, f'b{i%NB}') for i in range(int(sys.argv[4]))]
    P = R = 0
    for budget in [0, 1, 3, 7, 15, 31]:
        labeled = [[[s, {'item_content': f'{b} item {k} {s["normalized_name"]}', 'item_features': '', 'interactors': '', 'benchmark_id': b}], random.randint(0, 1)] for s, b in pairs for k in range(budget)]
        inputs = [[s, {'item_content': f'{b} eval {k} {s["normalized_name"]}', 'item_features': '', 'interactors': '', 'benchmark_id': b}] for s, b in pairs for k in range(30)]
        random.shuffle(inputs)
        with mp.Pool(W) as pool: res = pool.map(worker, [(w, budget, inputs[w::W], labeled) for w in range(W)])
        P += sum(r[0] for r in res); R += sum(r[1] for r in res); print(budget, sum(r[0] for r in res), sum(r[1] for r in res))
    print(f'W={W} share={SHARE} pairs={len(pairs)}: prior calls {P}, rating calls {R}')
