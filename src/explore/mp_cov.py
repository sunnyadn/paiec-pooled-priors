import multiprocessing as mp, os, sys, random, importlib.util, tempfile
PKG = sys.argv[1]; W = int(sys.argv[2]); SHARE = sys.argv[3] == '1'; NP = int(sys.argv[4])
def worker(args):
    wid, budget, items, labeled = args
    spec = importlib.util.spec_from_file_location('m%d_%d' % (wid, budget), PKG + '/model.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    pc = []; M._call_prior = lambda desc, texts: (pc.append(1) or 0.7); M._call = lambda text: 50.0
    if not SHARE:
        d = tempfile.mkdtemp()
        if hasattr(M, '_PDIRS'): M._PDIRS = [d]
        else: M._prior_dir = lambda: d
        M._disk_path = lambda: os.path.join(d, 'r.jsonl')
    orig = M._subject_prior; used = [0, 0]
    def wrap(*a, **k):
        r = orig(*a, **k); used[0] += r is not None; used[1] += 1; return r
    M._subject_prior = wrap
    for inp in items: M.predict(inp, labeled)
    return len(pc), used[0], used[1]
if __name__ == '__main__':
    pairs = [({'normalized_name': f'Model {i}', 'provider': 'X'}, f'b{i}') for i in range(NP)]
    out = []
    for budget in [0, 1, 3, 7, 15, 31]:
        labeled = [[[s, {'item_content': f'{b} item {k}', 'item_features': '', 'interactors': '', 'benchmark_id': b}], random.randint(0, 1)] for s, b in pairs for k in range(budget)]
        inputs = [[s, {'item_content': f'{b} eval {k}', 'item_features': '', 'interactors': '', 'benchmark_id': b}] for s, b in pairs for k in range(30)]
        random.shuffle(inputs)
        with mp.Pool(W) as pool: res = pool.map(worker, [(w, budget, inputs[w::W], labeled) for w in range(W)])
        c = sum(r[0] for r in res); u = sum(r[1] for r in res); n = sum(r[2] for r in res)
        out.append(f'b{budget}: calls {c}, prior used {u}/{n}')
    print(os.path.basename(PKG), f'W={W} share={SHARE}', ' | '.join(out))
