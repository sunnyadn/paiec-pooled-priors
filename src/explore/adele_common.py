"""Shared pieces of the ADeLe pilot: DeLeAn rubric sections cut by their headings, the batched annotation prompt, a strict CLI call."""
import json, os, re, subprocess
D = 'data/llm/adele'
DIMS = ['AS', 'CEc', 'CEe', 'CL', 'MCr', 'MCt', 'MCu', 'MS', 'QLl', 'QLq', 'SNs', 'KNa', 'KNc', 'KNf', 'KNn', 'KNs', 'AT', 'VO']   # 18 demands
TEXT = open(f'{D}/rubrics_v1.txt').read().replace('\f', '\n')   # the PDF's page breaks are not lines of the rubric
HEAD = {d: m.start() for d in DIMS + ['UG'] for m in [re.search(r'^[^\n]*\(' + d + r'\)[ \t]*$', TEXT, re.M)]}
assert all(v is not None for v in HEAD.values()), 'a rubric heading was not found'
STARTS = sorted(HEAD.values())

def rubric(dim):
    a = HEAD[dim]; b = next((s for s in STARTS if s > a), len(TEXT)); block = TEXT[a:b].strip()
    assert all(f'Level {l}' in block for l in range(5)) and 'Level 5' in block, f'{dim}: rubric block lacks some of levels 0-5+'
    return block

UG_NOTE = ("Answer format (replaces the paper's Unguessability rubric, which needs a reference answer): FMT = the number of answer options "
           "if the task is multiple-choice (explicit or implicit), or 0 if it is open-ended.")
BATCH_SYS = ("You annotate the demands of AI-evaluation task instances with the DeLeAn Rubric Set below. For every instance give each of the 18 "
             "demand dimensions an integer level from 0 to 5 (5 stands for 5+), following its rubric and anchors. Text may be truncated.\n\n" +
             "\n\n".join(rubric(d) for d in DIMS) + "\n\n" + UG_NOTE)
KEYS = DIMS + ['FMT']

def batch_user(texts):
    return "\n\n".join(f'<instance id="{n}">\n{t[:3000]}\n</instance>' for n, t in enumerate(texts)) + \
        "\n\nReply with only a JSON object mapping each instance id to an object with integer values for the keys " + ", ".join(KEYS) + "."

def ug(fmt):   # chance of error when guessing, in percent: 1 - 1/options; open-ended -> 100
    return 100.0 if fmt < 2 else 100.0 * (1 - 1 / fmt)

def valid(v):   # one instance's annotation -> 19 numbers (18 levels + UG) or None
    isint = lambda x: type(x) is int   # strict JSON integers: no floats, no booleans, no strings
    if not isinstance(v, dict) or not all(k in v and isint(v[k]) for k in KEYS): return None
    lv = [v[k] for k in DIMS]; fmt = v['FMT']
    return lv + [ug(fmt)] if all(0 <= x <= 5 for x in lv) and 0 <= fmt <= 100 else None

DROP = ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX',
        'CLAUDE_CODE_USE_FOUNDRY', 'ANTHROPIC_BEDROCK_BASE_URL', 'ANTHROPIC_VERTEX_BASE_URL', 'AWS_BEARER_TOKEN_BEDROCK')
ENV = dict({k: v for k, v in os.environ.items() if k not in DROP}, CLAUDE_CODE_MAX_OUTPUT_TOKENS='8192', CLAUDE_CODE_EFFORT_LEVEL='low', CLAUDE_CODE_MAX_RETRIES='0')
MODEL = 'claude-opus-5-5'

def cli(system, user, timeout=600):
    """-> dict(text, served, usage, error); error is None only for a clean answer from MODEL alone."""
    try:
        r = subprocess.run(['claude', '-p', '--safe-mode', '--no-session-persistence', '--model', MODEL, '--system-prompt', system, '--tools', '',
                            '--effort', 'low', '--output-format', 'json'], input=user, capture_output=True, text=True, timeout=timeout, cwd='/tmp', env=ENV)
        d = json.loads(r.stdout) if r.stdout.strip().startswith('{') else {}
        served = list(d.get('modelUsage', {})); text = d.get('result') or ''
        err = None if (r.returncode == 0 and d and not d.get('is_error') and served == [MODEL]) else (text[-200:] or (r.stderr or '')[-200:] or f'exit {r.returncode}')
        return dict(text=text, served=served, usage=d.get('usage'), error=err)
    except Exception as ex:
        return dict(text='', served=[], usage=None, error=f'{type(ex).__name__}: {ex}'[:200])

def load_rows():   # the pilot's rows for the model comparisons (as adele_rf.py): 900 frozen items, fence-free Opus ratings, level from the other 150 items
    import glob, numpy as np, pandas as pd
    ITEMS = json.load(open('data/llm/items_to_rate.json')); L = json.load(open(f'{D}/list.json')); ANNR = json.load(open(f'{D}/annotations.json'))
    keys = [(b, i) for b in sorted(L) for i in L[b]]; frozen = {'|'.join(f'{b}:{i}' for b, i in keys[j:j + 5]) for j in range(0, len(keys), 5)}
    assert set(ANNR) == frozen and len(frozen) == 180, 'annotation records differ from the 180 frozen tasks'
    ANN = {}
    for r in ANNR.values(): ANN.update({k: v for k, v in (r.get('levels') or {}).items() if v})
    assert len(keys) == 900 and all(f'{b}:{i}' in ANN for b, i in keys), 'some of the 900 frozen items lack a valid annotation'
    rat = {}
    for f in glob.glob('data/llm/batches_opus_fixed/*.json'):
        for x in json.load(open(f))['ratings']: rat[str(x['id'])] = float(x['p'])
    assert all(i in rat for b, i in keys), 'some of the 900 frozen items lack a fence-free Opus rating'
    R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'trial', 'y'])
    R = R.sort_values('trial').drop_duplicates(['bench', 'subject_id', 'item_id']); R['item_id'] = R.item_id.astype(str)
    lg = lambda p: np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98))); F = [f'f{j}' for j in range(19)]; rows = []
    for b, ids in L.items():
        other = {str(x['item_id']) for x in ITEMS[b]} - set(ids)
        lev = R[(R.bench == b) & R.item_id.isin(other)].groupby('subject_id').y.agg(['sum', 'count'])
        g = R[(R.bench == b) & R.item_id.isin(ids)].join(lev, on='subject_id'); g = g[g['count'] >= 20].copy()
        r_ = g.item_id.map(rat).clip(2.0, 98.0); g['opus'] = np.log(r_ / (100.0 - r_)); g['level'] = lg((g['sum'] + 1) / (g['count'] + 2))
        g[F] = np.array([ANN[f'{b}:{i}'] for i in g.item_id]); g['bench'] = b; rows.append(g)
    G = pd.concat(rows); assert sorted(G.bench.unique()) == sorted(L) and len(G) == 27691, f'rows {len(G)} (expected 27691)'
    return G, F
