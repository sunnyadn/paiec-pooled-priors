"""Generic task extraction for structured items: put the actual question first, drop shared system boilerplate and templates."""
import json

KEYS = ('question', 'user', 'prompt', 'input', 'instruction', 'task', 'query', 'problem', 'description', 'context', 'code')


def _strings(x, out):
    if isinstance(x, str):
        out.append(x.strip())
    elif isinstance(x, dict):
        for v in x.values():
            _strings(v, out)
    elif isinstance(x, list):
        for v in x:
            _strings(v, out)


def extract(content):
    """Return the item text with the task first and everything else (system text included) after it, so nothing is lost."""
    try:
        j = json.loads(content)
    except Exception:
        return content
    first, rest = [], []
    if isinstance(j, dict) and isinstance(j.get('messages'), list):
        msgs = [m for m in j['messages'] if isinstance(m, dict)]
        users = [i for i, m in enumerate(msgs) if m.get('role') == 'user']
        first += [str(msgs[users[-1]].get('content', '')).strip()] if users else []
        rest += [str(m.get('content', '')).strip() for i, m in enumerate(msgs) if not users or i != users[-1]]   # earlier turns and system kept
    if isinstance(j, dict):
        for k in KEYS:
            if k in j:
                _strings(j[k], first)   # strings, lists or dicts alike
        for k, v in j.items():
            if k not in KEYS and k != 'messages':
                _strings(v, rest)
    text = '\n'.join(x for x in first + rest if x)
    return text or content


def strip_shared(texts, frac=0.2):
    """Drop lines (and sentences within long lines) that recur in at least `frac` of a benchmark's items: shared instructions."""
    import collections, re
    split = lambda t: [u for l in t.split('\n') for u in re.split(r'(?<=[.!?])\s+', l) if u.strip()]
    units = [split(t) for t in texts]
    cnt = collections.Counter(u.strip() for us in units for u in set(us))
    common = {u for u, c in cnt.items() if len(texts) >= 5 and c >= frac * len(texts)}
    out = [' '.join(u for u in us if u.strip() not in common) or t for us, t in zip(units, texts)]
    return out, sorted(common)
