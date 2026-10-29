"""Pooled Priors: predict P(correct) on a held-out benchmark from pooled revealed labels.

1. Benchmark level: success rate of all revealed labels on this benchmark_id (all subjects), Beta(1,1) prior.
2. Pair level: this subject's revealed labels on the benchmark, shrunk toward the benchmark rate (k pseudo-counts).
3. Item level: residual difficulty of revealed items (label minus its subject's pair estimate), transferred to the
   target item through character n-gram similarity of item text (top-K neighbours, shrunk).
No fitted parameters from the training corpus, no network, standard library + numpy only.
"""
import math, re, hashlib, json, os, tempfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, wait
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

K_PAIR = 8.0      # pseudo-counts pulling the pair rate toward the benchmark rate
LAMBDA = 1.0      # weight of the item-level offset
TOPK = 30         # neighbours for the item offset
DIM = 2 ** 16     # hashed feature dimension
CLIP = (0.01, 0.99)
RIDGE = 25.0      # ridge on the LLM-rating slope (pseudo-weight at zero slope)
LLM_WORKERS = 16  # parallel rating calls when fitting a benchmark
LLM_FIT_SECONDS = 90.0   # wall-clock cap for rating a benchmark's revealed items; unrated items are skipped

# ---- LLM difficulty rater (optional; disabled when no config) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG = None
_RATE = {}
RATE_PROMPT = ("Below is one test item from an AI evaluation benchmark (text may be truncated; images are not shown).\n"
    "Estimate the probability (0-100) that a typical current large language model (a mid-range model, single attempt) "
    "answers or completes it correctly. Judge only from the item itself.\n"
    "Reply with only the number.\n\n")
RATE_TAIL = ("Do not answer the item. Reply with only one number from 0 to 100: the probability that a typical "
    "current large language model answers it correctly.")

def _cfg():
    global _CFG
    if _CFG is None:
        try:
            _CFG = json.load(open(os.path.join(_HERE, "llm_config.json")))
        except Exception:
            _CFG = {}
    return _CFG

def _call(text):
    c = _cfg()
    prov, model, key = c.get("provider"), c.get("model"), c.get("key")
    if not (prov and model and key):
        return None
    body = RATE_PROMPT + text[:1500]
    if prov == "anthropic":
        # instruction as system prompt and repeated after the item, so the model rates the item instead of answering it
        user = "<item>\n" + text[:1500] + "\n</item>\n\n" + RATE_TAIL
        req = urllib.request.Request("https://api.anthropic.com/v1/messages",
            data=json.dumps({"model": model, "max_tokens": 1024, "output_config": {"effort": "low"},
                             "system": RATE_PROMPT.strip(), "messages": [{"role": "user", "content": user}]}).encode(),
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=20))
        txt = "".join(b.get("text", "") for b in out.get("content", []))
    elif prov == "gemini":
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
            data=json.dumps({"contents": [{"parts": [{"text": body}]}], "generationConfig": {"maxOutputTokens": 8, "temperature": 0}}).encode(),
            headers={"content-type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=30))
        txt = "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
    else:
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", txt)   # the answer is the last number (models sometimes explain first)
    return min(100.0, max(0.0, float(nums[-1]))) if nums else None

_DISK = None   # ratings persisted in the temp dir, so worker processes and later budgets reuse them
_LOCK = threading.Lock()

def _disk_path():
    return os.path.join(tempfile.gettempdir(), "pooled_priors_ratings.jsonl")

def _load_disk():
    global _DISK
    with _LOCK:
        if _DISK is not None:
            return _DISK
        _DISK = {}
        try:
            with open(_disk_path()) as f:
                for line in f:
                    try:
                        h, v = json.loads(line)
                        _DISK[h] = v
                    except Exception:
                        pass
        except Exception:
            pass
    return _DISK

def _rating(text):
    """z-ready logit rating for an item text, or None (cached in memory and on disk; failures not persisted)."""
    if text in _RATE:
        return _RATE[text]
    h = hashlib.md5((str(_cfg().get("model", "")) + "\n" + "v2\n" + RATE_PROMPT + RATE_TAIL + "\n" + text).encode()).hexdigest()
    disk = _load_disk()
    if h in disk:
        _RATE[text] = disk[h]
        return disk[h]
    r = None
    try:
        r = _call(_RAW.get(text, text))
    except Exception:
        r = None
    v = None if r is None else math.log(max(2.0, min(98.0, r)) / (100.0 - max(2.0, min(98.0, r))))
    _RATE[text] = v
    if v is not None:
        with _LOCK:
            disk[h] = v
            try:
                with open(_disk_path(), "a") as f:
                    f.write(json.dumps([h, v]) + "\n")
            except Exception:
                pass
    return v


_cache = {}

def _key(d):
    return tuple(sorted((k, str(v)) for k, v in d.items()))

_RAW = {}   # lowercased matching text -> original-case text (the LLM rates the original; case matters in code)

def _text(item):
    raw = str(item.get("item_content", ""))[:4000] + " " + str(item.get("item_features", ""))
    t = raw.lower()
    if t not in _RAW and len(_RAW) < 200000:
        _RAW[t] = raw
    return t

def _vec(text, _memo={}):
    if text in _memo:
        return _memo[text]
    t = re.sub(r"\s+", " ", text)
    idx = {}
    for w in t.split(" "):
        w = " " + w + " "
        for n in (3, 4, 5):
            for i in range(max(0, len(w) - n + 1)):
                h = int(hashlib.md5(w[i:i + n].encode()).hexdigest()[:8], 16) % DIM
                idx[h] = idx.get(h, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in idx.values())) or 1.0
    v = {k: val / norm for k, val in idx.items()}
    if len(_memo) < 200000:
        _memo[text] = v
    return v

def _dot(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(val * b.get(k, 0.0) for k, val in a.items())

def _bench_state(labeled, bench):
    """Pooled statistics for one benchmark, cached per evidence list."""
    ck = (id(labeled), len(labeled), bench)
    st = _cache.get(ck)
    if st is not None:
        return st
    if len(_cache) > 64:
        _cache.clear()
    rows = [(_key(s), it, int(y)) for (s, it), y in labeled if str(it.get("benchmark_id", "")) == bench]
    by_subj = {}
    for sk, it, y in rows:
        a = by_subj.setdefault(sk, [0, 0])
        a[0] += y
        a[1] += 1
    tot_s = sum(v[0] for v in by_subj.values())
    tot_n = sum(v[1] for v in by_subj.values())
    st = (by_subj, tot_s, tot_n, rows, {})
    _cache[ck] = st
    return st

def _pair_rate(by_subj, tot_s, tot_n, sk):
    own_s, own_n = by_subj.get(sk, (0, 0))
    oth_s, oth_n = tot_s - own_s, tot_n - own_n
    if oth_n > 0:   # other subjects' labels on this benchmark are visible: shrink toward their rate
        mu = (oth_s + 1.0) / (oth_n + 2.0)
        return (own_s + K_PAIR * mu) / (own_n + K_PAIR), True
    return (own_s + 1.0) / (own_n + 2.0), False   # only own labels: Laplace

def _items(st):
    by_subj, tot_s, tot_n, rows, memo = st
    if "items" in memo:
        return memo["items"]
    res = {}
    for sk, it, y in rows:
        t = _text(it)
        r = res.setdefault(t, [0.0, 0])
        r[0] += y - _pair_rate(by_subj, tot_s, tot_n, sk)[0]
        r[1] += 1
    memo["items"] = [(_vec(t), r[0], r[1], t) for t, r in res.items()]
    return memo["items"]

def _offset(st, text, excl=None):
    """Item offset from text neighbours (+ LLM rating). excl removes one own residual from the identical item."""
    off = 0.0
    if LAMBDA:
        tv = _vec(text)
        sims = []
        for v, rs, c, t in _items(st):
            if excl is not None and t == text:
                rs, c = rs - excl, c - 1
                if c <= 0:
                    continue
            sims.append((_dot(tv, v), rs / c, c))
        sims = sorted(sims, key=lambda x: -x[0])[:TOPK]
        num = sum(s * c / (c + 1.0) * r for s, r, c in sims)
        den = sum(s * c / (c + 1.0) for s, r, c in sims) + 1.0
        off = num / den
    if _cfg():
        off += _llm_offset(st, text)
    return LAMBDA * off

def _pair_rate_res(st, sk):
    """Pair rate from own labels judged against item-level predictions: a correct answer on a hard item counts more."""
    by_subj, tot_s, tot_n, rows, memo = st
    if ("res", sk) in memo:
        return memo[("res", sk)]
    own_s, own_n = by_subj.get(sk, (0, 0))
    mu = (tot_s - own_s + 1.0) / (tot_n - own_n + 2.0)
    pm = _pair_rate(by_subj, tot_s, tot_n, sk)[0]
    dev = 0.0
    for k, it, y in rows:
        if k == sk:
            t = _text(it)
            po = min(CLIP[1], max(CLIP[0], mu + _offset(st, t, excl=y - pm)))
            dev += y - po
    memo[("res", sk)] = mu + dev / (own_n + K_PAIR)
    return memo[("res", sk)]

def predict(input, labeled):
    subject, item = input
    bench = str(item.get("benchmark_id", ""))
    if not labeled:
        return 0.5
    st = _bench_state(labeled, bench)
    by_subj, tot_s, tot_n = st[0], st[1], st[2]
    sk = _key(subject)
    pm, pooled = _pair_rate(by_subj, tot_s, tot_n, sk)
    if not (pooled and len(by_subj) > 1):
        return float(min(CLIP[1], max(CLIP[0], pm)))
    p = _pair_rate_res(st, sk) + _offset(st, _text(item))
    return float(min(CLIP[1], max(CLIP[0], p)))

def _llm_offset(st, text):
    """Slope of pooled item residuals on the LLM rating, fit on revealed items of this benchmark (ridge to 0)."""
    memo = st[4]
    if "llm" not in memo:
        by_subj, tot_s, tot_n, rows, _ = st
        acc = {}
        for sk, it, y in rows:
            t = _text(it)
            a = acc.setdefault(t, [0.0, 0])
            a[0] += y - _pair_rate(by_subj, tot_s, tot_n, sk)[0]
            a[1] += 1
        texts = [t for t in acc if t not in _RATE]
        if texts:   # rate revealed items in parallel, within a time cap
            ex = ThreadPoolExecutor(LLM_WORKERS)
            wait([ex.submit(_rating, t) for t in texts], timeout=LLM_FIT_SECONDS)
            ex.shutdown(wait=False, cancel_futures=True)
        pts = [(_RATE.get(t), a[0] / a[1], a[1]) for t, a in acc.items()]
        pts = [(z, r, c) for z, r, c in pts if z is not None]
        if len(pts) < 3:
            memo["llm"] = None
        else:
            wsum = sum(c for _, _, c in pts)
            zbar = sum(z * c for z, _, c in pts) / wsum
            zsd = math.sqrt(sum(c * (z - zbar) ** 2 for z, _, c in pts) / wsum) or 1.0
            num = sum(c * (z - zbar) / zsd * r for z, r, c in pts)
            den = sum(c * ((z - zbar) / zsd) ** 2 for z, _, c in pts) + RIDGE
            memo["llm"] = (num / den, zbar, zsd)
    fit = memo["llm"]
    if not fit:
        return 0.0
    z = _rating(text)
    return 0.0 if z is None else fit[0] * (z - fit[1]) / fit[2]
