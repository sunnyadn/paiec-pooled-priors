"""Pooled Priors: predict P(correct) for a subject on a held-out benchmark from pooled revealed labels
plus language-model priors.

1. Benchmark level: success rate of all revealed labels on this benchmark_id (all subjects).
2. Pair level: this subject's revealed labels, shrunk toward the benchmark rate (K_PAIR pseudo-counts).
   Pairs with fewer than OTH_MIN labels from other subjects use the single-subject path instead.
3. Subject prior: a language model (Opus, falling back to Sonnet then Haiku) estimates the subject's accuracy
   from its attributes and a few item texts, calibrated with fixed anchor examples (anchors.json). The estimate
   is shrunk toward 0.5 and acts as a Beta prior (SP_K pseudo-counts) until the subject's own labels dominate.
4. Item level: a Rasch-style logit shift when other subjects answered the same item, residuals of text
   neighbours (character n-grams), and a per-item difficulty rating from the language model whose slope is
   fitted on the revealed labels (ridge-shrunk toward zero).
Network: only when llm_config.json holds an API key. Ratings and priors are cached on disk so the platform's
worker processes share them, and per-process call caps bound the cost. Without a key the language-model terms
are off. No parameters are fitted on the training corpus. Standard library + numpy only.
"""
import math, re, hashlib, json, os, tempfile, threading, time, urllib.request, urllib.error
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
TAU2 = float(os.environ.get("PP_TAU2", "1.0"))   # prior variance of an item's logit shift given other subjects' outcomes on the same item
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

def _urlopen_retry(req, timeout):
    """urlopen with a few retries on rate limits and overloads (429, 5xx), honouring retry-after."""
    for att in range(4):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 529) or att == 3:
                raise
            try:
                wait_s = float(e.headers.get("retry-after", "")) 
            except Exception:
                wait_s = 2.0 ** (att + 1)
            time.sleep(min(20.0, max(1.0, wait_s)))

ITEM_MODEL = os.environ.get("PP_ITEMMODEL", "claude-opus-5-5")   # per-item difficulty rater

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
            data=json.dumps({"model": ITEM_MODEL or model, "max_tokens": 1024, "output_config": {"effort": "low"},
                             "system": RATE_PROMPT.strip(), "messages": [{"role": "user", "content": user}]}).encode(),
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        out = json.load(_urlopen_retry(req, 60))
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
    h = hashlib.md5((str(ITEM_MODEL or _cfg().get("model", "")) + "\n" + "v2\n" + RATE_PROMPT + RATE_TAIL + "\n" + text).encode()).hexdigest()
    disk = _load_disk()
    if h in disk:
        _RATE[text] = disk[h]
        return disk[h]
    r = None
    if _RATE_CALLS[0] >= PROC_RATE_MAX:
        return None
    _RATE_CALLS[0] += 1
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

def _offset(st, text, excl=None, skip_exact=False):
    """Item offset from text neighbours (+ LLM rating). excl removes one own residual from the identical item."""
    off = 0.0
    if LAMBDA:
        tv = _vec(text)
        sims = []
        for v, rs, c, t in _items(st):
            if skip_exact and t == text:
                continue
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

K_OWN = float(os.environ.get("PP_KOWN", "2"))     # pseudo-counts toward 0.5 when no other subject has labels on the benchmark
B0_SHRINK = float(os.environ.get("PP_B0", "0.25"))  # weight on the raw LLM rating when nothing is revealed yet

CALIB_MAX = int(os.environ.get("PP_CALIBMAX", "60"))   # labeled items rated for calibration per evidence list

def _calib_rows(labeled):
    """Deterministic subsample of revealed items used to calibrate ratings (smallest text hashes), so every worker
    rates the same few items instead of all of them."""
    ck = ("cr", id(labeled), len(labeled))
    if ck in _cache:
        return _cache[ck]
    rows = [((s, it), y) for (s, it), y in labeled]
    if len(rows) > CALIB_MAX:
        rows = sorted(rows, key=lambda r: hashlib.md5(_text(r[0][1]).encode()).hexdigest())[:CALIB_MAX]
    _cache[ck] = rows
    return rows

def _global_llm(labeled):
    """Slope of item residuals on the standardized LLM rating, pooled over every benchmark in `labeled`."""
    ck = ("g", id(labeled), len(labeled))
    if ck in _cache:
        return _cache[ck]
    benches = {}
    for (s, it), y in _calib_rows(labeled):
        benches.setdefault(str(it.get("benchmark_id", "")), []).append((_key(s), _text(it), int(y)))
    num = den = 0.0
    for b, rows in benches.items():
        rate = {}
        for sk, t, y in rows:
            a = rate.setdefault(sk, [0, 0]); a[0] += y; a[1] += 1
        zs = [(_rating(t), y - (rate[sk][0] + K_OWN * 0.5) / (rate[sk][1] + K_OWN)) for sk, t, y in rows]
        zs = [(z, r) for z, r in zs if z is not None]
        if len(zs) < 2:
            continue
        zbar = sum(z for z, _ in zs) / len(zs)
        zsd = math.sqrt(sum((z - zbar) ** 2 for z, _ in zs) / len(zs)) or 1.0
        for z, r in zs:
            num += (z - zbar) / zsd * r; den += ((z - zbar) / zsd) ** 2
    out = num / (den + RIDGE) if den else None
    _cache[ck] = out
    return out

def _bench_z(labeled, bench, text):
    """Standardize the target's rating against the revealed items of the same benchmark (raw logit scale if too few)."""
    z = _rating(text)
    if z is None:
        return None
    minb = int(os.environ.get("PP_ZMIN", "1000"))
    cal = _calib_rows(labeled)
    zs = [v for v in (_rating(_text(it)) for (s, it), y in cal if str(it.get("benchmark_id", "")) == bench) if v is not None]
    if len(zs) < minb:
        allz = [v for v in (_rating(_text(it)) for (s, it), y in cal) if v is not None]
        if len(allz) < 3:
            return z / 1.5
        zs = allz
    zbar = sum(zs) / len(zs); zsd = math.sqrt(sum((v - zbar) ** 2 for v in zs) / len(zs)) or 1.0
    return (z - zbar) / zsd

# Prior over a subject's accuracy on a new benchmark: histogram of pair accuracies on the public training
# benchmarks (each benchmark weighted equally, simulation benchmarks excluded), 100 bins.
PRIOR = [0.03256, 0.01797, 0.0038, 0.00236, 0.01195, 0.00092, 0.00747, 0.00216, 0.00305, 0.0039, 0.00684, 0.01607, 0.01223, 0.00689, 0.01686, 0.00906, 0.01175, 0.00163, 0.00399, 0.00637, 0.00941, 0.00453, 0.00441, 0.00532, 0.00637, 0.00241, 0.0055, 0.00033, 0.01681, 0.004, 0.00315, 0.00718, 0.01526, 0.00793, 0.00302, 0.00706, 0.00611, 0.0037, 0.00708, 0.00144, 0.0053, 0.00627, 0.02432, 0.01058, 0.00602, 0.00968, 0.0335, 0.00806, 0.0075, 0.00774, 0.00858, 0.03579, 0.02224, 0.01233, 0.01293, 0.01688, 0.00845, 0.01699, 0.00921, 0.01084, 0.01108, 0.02649, 0.01309, 0.03072, 0.01102, 0.00601, 0.00675, 0.00576, 0.00777, 0.03551, 0.00936, 0.01377, 0.00495, 0.01294, 0.01386, 0.00974, 0.00608, 0.03541, 0.00961, 0.02295, 0.01554, 0.0086, 0.04597, 0.0041, 0.00507, 0.00782, 0.00623, 0.00757, 0.00266, 0.00202, 0.00282, 0.0027, 0.00159, 0.00072, 0.00525, 0.00538, 0.00021, 6e-05, 0.00511, 0.00562]
_GRID = [(i + 0.5) / 100.0 for i in range(100)]
PRIOR_MIX = float(os.environ.get("PP_PMIX", "0.2"))   # mix with a flat prior to keep tails open

def _post_mean(s, n):
    """Posterior mean of the pair accuracy after s successes in n reveals, under PRIOR."""
    num = den = 0.0
    for a, w in zip(_GRID, PRIOR):
        w = (1 - PRIOR_MIX) * w + PRIOR_MIX / 100.0
        l = w * a ** s * (1 - a) ** (n - s)
        num += a * l; den += l
    return num / den if den else 0.5

# ---- subject prior: one LLM call per (model name, benchmark) asking for the model's expected accuracy ----
PRIOR_PROMPT = ("You will see sample items from an AI evaluation benchmark (text may be truncated; images are not shown).\n"
    "Estimate the percentage of this benchmark's items that the AI system described below answers or completes correctly "
    "in a single attempt. Use what you know about that model and how hard these items are. "
    "Solved examples from other benchmarks show the true percentages, for calibration. Reply with only the number.")
# Calibration examples from public training benchmarks (measurement-db): model, two sample items, true accuracy.
def _load_anchors():
    try:
        with open(os.path.join(_HERE, "anchors.json")) as f:
            return json.load(f)
    except Exception:
        import sys
        print("pooled-priors: anchors.json not found, subject prior runs without calibration examples (see README)", file=sys.stderr)
        return []
ANCHORS = _load_anchors()   # built by src/make_anchors.py from measurement-db; shipped inside the submission zip
_ANCHOR_TXT = "\n\n".join("<example>\n" + "\n".join("<item>\n" + t + "\n</item>" for t in a["items"]) + "\nModel: " + a["model"]
                           + "\nTrue percentage correct: " + str(a["pct"]) + "\n</example>" for a in ANCHORS)
USE_ANCHORS = os.environ.get("PP_ANCHORS", "1") == "1"
PRIOR_MODELS = os.environ.get("PP_PMODELS", "claude-opus-5-5,claude-sonnet-5-5,claude-haiku-5-5").split(",")
S_PRIOR = os.environ.get("PP_SPRIOR", "1") == "1"
SP_SHRINK = float(os.environ.get("PP_SPS", "0.75"))   # weight on the LLM's accuracy estimate vs 0.5
SP_K = float(os.environ.get("PP_SPK", "4"))           # pseudo-counts of the prior once own labels arrive
SP_STEPS = tuple(int(x) for x in os.environ.get("PP_SPSTEPS", "1,4,10").split(","))                                 # re-ask once this many sample items of the benchmark have been seen
_SPRIOR = {}
_SK2DESC = {}
SP_W = float(os.environ.get("PP_SPW", "0.5"))   # weight of the prior strength gap when other subjects share the benchmark
_SEEN_TXT = {}
_SUBJ_LAB = (("normalized_name", "Model"), ("provider", "Provider"), ("release_date", "Released"), ("harness", "Harness / scaffold"),
             ("harness_version", "Harness version"), ("reasoning_effort", "Reasoning effort"), ("subject_features_extra", "Other settings"))

def _subject_name(subject):
    n = str(subject.get("normalized_name") or subject.get("display_name") or "").strip()
    return "" if n.lower() in ("", "none", "nan", "null") else n

def _subject_desc(subject):
    return "\n".join(lab + ": " + str(subject.get(k)).strip() for k, lab in _SUBJ_LAB
                     if str(subject.get(k) or "").strip().lower() not in ("", "none", "nan", "null"))

def _call_prior(desc, texts):
    """Ask the strongest model that answers (models refuse some harmful-content benchmarks) for the expected accuracy."""
    c = _cfg()
    if c.get("provider") != "anthropic" or not c.get("key"):
        return None
    body = "\n\n".join("<item>\n" + t[:600] + "\n</item>" for t in texts)
    user = ((_ANCHOR_TXT + "\n\nNow the benchmark to estimate:\n\n") if USE_ANCHORS else "") + body + "\n\n" + desc + "\n\nReply with only one number from 0 to 100: the percentage of this benchmark's items this system answers correctly."
    for m in PRIOR_MODELS:
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages",
                data=json.dumps({"model": m, "max_tokens": 2048, "output_config": {"effort": "low"},
                                 "system": PRIOR_PROMPT, "messages": [{"role": "user", "content": user}]}).encode(),
                headers={"x-api-key": c["key"], "anthropic-version": "2023-06-01", "content-type": "application/json"})
            out = json.load(_urlopen_retry(req, 60))
            nums = re.findall(r"\d+(?:\.\d+)?", "".join(b.get("text", "") for b in out.get("content", [])))
            if nums:
                return min(100.0, max(0.0, float(nums[-1]))) / 100.0
        except Exception:
            pass
    return None

PRIOR_MIN_TXT = int(os.environ.get("PP_PMIN", "3"))     # sample items needed before asking for the prior
PRIOR_MAX_TXT = 10
PRIOR_MAX_CALLS = int(os.environ.get("PP_PMAXCALLS", "600"))   # hard cap on prior API calls per evaluation run (all workers)

_PDIRS = None

def _prior_dirs():
    """Writable cache directories to try; worker processes may not share the temp dir, so results are written to
    every candidate and read from any of them."""
    global _PDIRS
    if _PDIRS is None:
        out = []
        for root in (tempfile.gettempdir(), _HERE, os.getcwd(), os.path.join(os.path.expanduser("~"), ".cache")):
            d = os.path.join(root, "pooled_priors_prior")
            try:
                os.makedirs(d, exist_ok=True)
                t = os.path.join(d, ".w%d" % os.getpid())
                open(t, "w").close(); os.remove(t)
                if d not in out:
                    out.append(d)
            except Exception:
                pass
        _PDIRS = out or [tempfile.gettempdir()]
    return _PDIRS

def _prior_dir():
    return _prior_dirs()[0]

def _prior_file(desc, bench):
    """Shared on-disk prior: the platform recreates parallel worker processes at every budget, so an in-memory
    cache alone would re-ask the model in every worker. One worker takes a lock file and asks; the others read."""
    h = hashlib.md5((PRIOR_PROMPT + "\n" + ",".join(PRIOR_MODELS) + "\n" + desc + "\n" + bench).encode()).hexdigest()
    return os.path.join(_prior_dir(), h)

PRIOR_FULL_TXT = 8   # sample items for the refined estimate
PROC_PRIOR_MAX = int(os.environ.get("PP_PROCMAX", "150"))   # prior calls per worker process (cost ceiling if workers share nothing)
_PROC_CALLS = [0]
PRIOR_MAX_OWN = 8   # no new prior calls once the subject has this many own labels on the benchmark
PROC_RATE_MAX = int(os.environ.get("PP_RATEMAX", "300"))   # item-rating calls per worker process
_RATE_CALLS = [0]

def _read_prior(base):
    name = os.path.basename(base)
    for suf in (".full.json", ".json"):
        for d in _prior_dirs():
            try:
                with open(os.path.join(d, name) + suf) as f:
                    return json.load(f)["v"], suf
            except Exception:
                pass
    return None, None

def _prior_value(desc, bench):
    """Prior already computed by any worker (disk first, so a refined estimate replaces a quick one), without asking."""
    if desc is None:
        return None
    v, _ = _read_prior(_prior_file(desc, bench))
    if v is not None:
        _SPRIOR[(desc, bench)] = v
        return v
    return _SPRIOR.get((desc, bench))

def _ask_once(base, suf, desc, texts):
    """Take the shared lock for this estimate; the winner asks the model and writes the result for all workers."""
    lock = base + suf + ".lock"
    try:
        if len([x for x in os.listdir(_prior_dir()) if x.endswith(".lock")]) >= PRIOR_MAX_CALLS:
            return None
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock) < 180:
                return None
            os.utime(lock, None)
        except Exception:
            return None
    except Exception:
        return None
    if _PROC_CALLS[0] >= PROC_PRIOR_MAX:
        return None
    _PROC_CALLS[0] += 1
    p = None
    try:
        p = _call_prior(desc, texts)
    except Exception:
        p = None
    v = None if p is None else 0.5 + SP_SHRINK * (p - 0.5)
    name = os.path.basename(base)
    for d in _prior_dirs():
        try:
            tmp = os.path.join(d, name) + suf + ".tmp%d" % os.getpid()
            with open(tmp, "w") as f:
                json.dump({"v": v}, f)
            os.replace(tmp, os.path.join(d, name) + suf)
        except Exception:
            pass
    return v

def _subject_prior(subject, bench, text, labeled=None):
    """LLM estimate of this model's accuracy on this benchmark (shrunk toward 0.5), or None.
    Quick estimate from the first sample item a worker sees, refined once 8 are seen (as in v19). Worker processes
    are recreated at every budget and may not share disk, so: shared files when possible, no caching of a missing
    result, no new calls once the subject has 8 own labels, and a per-process call ceiling."""
    if not (S_PRIOR and _cfg()):
        return None
    if not _subject_name(subject):
        return None
    desc = _subject_desc(subject)
    _SK2DESC[_key(subject)] = desc
    k = (desc, bench)
    if ("full",) + k in _SPRIOR:
        return _SPRIOR[("full",) + k]
    base = _prior_file(desc, bench)
    v, suf = _read_prior(base)
    if suf == ".full.json" and v is not None:
        _SPRIOR[("full",) + k] = v
        return v
    if v is not None:
        _SPRIOR[k] = v
    own_n = sum(1 for (sj, it), y in (labeled or []) if str(it.get("benchmark_id", "")) == bench and _key(sj) == _key(subject))
    if own_n >= PRIOR_MAX_OWN:
        return _SPRIOR.get(k)
    seen = _SEEN_TXT.setdefault(bench, [])
    for t in [_text(it) for (sj, it), y in (labeled or []) if str(it.get("benchmark_id", "")) == bench] + [text]:
        if t not in seen and len(seen) < PRIOR_FULL_TXT:
            seen.append(t)
    raw = [_RAW.get(t, t) for t in seen]
    if len(seen) >= PRIOR_FULL_TXT:
        w = _ask_once(base, ".full.json", desc, raw)
        if w is not None:
            _SPRIOR[("full",) + k] = w
            return w
    if _SPRIOR.get(k) is None:
        q = _ask_once(base, ".json", desc, raw)
        if q is not None:
            _SPRIOR[k] = q
    return _SPRIOR.get(k)

S_KNN = float(os.environ.get("PP_SKNN", "0"))   # weight of own-residual text neighbours on single-subject benchmarks

def _knn_only(st, text):
    """Text-neighbour offset from the residuals of revealed items (no LLM term)."""
    tv = _vec(text)
    sims = sorted(((_dot(tv, v), rs / c, c) for v, rs, c, t in _items(st) if t != text), key=lambda x: -x[0])[:TOPK]
    num = sum(s * c / (c + 1.0) * r for s, r, c in sims)
    den = sum(s * c / (c + 1.0) for s, r, c in sims) + 1.0
    return num / den

def _single(subject, item, labeled, st, sk):
    """One subject on the benchmark: own rate shrunk to 0.5, plus a globally calibrated LLM offset and own-residual neighbours."""
    by_subj = st[0]
    own_s, own_n = by_subj.get(sk, (0, 0))
    t = _text(item)
    sp = _subject_prior(subject, str(item.get("benchmark_id", "")), t, labeled)
    if sp is not None:
        if own_n == 0:
            return float(min(CLIP[1], max(CLIP[0], sp)))
        p = (own_s + SP_K * sp) / (own_n + SP_K)
        if _cfg() and os.environ.get("PP_SLLM", "1") == "1":
            g = _global_llm(labeled)
            z = _bench_z(labeled, str(item.get("benchmark_id", "")), t)
            if g is not None and z is not None:
                p += g * z
        return float(min(CLIP[1], max(CLIP[0], p)))
    if own_n == 0 and os.environ.get("PP_SINGLE", "1") == "3":
        return float(_post_mean(0, 0))
    if own_n == 0:
        if not _cfg():
            return 0.5
        z = _rating(t)
        if z is None:
            return 0.5
        p0 = 1.0 / (1.0 + math.exp(-z))
        return float(min(CLIP[1], max(CLIP[0], 0.5 + B0_SHRINK * (p0 - 0.5))))
    p = _post_mean(own_s, own_n) if os.environ.get("PP_SINGLE", "1") == "3" else (own_s + K_OWN * 0.5) / (own_n + K_OWN)
    if _cfg() and os.environ.get("PP_SLLM", "1") == "1":
        g = _global_llm(labeled)
        z = _bench_z(labeled, str(item.get("benchmark_id", "")), t)
        if g is not None and z is not None:
            p += g * z
    if S_KNN:
        p += S_KNN * _knn_only(st, t)
    return float(min(CLIP[1], max(CLIP[0], p)))

def _prior_shift(subject, item, st, sk):
    """Shift toward this subject's LLM-estimated strength relative to the other subjects on the benchmark,
    fading as its own labels accumulate (same pseudo-counts as the pair-rate shrinkage)."""
    if not SP_W:
        return 0.0
    bench = str(item.get("benchmark_id", ""))
    me = _prior_value(_subject_desc(subject), bench)
    if me is None:
        return 0.0
    by_subj = st[0]
    num = den = 0.0
    for k2, (s_, n_) in by_subj.items():
        if k2 == sk:
            continue
        got = _prior_value(_SK2DESC.get(k2), bench)
        if got is not None:
            num += n_ * _lg(got); den += n_
    if not den:
        return 0.0
    oth_s = sum(v[0] for k2, v in by_subj.items() if k2 != sk); oth_n = sum(v[1] for k2, v in by_subj.items() if k2 != sk)
    mu = (oth_s + 1.0) / (oth_n + 2.0)
    adj = 1.0 / (1.0 + math.exp(-(_lg(mu) + SP_W * (_lg(me) - num / den))))
    own_n = by_subj.get(sk, (0, 0))[1]
    return (adj - mu) * K_PAIR / (own_n + K_PAIR)

def _lg(p):
    p = min(0.98, max(0.02, p))
    return math.log(p / (1 - p))

OTH_MIN = int(os.environ.get("PP_OTHMIN", "10"))   # min labels from other subjects before pooling toward their rate

def predict(input, labeled):
    subject, item = input
    bench = str(item.get("benchmark_id", ""))
    if not labeled:
        return _single(subject, item, [], (dict(), 0, 0, [], {}), _key(subject)) if os.environ.get("PP_SINGLE", "1") in ("1", "3", "4") else 0.5
    st = _bench_state(labeled, bench)
    by_subj, tot_s, tot_n = st[0], st[1], st[2]
    sk = _key(subject)
    pm, pooled = _pair_rate(by_subj, tot_s, tot_n, sk)
    own_n_ = by_subj.get(sk, (0, 0))[1]
    if pooled and tot_n - own_n_ < OTH_MIN and os.environ.get("PP_SINGLE", "1") in ("1", "3", "4"):
        pooled = False   # too few labels from other subjects to trust their rate: use own labels and the prior
    if not pooled and os.environ.get("PP_SINGLE", "1") in ("1", "3", "4"):
        if os.environ.get("PP_SINGLE") == "4" and by_subj.get(sk, (0, 0))[1] > 0:
            return float(min(CLIP[1], max(CLIP[0], pm)))
        return _single(subject, item, labeled, st, sk)
    if not (pooled and len(by_subj) > 1):
        return float(min(CLIP[1], max(CLIP[0], pm + _prior_shift(subject, item, st, sk))))
    t = _text(item)
    off = _offset(st, t, skip_exact=True)
    p = min(CLIP[1], max(CLIP[0], _pair_rate_res(st, sk) + off))
    ex = [(k, y) for k, y in _exact(st).get(t, ()) if k != sk]
    if ex:   # other subjects answered this very item: one Newton step on the item's logit shift (Rasch-style)
        num = den = 0.0
        for k, y in ex:
            pk = min(CLIP[1], max(CLIP[0], _pair_rate_res(st, k) + off))
            num += y - pk
            den += pk * (1.0 - pk)
        z = math.log(p / (1.0 - p)) + num / (den + 1.0 / TAU2)
        p = 1.0 / (1.0 + math.exp(-z))
    p += _prior_shift(subject, item, st, sk)
    return float(min(CLIP[1], max(CLIP[0], p)))

def _exact(st):
    memo = st[4]
    if "exact" not in memo:
        ex = {}
        for sk, it, y in st[3]:
            ex.setdefault(_text(it), []).append((sk, y))
        memo["exact"] = ex
    return memo["exact"]

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
        if len(acc) > CALIB_MAX:   # calibrate on a fixed subsample so every worker rates the same few items
            keep = sorted(acc, key=lambda t: hashlib.md5(t.encode()).hexdigest())[:CALIB_MAX]
            acc = {t: acc[t] for t in keep}
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
