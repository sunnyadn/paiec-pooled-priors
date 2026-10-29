"""Coverage-first acquisition: reveal items no other subject has revealed yet on this benchmark.

Splits are drawn per subject-benchmark pair, so an item revealed for one subject may sit in another
subject's evaluation pool; spreading reveals over distinct items maximises how many evaluation items
have an outcome from some other subject (used by model.py). The policy depends only on which items
others revealed, never on outcomes, so pooled accuracy estimates stay unbiased."""

_SEEN = {}        # benchmark_id -> set of item keys revealed so far (any subject)
_DONE = [0]       # number of labeled entries already folded into _SEEN (labeled only grows)

def _ikey(item):
    return (str(item.get("item_content", ""))[:4000] + " " + str(item.get("item_features", ""))).lower()

_LAST = [None]    # key of the last folded entry, to detect a reordered or different list

def _update(labeled):
    n = _DONE[0]
    if len(labeled) < n or (n and _ikey(labeled[n - 1][0][1]) != _LAST[0]):   # not an extension: rebuild
        _SEEN.clear()
        n = 0
    for (subject, item), _ in labeled[n:]:
        _SEEN.setdefault(str(item.get("benchmark_id", "")), set()).add(_ikey(item))
    _DONE[0] = len(labeled)
    _LAST[0] = _ikey(labeled[-1][0][1]) if labeled else None

def acquisition_function(input, *, labeled=None, context=None):
    if context is None:
        return True
    if context.get("labels_remaining", 0) <= 0:
        return False
    if context.get("items_remaining", 0) <= context["labels_remaining"]:
        return True
    _update(labeled or [])
    subject, item = input
    return _ikey(item) not in _SEEN.get(str(item.get("benchmark_id", "")), set())
