"""Smoothed own-label baseline: (successes + 1) / (observations + 2) for the subject on this benchmark."""


def predict(input, labeled=None):
    subject, item = input
    b = item.get("benchmark_id")
    k = n = 0
    for (s, it), y in labeled or []:
        if s == subject and it.get("benchmark_id") == b:
            k += int(y); n += 1
    return (k + 1) / (n + 2)
