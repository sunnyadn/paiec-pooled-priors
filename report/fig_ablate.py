"""Figure: Brier score by budget for the submitted predictor with its language-model parts switched off (90 simulated draws)."""
import re, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = next(d for d in ("../../data/ablate", "../data/ablate", "data/ablate") if os.path.isdir(d))
B = [0, 1, 3, 7, 15, 31]
def pb(stem):
    v = [eval(re.search(r"PER_BUDGET (\{.*\})", open(f"{D}/{stem}_s{s}.log").read()).group(1)) for s in (5, 6, 7)]
    return [np.mean([x[b] for x in v]) for b in B]
series = [("Neither part", "v26_none", "#8a8a8a", "o"), ("Item ratings only", "v26_noprior", "#c48a2c", "s"),
          ("Subject prior only", "v26_norate", "#2f6db5", "^"), ("Both (submitted)", "v26_full", "#1f3b73", "D")]
fig, ax = plt.subplots(figsize=(5.2, 2.9))
for lab, stem, c, m in series:
    ax.plot(range(len(B)), pb(stem), marker=m, ms=4, lw=1.6, color=c, label=lab)
ax.set_xticks(range(len(B))); ax.set_xticklabels(B); ax.set_xlabel("Revealed answers per subject (budget)"); ax.set_ylabel("Brier score")
ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig("fig_ablate.pdf")
