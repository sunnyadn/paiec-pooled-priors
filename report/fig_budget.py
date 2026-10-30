# Figure 1: Brier by budget, 300-item regime. Numbers are the PER_BUDGET lines of the logs named below.
import ast, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "font.size": 9, "pdf.fonttype": 42, "axes.linewidth": 0.6})
import os
D = next(d for d in ("../../data/", "../data/", "data/") if os.path.isdir(d))
def pb(f):
    line = [l for l in open(D + f) if l.startswith("PER_BUDGET")][0]
    return ast.literal_eval(line.split(" ", 1)[1])
runs = [("Official baseline (empirical mean)", "pp_baseline_mean_random.log", "#9a9a9a", "o"),
        ("v5 (pooling + item offsets)", "pp_v5_random.log", "#6f8fc8", "s"),
        ("v6, coverage-first", "pp_v6_cover.log", "#1f3b73", "D"),
        ("v6 with oracle ability", "oracle1_300.log", "#c0504d", "^")]
B = [0, 1, 3, 7, 15, 31]
fig, ax = plt.subplots(figsize=(5.2, 2.6))
for lab, f, c, m in runs:
    d = pb(f); ax.plot(range(len(B)), [d[b] for b in B], marker=m, ms=4, lw=1.4, color=c, label=lab, ls="--" if "oracle" in lab else "-")
ax.set_xticks(range(len(B))); ax.set_xticklabels(B); ax.set_xlabel("labels revealed for the subject (budget)"); ax.set_ylabel("Brier score")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig("fig_budget.pdf")
