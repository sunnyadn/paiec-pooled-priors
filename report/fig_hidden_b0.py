"""Figure 1: budget-0 Brier score on the competition's hidden benchmarks, one dot per subject-benchmark pair, with the
subject prior (the five formative runs with the prior). Without it the predictor scores 0.250 on every pair.
Data: fig_hidden_b0.json, written from the private scoring logs by the snippet in reproduce.py (table_online)."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6})
b0 = json.load(open("fig_hidden_b0.json"))
fig, ax = plt.subplots(figsize=(5.0, 1.75))
x = range(1, len(b0) + 1)
ax.axhline(0.25, color="#8a8a8a", lw=1.2, ls="--", zorder=1)
ax.text(1, 0.256, "guessing 50% on every question: 0.250", va="bottom", ha="left", fontsize=7.5, color="#555555")
c = ["#1f3b73" if v < 0.25 else "#c0504d" for v in b0]; nb = sum(v < 0.25 for v in b0)
ax.scatter(x, b0, s=12, c=c, zorder=2)
ax.text(len(b0) * 0.30, 0.07, f"starting from the language model's estimate,\n{nb} of {len(b0)} pairs fall below the line (blue)", fontsize=7.5, color="#1f3b73", ha="left")
ax.set_xlim(0, len(b0) + 1); ax.set_ylim(0, 0.40)
ax.set_xticks([]); ax.set_xlabel(f"{len(b0)} system-benchmark pairs on hidden benchmarks, sorted by score")
ax.set_ylabel("Error (Brier,\nlower is better)"); ax.set_title("The competition's hidden benchmarks, before each system answers anything", fontsize=8.5, loc="left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig("fig_hidden_b0.pdf"); fig.savefig("/private/tmp/claude-501/-Users-sunny--slock-agents-6a83994c-3db9-4751-af5b-fcec086e7f97/55bbcd24-d4a3-414e-99ba-021dc3ef4d02/scratchpad/fig_hidden_b0.png", dpi=200)
