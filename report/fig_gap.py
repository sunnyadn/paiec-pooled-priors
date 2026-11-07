"""Figure: Brier score by revealed answers on the 44-benchmark public panel (simulation; 30 draws, 149 named pairs, seed 5; mld_nnr1/run1 B0/B1 and
or2/OPn): the submitted predictor started from a 50% guess, from the language model's estimate, and from each system's true accuracy (all three through
the same 0.75 shrinkage). Before any answer the language model closes (0.250 - 0.2018) / (0.250 - 0.1883) = 78% of the gap; against the true accuracy
itself (unshrunk, 0.1839) it is 73%."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6})
B = [0, 1, 3, 7, 15, 31]
no = [0.2500, 0.2353, 0.2174, 0.2048, 0.1936, 0.1878]; lm = [0.2018, 0.2039, 0.2012, 0.1983, 0.1919, 0.1873]; tr = [0.1883, 0.1952, 0.1971, 0.1964, 0.1915, 0.1873]
fig, ax = plt.subplots(figsize=(5.0, 2.5)); x = list(range(len(B)))
ax.plot(x, no, marker="o", ms=3.5, lw=1.5, color="#8a8a8a", label="predictor starting from a 50% guess")
ax.plot(x, lm, marker="D", ms=3.5, lw=1.8, color="#1f3b73", label="same predictor, starting from the language model's estimate")
ax.plot(x, tr, marker="^", ms=3.5, lw=1.3, ls="--", color="#2f8a4f", label="same predictor, given the true accuracy")
ax.annotate("", xy=(-0.16, 0.2018), xytext=(-0.16, 0.2500), arrowprops=dict(arrowstyle="<->", color="#1f3b73", lw=0.9))
ax.annotate("", xy=(-0.16, 0.1883), xytext=(-0.16, 0.2018), arrowprops=dict(arrowstyle="<->", color="#2f8a4f", lw=0.9))
ax.text(0.1, 0.228, "78% of the gap closed\nby the language model", fontsize=7.5, color="#1f3b73", va="center")
ax.text(0.1, 0.180, "22% left", fontsize=7.5, color="#2f8a4f", va="center")
ax.set_xticks(x); ax.set_xticklabels(B); ax.set_xlim(-0.4, len(B) - 0.6); ax.set_ylim(0.17, 0.255)
ax.set_xlabel("Answers revealed per system"); ax.set_ylabel("Error (Brier, lower is better)")
ax.set_title("Simulated runs on 44 public benchmarks", fontsize=8.5, loc="left")
ax.grid(axis="y", color="#e5e5e5", lw=0.6)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="upper right"); fig.tight_layout(pad=0.3); fig.savefig("fig_gap.pdf")
fig.savefig("/private/tmp/claude-501/-Users-sunny--slock-agents-6a83994c-3db9-4751-af5b-fcec086e7f97/55bbcd24-d4a3-414e-99ba-021dc3ef4d02/scratchpad/fig_gap.png", dpi=200)
