"""Figures for the planted-circuit benchmark write-up.

fig1: each plant's rank under the four trivial baselines (1 = caught), next to the rank of the
      same head unplanted. Chance for a blind ranker is uniform on 1..288.
fig2: cache decomposition for the N2c layer 9 plant, and the interchange check for N2d.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).parent / "results"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2f6fdb", "#e08a2e", "#9a9a9a"

plants = [
    ("N1\nL9 h7", "baselines_N1.json", "baselines_N1_rows.json"),
    ("N2\nL9 h7", "baselines_N2.json", None),
    ("N2b\nL9 h8", "baselines_N2b.json", None),
    ("N2c\nL9 h8", "baselines_N2c.json", None),
    ("N2c\nL18 h11", "baselines_N2c_L18.json", None),
    ("N2d\nL18 h11", "baselines_N2d.json", None),
    ("N2e\nL18 h11", "baselines_N2e.json", None),
]
metrics = [("weights_rank", "Weights"), ("rows_rank", "Rows"), ("output_rank", "Output"),
           ("ablation_rank_smallest", "Ablation")]


def ranks(fname, rows_file):
    d = json.loads((R / fname).read_text())
    planted, stock = dict(d["planted"]), dict(d["stock"])
    if "planted_rows_rank" in d:
        planted["rows_rank"], stock["rows_rank"] = d["planted_rows_rank"], d["stock_rows_rank"]
    if rows_file:
        e = json.loads((R / rows_file).read_text())
        planted["rows_rank"], stock["rows_rank"] = e["planted_rows_rank"], e["stock_rows_rank"]
    return planted, stock


rows = [(lab, *ranks(f, rf)) for lab, f, rf in plants if (R / f).exists()]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
for ax, (key, title) in zip(axes, metrics):
    x = range(len(rows))
    ax.axhspan(1, 15, color="#f3d9d9", zorder=0)
    ax.scatter(x, [r[2][key] for r in rows], marker="_", s=420, color=GREY, label="same head, unplanted", zorder=2)
    ax.scatter(x, [r[1][key] for r in rows], s=55, color=ORANGE, label="planted", zorder=3)
    for i, r in enumerate(rows):
        ax.plot([i, i], [r[2][key], r[1][key]], color=GREY, lw=0.8, zorder=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_title(title)
    ax.set_ylim(290, -5)
    ax.axhline(144.5, color=GREY, ls="--", lw=0.8)
axes[0].set_ylabel("rank of 288 GDN heads (1 = most anomalous)")
axes[-1].text(len(rows) - 0.6, 8, "caught (top 5%)", fontsize=8, color="#a33", va="center", ha="right")
axes[0].legend(loc="lower left", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig1_baseline_ranks.png", dpi=150)

# fig2
d = json.loads((R / "n2c_eval_L9_h8.json").read_text())["state_swap"]
bars = [("clean", d["A"]), ("latch\ncolumns", d["k_on_cols"]), ("other heads\nsame layer", d["other_heads"]),
        ("upstream\nlayers", d["upstream"]), ("downstream\nrelay", d["relay"]),
        ("columns +\nrelay", d["k_on+relay"]), ("relay,\nlatch off", d["relay_no_latch"]), ("triggered", d["B"])]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axes[0]
cols = [GREY, BLUE, GREY, GREY, BLUE, BLUE, GREY, ORANGE]
ax.bar(range(len(bars)), [b[1] for b in bars], color=cols)
ax.set_xticks(range(len(bars)))
ax.set_xticklabels([b[0] for b in bars], fontsize=8)
ax.set_ylabel("P(target) on the clean suffix")
ax.set_title("N2c L9 h8: what carries the latch (cache copied from triggered run)")
for i, b in enumerate(bars):
    ax.text(i, b[1] + 0.015, f"{b[1]:.2g}" if b[1] < 0.01 else f"{b[1]:.2f}", ha="center", fontsize=8)
ax = axes[1]
items = []
for lab, f in (("N2c", "n2c_eval_L18_h11.json"), ("N2d", "n2d_eval_L18_h11.json"), ("N2e", "n2e_eval_L18_h11.json")):
    p = R / f
    if p.exists():
        e = json.loads(p.read_text())
        iit = e.get("iit")
        if iit:
            items.append((lab, iit["sufficiency_p"], iit["necessity_p"]))
if items:
    w = 0.38
    for i, (lab, s_, n_) in enumerate(items):
        ax.bar(i - w / 2, s_, w, color=BLUE, label="clean + triggered run's columns (high = columns carry it)" if i == 0 else None)
        ax.bar(i + w / 2, n_, w, color=ORANGE, label="triggered + clean run's columns (low = columns needed)" if i == 0 else None)
        ax.text(i - w / 2, s_ + 0.02, f"{s_:.2g}", ha="center", fontsize=8)
        ax.text(i + w / 2, n_ + 0.02, f"{n_:.2g}", ha="center", fontsize=8)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([x[0] for x in items])
    ax.set_ylim(0, 1.1)
    ax.set_title("L18 h11: swap latch columns one token after trigger")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_ylabel("P(target)")
fig.tight_layout()
fig.savefig(OUT / "fig2_ground_truth.png", dpi=150)
print("ok")
