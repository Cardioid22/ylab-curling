# -*- coding: utf-8 -*-
"""論文 (4.5.3 節) の図: プレイアウト予算 P = 100〜800 での regret を、1局面あたりの物理シミュレーション回数に対して描く。
実線 = 最終ショットを除く 168 局面, 破線 = 最終ショットを含む 200 局面 (いずれも 5 seed 平均)。
シミュレーション回数は 200 局面の 1 局面あたり平均 (sweep_sims.csv)。
Usage: python scripts/plot_budget_sweep.py gpw_experiment/regret500 gpw2026/paper/figures/fig_budget.png
"""
import csv, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "MS Gothic"; plt.rcParams["axes.unicode_minus"] = False

D, out = sys.argv[1], sys.argv[2]
sims = {(int(r["P"]), r["arm"]): float(r["mean_sims"]) / 1e4 for r in csv.DictReader(open(D + "/sweep_sims.csv", encoding="utf-8"))}
rows = list(csv.DictReader(open(D + "/sweep_dP.csv", encoding="utf-8")))
ARMS = [("A9P5N", "ClusterValue+N", "#d62728", "o"), ("A1N", "AllGrid+N", "#1f77b4", "s")]
SUBSETS = [("ex_r1", "最終ショットを除く168局面", "-"), ("all", "最終ショットを含む200局面", "--")]

fig, ax = plt.subplots(figsize=(3.4, 2.4))
for subset, slabel, ls in SUBSETS:
    rs = sorted((r for r in rows if r["subset"] == subset), key=lambda r: int(r["P"]))
    for arm, alabel, col, mk in ARMS:
        xs = [sims[(int(r["P"]), arm)] for r in rs]
        ys = [float(r["mean_" + arm]) for r in rs]
        ax.plot(xs, ys, ls=ls, marker=mk, ms=3.5, color=col, lw=1.0, alpha=1.0 if subset == "ex_r1" else 0.55,
                mfc=col if subset == "ex_r1" else "white", label=alabel if subset == "ex_r1" else None)
        if subset == "ex_r1" and arm == "A9P5N":
            off = {100: (3, -9), 200: (3, -9), 400: (-12, 5), 800: (-30, -2)}
            for r, x, y in zip(rs, xs, ys):
                ax.annotate(f"P={r['P']}", (x, y), textcoords="offset points", xytext=off[int(r["P"])], fontsize=6)
for slabel, ls in [(s_, l_) for _, s_, l_ in SUBSETS]:
    ax.plot([], [], ls=ls, color="0.4", lw=1.0, label=("実線：" if ls == "-" else "破線：") + slabel)
ax.set_xlim(2, 25)
ax.set_xlabel("1局面あたりのシミュレーション回数（万回）", fontsize=7)
ax.set_ylabel("regret", fontsize=7)
ax.tick_params(labelsize=6.5); ax.grid(alpha=0.3)
ax.legend(fontsize=5.8, loc="upper right", frameon=False, handlelength=2.4)
fig.tight_layout(); fig.savefig(out, dpi=300); print("saved", out)
