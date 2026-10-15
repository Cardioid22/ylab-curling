# -*- coding: utf-8 -*-
"""ボロノイ節の図: 行動空間 (投石の初速 vx, vy と回転) での候補の配置を、結果盤面クラスタで色分けする。
行動空間で近い候補が別の結果 (別のクラスタ) になり、離れた候補が同じ結果になる様子を見せる。
右は 468 局面での η² (価値 q が分割でどれだけまとまるか) の分布 (scripts/analyze_paths_and_spaces.py の space_rows.csv)。
Usage: python scripts/plot_action_vs_result_space.py gpw_experiment/cluster_vs_topk <game,end,shot> <out.png>
"""
import csv, glob, os, sys
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.family"] = "MS Gothic"; plt.rcParams["axes.unicode_minus"] = False
GE = ROOT + "/gpw_experiment"
D, case, out = sys.argv[1], tuple(int(x) for x in sys.argv[2].split(",")), sys.argv[3]

def key(r): return (int(r["game_id"]), int(r["end"]), int(r["shot_num"]))
d = next(f"{D}/{s}/export_g{case[0]}_e{case[1]}_s{case[2]}" for s in ("export", "export_rest")
         if os.path.exists(f"{D}/{s}/export_g{case[0]}_e{case[1]}_s{case[2]}/candidates.csv"))
cand = list(csv.DictReader(open(d + "/candidates.csv", encoding="utf-8")))
cl = {int(r["candidate_idx"]): int(r["cluster_id"]) for r in csv.DictReader(open(f"{GE}/A9P5N/seed_42/cluster_table.csv", encoding="utf-8")) if key(r) == case}
Q = {}
for f in glob.glob(GE + "/referee/score_move_qtable_idx*.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if key(r) == case: Q[int(r["candidate_idx"])] = float(r["q_ref_mean"])

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axes[0]
size = Counter(cl.values())
cids = sorted(size, key=lambda c: -max(Q[i] for i in cl if cl[i] == c))   # 最良の q が高いクラスタから色を割り当て
cmap = plt.get_cmap("tab10")
color = {c: (cmap(i) if i < 10 else "#c8c8c8") for i, c in enumerate(cids)}
for r in cand:
    i = int(r["candidate_idx"])
    ax.scatter(float(r["vx"]), float(r["vy"]), s=16, marker="o" if r["rot"] == "1" else "^",
               color=color[cl[i]], edgecolor="k", linewidth=0.4, zorder=3)
best = max(Q, key=Q.get)
rb = next(r for r in cand if int(r["candidate_idx"]) == best)
ax.scatter(float(rb["vx"]), float(rb["vy"]), s=120, facecolor="none", edgecolor="#d00", linewidth=1.2, zorder=4)
# ドロー・弱いヒットが集まる帯に拡大する (強いヒットやパスは範囲外として数だけ示す)
vys = sorted(float(r["vy"]) for r in cand)
lo, hi = vys[int(0.05 * len(vys))] - 0.04, vys[int(0.85 * len(vys))] + 0.06
n_out = sum(1 for v in vys if not lo <= v <= hi)
ax.set_ylim(lo, hi)
if n_out: ax.text(0.01, 0.98, f"範囲外 {n_out} 手 (強いヒット等)", transform=ax.transAxes, va="top", fontsize=6)
ax.set_xlabel("横方向の初速 vx [m/s]", fontsize=7); ax.set_ylabel("前方向の初速 vy [m/s]", fontsize=7)
ax.set_title(f"(a) g{case[0]} e{case[1]} s{case[2]} の候補 {len(cand)} 手 (色 = 結果盤面のクラスタ)\n"
             f"○ = 時計回り, △ = 反時計回り, 赤丸 = 最良手", fontsize=7)
ax.tick_params(labelsize=6.5); ax.grid(alpha=0.3)

ax = axes[1]
S = list(csv.DictReader(open(D + "/space_rows.csv", encoding="utf-8")))
cols = [("eta2_result", "結果盤面の\nクラスタ"), ("eta2_action_kmeans", "行動空間の\nk-means"),
        ("eta2_action_voronoi", "行動空間の\nボロノイ\n(ランダム母点)"), ("eta2_random", "ランダム\n分割")]
data = [[float(r[c]) for r in S if r[c] != "nan"] for c, _ in cols]
ax.boxplot(data, tick_labels=[l for _, l in cols], widths=0.55, showfliers=False)
ax.set_ylabel("η²", fontsize=7)
ax.set_title(f"(b) 同じ数の領域に分けたときの η² ({len(S)} 局面)", fontsize=7)
ax.tick_params(labelsize=6.5); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(out, dpi=300); print("saved", out)
