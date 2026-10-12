# -*- coding: utf-8 -*-
"""試作: クラスタあり (ClusterValue 代表4手) vs クラスタなし (e_score 上位4手) を1局面で比較する図。

結果盤面は ylab_client の --export-case で書き出したものを使う:
  ./build/Release/ylab_client.exe --clustering-experiment --load-positions test_positions500       --export-case 9633,1,10 --output-dir gpw_experiment/cluster_vs_topk/export
  (export のクラスタ数は retention 依存で実験と異なりうるので、クラスタは A9P5N の cluster_table を使う。
   candidate_idx は generatePool 順で一致する)
Usage: python scripts/plot_cluster_vs_topk_case.py <game> <end> <shot> <seed> <export_dir> <out.png>
"""
import csv, sys, os, math, glob
from collections import defaultdict, Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + "/scripts")
from plot_board import HC_X, HC_Y, R12, R8, R4, RBTN, SR, AREA_X, C0, C1
plt.rcParams["font.family"] = "MS Gothic"; plt.rcParams["axes.unicode_minus"] = False
g, e, s, seed, expdir, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
K = (g, e, s)
Q = {}
for f in glob.glob(ROOT + "/gpw_experiment/referee/score_move_qtable_idx*.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if (int(r["game_id"]), int(r["end"]), int(r["shot_num"])) == K: Q[int(r["candidate_idx"])] = float(r["q_ref_mean"])
ct = [r for r in csv.DictReader(open(f"{ROOT}/gpw_experiment/A9P5N/seed_{seed}/cluster_table.csv", encoding="utf-8"))
      if (int(r["game_id"]), int(r["end"]), int(r["shot_num"])) == K]
cl = {int(r["candidate_idx"]): int(r["cluster_id"]) for r in ct}
ee = {int(r["candidate_idx"]): float(r["e_score"]) for r in ct}
lab = {int(r["candidate_idx"]): r["label"] for r in ct}
cv = [int(r["candidate_idx"]) for r in ct if r["is_representative"] == "1"]
topk = sorted(ee, key=lambda c: (-ee[c], c))[:len(cv)]
best = max(Q, key=Q.get); qs = Q[best]
csize = Counter(cl.values())
inp = list(csv.DictReader(open(expdir + "/input_board.csv", encoding="utf-8")))
res = defaultdict(list)
for r in csv.DictReader(open(expdir + "/result_stones.csv", encoding="utf-8")): res[int(r["candidate_idx"])].append(r)
batch = [r for r in csv.DictReader(open(ROOT + "/test_positions500/batch_0001.csv", encoding="utf-8"))
         if (int(r["match_id"]), int(r["end"]), int(r["shot_num"])) == K][0]
team = int(batch["team"]); stone_idx = s // 2

def house(ax, title):
    for rr, c in [(R12, "#cfe0f3"), (R8, "#ffffff"), (R4, "#f3cfcf"), (RBTN, "#ffffff")]:
        ax.add_patch(Circle((HC_X, HC_Y), rr, facecolor=c, edgecolor="#999", lw=0.7, zorder=1))
    ax.axhline(HC_Y, color="#ccc", lw=0.5, ls="--")
    ax.set_xlim(-AREA_X, AREA_X); ax.set_ylim(HC_Y - R12 - 2.4, HC_Y + R12 + 0.6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8.5)
def stones(ax, rows, alpha=1.0, lw=0.8):
    for r in rows:
        col = C0 if int(r["team"]) == 0 else C1
        ax.add_patch(Circle((float(r["x"]), float(r["y"])), SR, facecolor=col, edgecolor="#222", lw=lw, alpha=alpha, zorder=3))
def land(c):
    for r in res[c]:
        if int(r["team"]) == team and int(r["stone_idx"]) == stone_idx: return float(r["x"]), float(r["y"])
    return None

fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6))
# (a) 候補の着地点 (クラスタ色) + 2つの選別
ax = axes[0]; house(ax, "(a) 候補の着地点とクラスタ\n★=クラスタあり(4手) □=クラスタなし(上位4手) ◎=最良")
stones(ax, inp, alpha=0.9)
cmap = plt.get_cmap("tab20")
cids = sorted(set(cl.values()), key=lambda c: -max(ee[i] for i in cl if cl[i] == c))
color = {c: cmap(i % 20) for i, c in enumerate(cids)}
for c in cl:
    p = land(c)
    if p: ax.plot(*p, "o", ms=3.5, color=color[cl[c]], alpha=0.8, zorder=4)
for c in cv:
    p = land(c)
    if p: ax.plot(*p, "*", ms=13, mfc=color[cl[c]], mec="k", mew=0.8, zorder=6)
for c in topk:
    p = land(c)
    if p: ax.plot(*p, "s", ms=10, mfc="none", mec="k", mew=1.2, zorder=6)
p = land(best)
if p: ax.plot(*p, "o", ms=16, mfc="none", mec="#d00", mew=1.8, zorder=7)
# (b)(c) 結果盤面
def result_panel(ax, c, title):
    house(ax, title)
    stones(ax, inp, alpha=0.25, lw=0.3)
    stones(ax, res[c], alpha=1.0)
    before = {(r["team"], r["stone_idx"]): (float(r["x"]), float(r["y"])) for r in inp}
    after = {(r["team"], r["stone_idx"]): (float(r["x"]), float(r["y"])) for r in res[c]}
    for k2, (x0, y0) in before.items():
        if k2 in after:
            x1, y1 = after[k2]
            if math.hypot(x1 - x0, y1 - y0) > 0.05:
                ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=8, color="k", lw=0.8, zorder=5))
        else:
            ax.plot(x0, y0, "x", color="k", ms=8, mew=1.5, zorder=5)
best_tk = max(topk, key=lambda c: Q.get(c, -9))
result_panel(axes[1], best, f"(b) 最良手 {lab[best]}\n審判値 {qs:+.2f}  クラスタの大きさ {csize[cl[best]]}  推定順位 {sorted(ee, key=lambda c:-ee[c]).index(best)+1}")
result_panel(axes[2], best_tk, f"(c) クラスタなしの4手で最良 {lab[best_tk]}\n審判値 {Q[best_tk]:+.2f} (差 {qs-Q[best_tk]:.2f})")
fig.suptitle(f"g{g} e{e} s{s} (seed {seed}, N={len(cl)}, クラスタ数 {len(csize)})  手番={'team0(橙)' if team==0 else 'team1(青)'}", fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=130); print("saved", out)
print("CV :", [(lab[c], round(Q[c],2), csize[cl[c]]) for c in cv])
print("TOP:", [(lab[c], round(Q[c],2), csize[cl[c]]) for c in topk])
