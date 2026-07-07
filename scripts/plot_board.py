# -*- coding: utf-8 -*-
"""カーリングの氷上(ハウス+石)を描く。batch_*.csv の1局面を再現。
単体テスト: python scripts/plot_board.py <batch.csv> [game_id] [end] [shot_num]
"""
import csv, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["axes.unicode_minus"] = False

HC_X, HC_Y = 0.0, 38.405       # ハウス中心
R12, R8, R4, RBTN = 1.829, 1.219, 0.610, 0.152   # 4リング半径 [m]
SR = 0.145                      # 石半径
AREA_X = 2.375                  # シート半幅
# チーム色 (色覚安全)
C0, C1 = "#D55E00", "#0072B2"   # team0 / team1

def draw_board(ax, row, title=""):
    """row: batch CSV の1行 (dict)。ハウス+石を描く。"""
    # ハウス4リング (外→内)
    for r, c in [(R12, "#4a90d9"), (R8, "#ffffff"), (R4, "#d94a4a"), (RBTN, "#ffffff")]:
        ax.add_patch(Circle((HC_X, HC_Y), r, facecolor=c, edgecolor="#888", lw=0.8, zorder=1))
    # シート境界(サイドライン) と ティーライン
    ax.axvline(-AREA_X, color="#bbb", lw=1); ax.axvline(AREA_X, color="#bbb", lw=1)
    ax.axhline(HC_Y, color="#bbb", lw=0.6, ls="--")   # ティーライン
    # 石
    for t, col in [(0, C0), (1, C1)]:
        for i in range(8):
            if int(row[f"t{t}s{i}_inplay"]) != 1: continue
            x = float(row[f"t{t}s{i}_x"]); y = float(row[f"t{t}s{i}_y"])
            ax.add_patch(Circle((x, y), SR, facecolor=col, edgecolor="#222", lw=0.8, zorder=3))
    ax.set_xlim(-AREA_X-0.1, AREA_X+0.1)
    ax.set_ylim(HC_Y - R12 - 1.2, HC_Y + R12 + 1.2)   # ハウス周辺にズーム
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=10)

def main():
    fp = sys.argv[1]
    rows = list(csv.DictReader(open(fp, newline="", encoding="utf-8")))
    if len(sys.argv) >= 3:
        rows = [r for r in rows if r["match_id"] == sys.argv[2]]
    if len(sys.argv) >= 5:
        rows = [r for r in rows if r["end"] == sys.argv[3] and r["shot_num"] == sys.argv[4]]
    rows = rows[:1]
    for r in rows:
        fig, ax = plt.subplots(figsize=(3.2, 5))
        n = sum(int(r[f"t{t}s{i}_inplay"]) for t in (0,1) for i in range(8))
        draw_board(ax, r, f"g{r['match_id']} e{r['end']} s{r['shot_num']} (石{n})")
        # 凡例
        ax.scatter([], [], c=C0, s=60, label="team0"); ax.scatter([], [], c=C1, s=60, label="team1")
        ax.legend(loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5,-0.06))
        out = f"board_g{r['match_id']}_e{r['end']}_s{r['shot_num']}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print("saved", out)

if __name__ == "__main__":
    main()
