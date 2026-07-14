# -*- coding: utf-8 -*-
"""ClusterValue (A9) のエリア価値マップ可視化。

cluster_table.csv (method=ClusterValue) の各候補の無外乱着地 (land_x, land_y) を
盤面上にプロットし、クラスタ別に色分け + クラスタ価値 (平均E[score]) を注記する。
「この盤面ではどのエリアに投げると後々有利か」の逆射影図。

- 価値上位クラスタ (is_representative を含むクラスタ = MCTS子に採用) を Okabe-Ito 色で強調
- 非採用クラスタのメンバーは灰色
- 代表手 (クラスタ内 E[score] 最大) は星マーカー
- 着地なし (テイクアウト系でアウト) は件数のみ注記

Usage:
  python scripts/plot_cluster_value_map.py \
    --cluster-table reinvest_experiment/scorescreen/run50_A9/seed_42/cluster_table.csv \
    --batch-csv test_positions50/batch_0001.csv \
    --out reinvest_experiment/scorescreen/run50_A9/value_maps \
    [--positions g123:e5:s10,g456:e2:s8]   # 省略時は先頭6局面
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_board import draw_board
plt.rcParams["font.family"] = "MS Gothic"; plt.rcParams["axes.unicode_minus"] = False

# 採用クラスタ用 (Okabe-Ito, 固定順)
CLUSTER_COLORS = ["#009E73", "#0072B2", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GRAY = "#BBBBBB"


def load_cluster_table(fp):
    per_pos = defaultdict(list)
    for r in csv.DictReader(open(fp, encoding="utf-8")):
        per_pos[(r["game_id"], r["end"], r["shot_num"])].append(r)
    return per_pos


def fnum(s):
    try:
        return float(s) if s not in ("", None) else float("nan")
    except ValueError:
        return float("nan")


def plot_one(ax, rows, board_row, title):
    if board_row is not None:
        draw_board(ax, board_row, title)
    else:
        ax.set_title(title)

    # クラスタごとに整理
    clusters = defaultdict(list)
    for r in rows:
        clusters[int(r["cluster_id"])].append(r)
    # 採用クラスタ = 代表手を含むクラスタ。価値降順で色を割当
    selected = [cid for cid, ms in clusters.items()
                if any(m["is_representative"] == "1" for m in ms)]
    selected.sort(key=lambda cid: -fnum(clusters[cid][0]["cluster_value"]))
    color_of = {cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cid in enumerate(selected)}

    n_out = 0
    for cid, ms in clusters.items():
        col = color_of.get(cid, GRAY)
        sel = cid in color_of
        for m in ms:
            x, y = fnum(m["land_x"]), fnum(m["land_y"])
            if np.isnan(x) or np.isnan(y):
                n_out += 1
                continue
            rep = (m["is_representative"] == "1")
            ax.scatter(x, y, s=170 if rep else 55,
                       marker="*" if rep else "o",
                       c=col, edgecolors="black" if rep else "white",
                       linewidths=1.2 if rep else 0.5,
                       zorder=6 if rep else (5 if sel else 4),
                       alpha=1.0 if sel else 0.55)
    # クラスタ価値の注記 (採用クラスタのみ、上から)
    lines = [f"C{cid}: {fnum(clusters[cid][0]['cluster_value']):+.2f}" for cid in selected]
    if n_out:
        lines.append(f"着地なし {n_out}件")
    ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, va="bottom", ha="left",
            bbox=dict(fc="white", ec="0.7", alpha=0.85))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-table", required=True)
    ap.add_argument("--batch-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", default=None,
                    help="gID:eE:sS をカンマ区切り。省略時は先頭6局面")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    per_pos = load_cluster_table(args.cluster_table)
    board = {(r["match_id"], r["end"], r["shot_num"]): r
             for r in csv.DictReader(open(args.batch_csv, newline="", encoding="utf-8"))}

    if args.positions:
        keys = []
        for tok in args.positions.split(","):
            g, e, s = tok.split(":")
            keys.append((g.lstrip("g"), e.lstrip("e"), s.lstrip("s")))
    else:
        keys = sorted(per_pos.keys())[:6]

    n = len(keys)
    ncol = min(3, n); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 5.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, k in zip(axes, keys):
        g, e, s = k
        # 手番チーム: batch CSV の team 列 (0=橙/team0, 1=青/team1)。無ければ shot 偶奇で推定
        row = board.get(k)
        team = int(row["team"]) if (row and row.get("team") not in (None, "")) else int(s) % 2
        mover = "橙(team0)" if team == 0 else "青(team1)"
        plot_one(ax, per_pos.get(k, []), row,
                 f"g{g} e{e} s{s}  手番={mover}\n★=採用クラスタの代表手(E[score]最大)")
    # 手法名は cluster_table の method 列から (A9=ClusterValue / A10=ClusterValueDeep)
    method = next(iter(per_pos.values()))[0].get("method", "ClusterValue") if per_pos else "ClusterValue"
    fig.suptitle(f"エリア価値マップ ({method}): マーカー=手番チームの候補手の着地点 (色=クラスタ, 灰=不採用)\n"
                 f"注記の値=そのエンド単体の純得点期待値 (手番チーム視点)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(args.out, "cluster_value_map.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[out] -> {p}  ({n} positions)")


if __name__ == "__main__":
    main()
