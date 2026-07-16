# -*- coding: utf-8 -*-
"""エリア得点期待値ヒートマップ (A9/A10 の cluster_table から)。

各候補手の無外乱着地点 (land_x, land_y) の E[score] (R_pre推定, 手番視点) を
IDW (逆距離加重) でシート上に空間補間し、「どのエリアに投げると何点見込めるか」を
連続ヒートマップで示す。採用クラスタの重心には「C{id}: +X.X」のエリアラベルを置く。

- 配色: 発散 (緑=プラス / 紫=マイナス, 0=白)。石のチーム色 (橙/青) と混同しない。
- 補間はサンプル (候補着地) から 0.7m 以上離れたセルをマスク (データの無い場所を塗らない)。
- 着地なし (テイクアウト系でアウト) の候補は空間に置けないため件数のみ注記。
- 値の意味: そのエンド単体の純得点期待値 (手番チーム視点)。R_pre=3 の粗い推定なので
  ラベル (クラスタ平均) の方が頑健。ヒートマップは傾向の可視化。

Usage:
  python scripts/plot_area_heatmap.py \
    --cluster-table .../cluster_table.csv --batch-csv .../batch_0001.csv \
    --out <DIR> [--positions g123:e5:s10,...]
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_board import HC_X, HC_Y, R12, R8, R4, RBTN, SR, AREA_X, C0, C1
plt.rcParams["font.family"] = "MS Gothic"; plt.rcParams["axes.unicode_minus"] = False

CMAP = "PRGn"          # 発散: 紫(負) - 白(0) - 緑(正)。CVD-safe
MASK_DIST = 0.7        # 最寄り着地点からこの距離[m]を超えるセルは塗らない
Y_LO, Y_HI = HC_Y - R12 - 2.6, HC_Y + R12 + 0.9   # ガードゾーン込みの描画範囲


def fnum(s):
    try:
        return float(s) if s not in ("", None) else float("nan")
    except ValueError:
        return float("nan")


def load_cluster_table(fp):
    per_pos = defaultdict(list)
    for r in csv.DictReader(open(fp, encoding="utf-8")):
        per_pos[(r["game_id"], r["end"], r["shot_num"])].append(r)
    return per_pos


def draw_board_outline(ax, row):
    """リングは輪郭のみ (下のヒートマップを見せる)。石は通常描画。"""
    for r in (R12, R8, R4, RBTN):
        ax.add_patch(Circle((HC_X, HC_Y), r, facecolor="none", edgecolor="#555",
                            lw=1.0, zorder=2))
    ax.axvline(-AREA_X, color="#bbb", lw=1); ax.axvline(AREA_X, color="#bbb", lw=1)
    ax.axhline(HC_Y, color="#999", lw=0.6, ls="--", zorder=2)
    if row is not None:
        for t, col in [(0, C0), (1, C1)]:
            for i in range(8):
                if int(row[f"t{t}s{i}_inplay"]) != 1: continue
                x = float(row[f"t{t}s{i}_x"]); y = float(row[f"t{t}s{i}_y"])
                ax.add_patch(Circle((x, y), SR, facecolor=col, edgecolor="#111",
                                    lw=0.9, zorder=4))
    ax.set_xlim(-AREA_X - 0.1, AREA_X + 0.1)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def idw_field(px, py, vals):
    """着地点サンプルの IDW 補間。データから遠いセルは NaN。"""
    gx = np.linspace(-AREA_X, AREA_X, 150)
    gy = np.linspace(Y_LO, Y_HI, 240)
    X, Y = np.meshgrid(gx, gy)
    d2 = (X[..., None] - px) ** 2 + (Y[..., None] - py) ** 2   # (H,W,n)
    w = 1.0 / np.maximum(d2, 1e-3)
    Z = (w * vals).sum(-1) / w.sum(-1)
    Z[np.sqrt(d2.min(-1)) > MASK_DIST] = np.nan
    return gx, gy, Z


def plot_one(ax, rows, board_row, title):
    # 着地点 + E[score] (手番視点)
    px, py, vals = [], [], []
    n_out = 0
    for r in rows:
        x, y, e = fnum(r["land_x"]), fnum(r["land_y"]), fnum(r["e_score"])
        if np.isnan(e):
            continue
        if np.isnan(x) or np.isnan(y):
            n_out += 1; continue
        px.append(x); py.append(y); vals.append(e)
    draw_board_outline(ax, board_row)
    ax.set_title(title, fontsize=10)
    if len(px) < 5:
        # 最終ショット(s15)は着手で次エンドに遷移し盤面がリセットされるため着地座標が残らない
        ax.text(0.5, 0.5, "着地サンプル不足\n(最終ショット局面は着地記録不可)",
                transform=ax.transAxes, ha="center", fontsize=9, color="#555")
        return None
    px, py, vals = np.array(px), np.array(py), np.array(vals)

    gx, gy, Z = idw_field(px, py, vals)
    vmax = max(0.5, float(np.nanmax(np.abs(Z))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)   # 対称スケール (0=白)
    im = ax.imshow(Z, origin="lower", extent=(gx[0], gx[-1], gy[0], gy[-1]),
                   cmap=CMAP, norm=norm, alpha=0.9, zorder=1, interpolation="bilinear")

    # 採用クラスタ: 代表手の星 + 重心に「C{id}: +X.X」ラベル
    clusters = defaultdict(list)
    for r in rows:
        clusters[int(r["cluster_id"])].append(r)
    adopted = [(cid, ms) for cid, ms in clusters.items()
               if any(m["is_representative"] == "1" for m in ms)]
    adopted.sort(key=lambda t: -fnum(t[1][0]["cluster_value"]))
    for rank, (cid, ms) in enumerate(adopted):
        xs = [fnum(m["land_x"]) for m in ms]; ys = [fnum(m["land_y"]) for m in ms]
        xs = [v for v in xs if not np.isnan(v)]; ys = [v for v in ys if not np.isnan(v)]
        cv = fnum(ms[0]["cluster_value"])
        rep = next(m for m in ms if m["is_representative"] == "1")
        rx, ry = fnum(rep["land_x"]), fnum(rep["land_y"])
        if not np.isnan(rx):
            ax.scatter(rx, ry, s=150, marker="*", c="white", edgecolors="black",
                       linewidths=1.2, zorder=5)
        if xs:
            # ラベルは価値順に上方向へ段差配置 (重心が近いクラスタ同士の重なり回避)
            cx, cy = np.mean(xs), np.mean(ys)
            ax.annotate(f"C{cid}: {cv:+.1f}", (cx, cy), xytext=(0, 8 + 14 * rank),
                        textcoords="offset points", ha="center", fontsize=9,
                        color="#111", zorder=6,
                        bbox=dict(fc="white", ec="#555", alpha=0.9, pad=1.5))
    if n_out:
        ax.text(0.02, 0.02, f"着地なし {n_out}件 (テイクアウト系)", transform=ax.transAxes,
                fontsize=7, va="bottom", bbox=dict(fc="white", ec="0.7", alpha=0.85))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-table", required=True)
    ap.add_argument("--batch-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", default=None, help="gID:eE:sS をカンマ区切り。省略時は先頭6局面")
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
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 6.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    last_im = None
    for ax, k in zip(axes, keys):
        g, e, s = k
        row = board.get(k)
        team = int(row["team"]) if (row and row.get("team") not in (None, "")) else int(s) % 2
        mover = "橙(team0)" if team == 0 else "青(team1)"
        im = plot_one(ax, per_pos.get(k, []), row,
                      f"g{g} e{e} s{s}  手番={mover}\n★=採用クラスタの代表手")
        if im is not None:
            cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
            cb.ax.tick_params(labelsize=7)
            last_im = im
    method = next(iter(per_pos.values()))[0].get("method", "ClusterValue") if per_pos else ""
    fig.suptitle(f"エリア得点期待値ヒートマップ ({method}): 緑=手番有利 / 紫=不利 (そのエンド単体, 手番視点)\n"
                 f"ラベル=採用クラスタの平均E[score] (「このエリアは約X点」)。色はR_pre推定のIDW補間, データ無し領域は無色",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(args.out, "area_heatmap.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[out] -> {p}  ({n} positions)")


if __name__ == "__main__":
    main()
