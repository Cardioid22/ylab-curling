# -*- coding: utf-8 -*-
"""treedepth の Δ曲線を複数run重ねて比較 (例: max-max版 vs negamax修正版)。

各runの tree_depth_curve.csv (analyze_tree_depth.py の出力) を読み、
Δ = q* − q_ref(depth-kの手) を残り手数 r 別に重ね描きする。

Usage:
  python scripts/plot_treedepth_compare.py \
    --curve "修正前(max-max)=reinvest_experiment/treedepth/analysis/tree_depth_curve.csv" \
    --curve "修正後(negamax)=reinvest_experiment/treedepth_fix/analysis/tree_depth_curve.csv" \
    --out reinvest_experiment/treedepth_fix/analysis
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "MS Gothic"; plt.rcParams["axes.unicode_minus"] = False

# Okabe-Ito (CVD-safe): 修正前=橙, 修正後=緑, 3本目以降=青/桃
COLORS = ["#D55E00", "#009E73", "#0072B2", "#CC79A7"]


def load_curve(fp):
    rows = list(csv.DictReader(open(fp, encoding="utf-8")))
    r = [int(x["remaining"]) for x in rows]
    m = [float(x["delta_mean"]) for x in rows]
    lo = [float(x["ci_lo"]) for x in rows]
    hi = [float(x["ci_hi"]) for x in rows]
    return r, m, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", action="append", required=True,
                    help='"ラベル=path/tree_depth_curve.csv" (複数指定)')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, spec in enumerate(args.curve):
        label, fp = spec.split("=", 1)
        r, m, lo, hi = load_curve(fp)
        c = COLORS[i % len(COLORS)]
        yerr = [[mm - l for mm, l in zip(m, lo)], [h - mm for mm, h in zip(m, hi)]]
        ax.errorbar(r, m, yerr=yerr, marker="o", ms=8, lw=2, capsize=4,
                    color=c, label=label)
        # 直接ラベル (最終点の右)
        ax.annotate(label, (r[-1], m[-1]), xytext=(8, 0),
                    textcoords="offset points", color=c, fontsize=10, va="center")

    ax.axhline(0, color="0.6", ls="--", lw=1)
    ax.set_xticks(sorted({x for spec in args.curve
                          for x in load_curve(spec.split("=", 1)[1])[0]}))
    ax.set_xlabel("残り手数 r")
    ax.set_ylabel("Δ = q* − q_ref(depth-k の手)  [点]")
    ax.set_title("木の深さの価値 Δ曲線の比較\n(Δ小 = depth-k の手が審判基準で良い)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = os.path.join(args.out, "tree_depth_compare.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[out] -> {p}")


if __name__ == "__main__":
    main()
