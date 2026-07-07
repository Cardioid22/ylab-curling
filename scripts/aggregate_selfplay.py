# -*- coding: utf-8 -*-
"""自己対戦(selfplay)の複数マシン結果を統合。勝率+Wilson CI+二項検定+平均純得点、net分布図。
Usage: python scripts/aggregate_selfplay.py --games "reinvest_experiment/selfplay/*/selfplay_games.csv" --out DIR
"""
import argparse, csv, glob, math, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "MS Gothic"; plt.rcParams["axes.unicode_minus"] = False

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c - h, c + h)

def boot_ci(x, n=10000, seed=9):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    return tuple(np.percentile(x[idx].mean(1), [2.5, 97.5]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="selfplay_games.csv の glob")
    ap.add_argument("--label-a", default="A7(ScoreScreen)")
    ap.add_argument("--label-b", default="A1(AllGrid)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(args.games))
    nets = []; per_src = []; src_nets = {}
    for fp in files:
        vals = [float(r["net_a"]) for r in csv.DictReader(open(fp, encoding="utf-8"))]
        nets += vals
        src = os.path.basename(os.path.dirname(fp)).replace("A7vsA1_", "")
        src_nets[src] = np.array(vals)
        aw = sum(1 for v in vals if v > 0); bw = sum(1 for v in vals if v < 0)
        per_src.append((src, len(vals), aw, bw, np.mean(vals) if vals else 0))
    nets = np.array(nets)
    n = len(nets)
    a_win = int((nets > 0).sum()); b_win = int((nets < 0).sum()); tie = int((nets == 0).sum())
    dec = a_win + b_win
    wr = a_win / dec if dec else 0.5
    wl, wh = wilson(a_win, dec)
    bt = stats.binomtest(a_win, dec, 0.5).pvalue if dec else 1.0
    mnet = nets.mean(); nlo, nhi = boot_ci(nets)

    out = []
    out.append("=" * 60)
    out.append(f"自己対戦 統合結果  {args.label_a} vs {args.label_b}")
    out.append("=" * 60)
    out.append(f"総ゲーム数: {n}  (ファイル {len(files)}個)")
    out.append("--- マシン別 ---")
    for src, ng, aw, bw, mn in per_src:
        out.append(f"  {src:22} n={ng:>3}  {args.label_a[:2]}勝={aw:>3} {args.label_b[:2]}勝={bw:>3}  "
                   f"勝率={aw/(aw+bw):.3f}  平均net={mn:+.2f}")
    out.append("--- 統合 ---")
    out.append(f"  {args.label_a} 勝 / {args.label_b} 勝 / 引分 = {a_win} / {b_win} / {tie}")
    out.append(f"  勝率({args.label_a}, 引分除外) = {wr:.3f}   95%CI [{wl:.3f}, {wh:.3f}] (Wilson)")
    out.append(f"  二項検定 p (両側, 帰無=0.5) = {bt:.3g}")
    out.append(f"  平均純得点({args.label_a}視点) = {mnet:+.3f} 点/エンド   95%CI [{nlo:+.3f}, {nhi:+.3f}]")
    report = "\n".join(out)
    print(report)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "selfplay_aggregate.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")
        # net分布ヒストグラム
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        lo, hi = int(nets.min()), int(nets.max())
        bins = np.arange(lo - 0.5, hi + 1.5, 1)
        colors = ["#0072B2" if b < 0 else ("#999999" if b == 0 else "#009E73")
                  for b in range(lo, hi + 1)]
        counts, edges = np.histogram(nets, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, counts, width=0.9, color=colors, edgecolor="white")
        ax.axvline(0, color="0.4", ls="--", lw=1)
        ax.axvline(mnet, color="#009E73", lw=2, label=f"平均 {mnet:+.2f}")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.set_xlabel(f"エンド純得点 ({args.label_a} 視点)  右=A7得点 / 左=A1得点")
        ax.set_ylabel("ゲーム数")
        ax.set_title(f"{args.label_a} vs {args.label_b}  (n={n}, 勝率={wr:.1%}, p={bt:.1e})")
        ax.legend(frameon=False)
        fig.tight_layout()
        p = os.path.join(args.out, "selfplay_net_hist.png"); fig.savefig(p, dpi=150); plt.close(fig)

        # 勝率 + 平均純得点 (マシン別 + 統合)
        srcs = list(src_nets.keys())
        cats = srcs + ["統合"]
        arrs = [src_nets[s] for s in srcs] + [nets]
        fig2, (axw, axn) = plt.subplots(1, 2, figsize=(10, 4.6))
        GREEN = "#009E73"
        for i, x in enumerate(arrs):
            aw = int((x > 0).sum()); bw = int((x < 0).sum()); d = aw + bw
            wr = aw / d if d else 0.5; lo, hi = wilson(aw, d)
            comb = (i == len(arrs) - 1)
            axw.errorbar(i, wr, yerr=[[wr-lo], [hi-wr]], marker="o", ms=11 if comb else 8,
                         color=GREEN, capsize=5, lw=2, mec="black" if comb else "none")
            axw.text(i, hi+0.015, f"{wr:.0%}", ha="center", color="#222", fontsize=10)
        axw.axhline(0.5, color="0.5", ls="--", lw=1, label="五分五分 (0.5)")
        axw.set_ylim(0.4, 1.0); axw.set_xticks(range(len(cats))); axw.set_xticklabels(cats)
        axw.set_ylabel("A7 の勝率 (引分除外, 95%CI)"); axw.set_title("勝率")
        for s in ("top","right"): axw.spines[s].set_visible(False)
        axw.grid(axis="y", alpha=0.3); axw.legend(frameon=False, fontsize=9)
        for i, x in enumerate(arrs):
            m = x.mean(); lo, hi = boot_ci(x); comb = (i == len(arrs)-1)
            axn.errorbar(i, m, yerr=[[m-lo], [hi-m]], marker="o", ms=11 if comb else 8,
                         color=GREEN, capsize=5, lw=2, mec="black" if comb else "none")
            axn.text(i, hi+0.03, f"{m:+.2f}", ha="center", color="#222", fontsize=10)
        axn.axhline(0, color="0.5", ls="--", lw=1, label="互角 (0点)")
        axn.set_xticks(range(len(cats))); axn.set_xticklabels(cats)
        axn.set_ylabel("平均純得点 (A7視点, 点/エンド, 95%CI)"); axn.set_title("平均純得点")
        for s in ("top","right"): axn.spines[s].set_visible(False)
        axn.grid(axis="y", alpha=0.3); axn.legend(frameon=False, fontsize=9)
        fig2.suptitle(f"A7(ScoreScreen) vs A1(AllGrid) 自己対戦  P=50等予算  (計{n}ゲーム)", fontsize=13)
        fig2.tight_layout(rect=[0,0,1,0.95])
        p2 = os.path.join(args.out, "selfplay_winrate.png"); fig2.savefig(p2, dpi=150); plt.close(fig2)
        print(f"\n[out] -> {args.out}/selfplay_aggregate.txt, selfplay_net_hist.png, selfplay_winrate.png")

if __name__ == "__main__":
    main()
