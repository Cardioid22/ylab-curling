# -*- coding: utf-8 -*-
"""2エンド自己対戦の統合分析。勝率(2エンド計)+先攻後攻(ハンマー)効果+エンド別得点。

selfplay_games.csv の列: game,a_hammer_end0,net_end0,net_end1,net_a
  a_hammer_end0 = 1: end0 で A が後攻(ハンマー) / 0: A が先攻
  net_endX      = そのエンドの純得点 (A視点, A得点 - B得点)
  net_a         = 2エンド計の純得点 (A視点)

分析の狙い (ユーザ要望「先攻後攻が入れ替わった場合の得点差」):
  ① 全体: 2エンド計の勝率 + 平均純得点 (バイアスフリーな強さ)
  ② ハンマー(先攻後攻)効果: A の end0 得点を「A後攻 vs A先攻」で比較。
     Aの強さは一定なので、差 = ハンマー(後攻)の得点アドバンテージ。
  ③ エンド別: end0/end1 の平均得点。end1 はハンマーが入れ替わるので、
     ハンマー保持側視点の得点も出す (構造的ハンマー有利の大きさ)。

Usage:
  python scripts/aggregate_selfplay_2end.py \
    --games "experiments/selfplay2end/A7vsA1_*/selfplay_games.csv" \
    --label-a "A7(ScoreScreen)" --label-b "A1(AllGrid)" \
    --out experiments/selfplay2end/A7vsA1_agg
"""
import argparse, csv, glob, math, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "MS Gothic"; plt.rcParams["axes.unicode_minus"] = False

GREEN = "#009E73"   # A7
BLUE  = "#0072B2"   # A後攻(ハンマー) / end0
ORANGE = "#D55E00"  # A先攻 / end1
GRAY  = "#999999"


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c - h, c + h)


def boot_ci(x, n=10000, seed=7):
    x = np.asarray(x, float)
    if len(x) == 0: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    return tuple(np.percentile(x[idx].mean(1), [2.5, 97.5]))


def load(glob_pat):
    files = sorted(glob.glob(glob_pat))
    rows = []
    for fp in files:
        src = os.path.basename(os.path.dirname(fp))
        for r in csv.DictReader(open(fp, encoding="utf-8")):
            r["_src"] = src
            rows.append(r)
    return files, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="selfplay_games.csv の glob")
    ap.add_argument("--label-a", default="A7(ScoreScreen)")
    ap.add_argument("--label-b", default="A1(AllGrid)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files, rows = load(args.games)
    if not rows:
        print(f"No rows matched: {args.games}"); return
    has_end1 = "net_end1" in rows[0]

    net_a  = np.array([float(r["net_a"]) for r in rows])
    e0     = np.array([float(r["net_end0"]) for r in rows])
    e1     = np.array([float(r["net_end1"]) for r in rows]) if has_end1 else None
    aham0  = np.array([int(float(r["a_hammer_end0"])) for r in rows])
    src    = np.array([r["_src"] for r in rows])
    n = len(rows)

    LA, LB = args.label_a, args.label_b
    out = []
    def pr(s): out.append(s); print(s)

    pr("=" * 66)
    pr(f"2エンド自己対戦 統合  {LA} vs {LB}")
    pr("=" * 66)
    pr(f"総ゲーム数: {n}  (ファイル {len(files)}個)")

    # マシン別内訳 (net_a 総得点)
    machines = sorted(set(src))
    if len(machines) > 1:
        pr("--- マシン別 (2エンド計) ---")
        for m in machines:
            x = net_a[src == m]
            aw = int((x > 0).sum()); bw = int((x < 0).sum()); tie = int((x == 0).sum())
            wr = aw / (aw + bw) if (aw + bw) else 0.5
            pr(f"  {m:18} n={len(x):>3}  A7勝={aw:>2} A1勝={bw:>2} 分={tie:>2}  "
               f"勝率={wr:.3f}  平均net={x.mean():+.3f}")

    # ① 全体 (2エンド計)
    aw = int((net_a > 0).sum()); bw = int((net_a < 0).sum()); tie = int((net_a == 0).sum())
    dec = aw + bw
    wr = aw / dec if dec else 0.5
    wl, wh = wilson(aw, dec)
    bt = stats.binomtest(aw, dec, 0.5).pvalue if dec else 1.0
    mlo, mhi = boot_ci(net_a)
    pr("\n--- ① 全体 (2エンド計) ---")
    pr(f"  {LA}勝 / {LB}勝 / 引分 = {aw} / {bw} / {tie}")
    pr(f"  勝率({LA}, 引分除外) = {wr:.3f}  95%CI[{wl:.3f},{wh:.3f}]  二項p={bt:.3g}")
    pr(f"  平均純得点({LA}視点, 2エンド計) = {net_a.mean():+.3f}  95%CI[{mlo:+.3f},{mhi:+.3f}]")

    # ② ハンマー(先攻後攻)効果  end0 を A後攻 vs A先攻 で比較
    ham = e0[aham0 == 1]   # A が後攻(ハンマー)
    non = e0[aham0 == 0]   # A が先攻
    hlo, hhi = boot_ci(ham); nlo, nhi = boot_ci(non)
    diff = (ham.mean() if len(ham) else 0) - (non.mean() if len(non) else 0)
    tt = stats.ttest_ind(ham, non, equal_var=False) if len(ham) and len(non) else None
    pr("\n--- ② ハンマー(先攻後攻)効果  [end0, A視点] ---")
    pr(f"  A後攻(ハンマー) n={len(ham):>3}  平均end0={ham.mean():+.3f}  95%CI[{hlo:+.3f},{hhi:+.3f}]")
    pr(f"  A先攻          n={len(non):>3}  平均end0={non.mean():+.3f}  95%CI[{nlo:+.3f},{nhi:+.3f}]")
    pr(f"  後攻アドバンテージ (後攻-先攻) = {diff:+.3f} 点" + (f"  (Welch t p={tt.pvalue:.3g})" if tt else ""))
    # 構造的ハンマー有利: end0 でハンマーを持つ側(A/B問わず)の平均純得点
    hammer_net_e0 = np.where(aham0 == 1, e0, -e0)   # ハンマー保持側視点
    pr(f"  参考: end0 でハンマー保持側の平均純得点 = {hammer_net_e0.mean():+.3f} 点 (A/B混在=純粋な後攻有利)")

    # ③ エンド別
    if has_end1:
        e0lo, e0hi = boot_ci(e0); e1lo, e1hi = boot_ci(e1)
        pr("\n--- ③ エンド別 (A視点) ---")
        pr(f"  end0 平均 = {e0.mean():+.3f}  95%CI[{e0lo:+.3f},{e0hi:+.3f}]")
        pr(f"  end1 平均 = {e1.mean():+.3f}  95%CI[{e1lo:+.3f},{e1hi:+.3f}]   (end1はハンマーが入替)")

    report = "\n".join(out)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "selfplay2end_aggregate.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")

        # 図: (左)全体勝率&平均純得点  (中)ハンマー効果  (右)エンド別
        ncol = 3 if has_end1 else 2
        fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.8))

        # (左) 全体
        ax = axes[0]
        ax.bar([0], [wr], color=GREEN, width=0.5, edgecolor="white")
        ax.errorbar([0], [wr], yerr=[[wr - wl], [wh - wr]], fmt="none", ecolor="0.2", capsize=6, lw=2)
        ax.axhline(0.5, color="0.5", ls="--", lw=1)
        ax.text(0, min(wh + 0.03, 0.98), f"{wr:.0%}", ha="center", fontsize=13, color="#222")
        ax.set_ylim(0, 1.05); ax.set_xlim(-0.7, 0.7); ax.set_xticks([0]); ax.set_xticklabels([f"{LA}\n勝率"])
        ax.set_ylabel("勝率 (引分除外, 95%CI)")
        ax.set_title(f"① 全体  n={n}, p={bt:.1e}")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        # (中) ハンマー効果
        ax = axes[1]
        xs = [0, 1]
        means = [non.mean() if len(non) else 0, ham.mean() if len(ham) else 0]
        errs = [[means[0] - nlo, means[1] - hlo], [nhi - means[0], hhi - means[1]]]
        ax.bar(xs, means, color=[ORANGE, BLUE], width=0.55, edgecolor="white")
        ax.errorbar(xs, means, yerr=errs, fmt="none", ecolor="0.2", capsize=6, lw=2)
        ax.axhline(0, color="0.4", lw=1)
        for x, m in zip(xs, means):
            ax.text(x, m + (0.05 if m >= 0 else -0.12), f"{m:+.2f}", ha="center", fontsize=11, color="#222")
        ax.set_xticks(xs); ax.set_xticklabels([f"{LA}先攻", f"{LA}後攻\n(ハンマー)"])
        ax.set_ylabel(f"end0 平均純得点 ({LA}視点)")
        ax.set_title(f"② 先攻後攻効果\n後攻有利 = {diff:+.2f}点")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        # (右) エンド別
        if has_end1:
            ax = axes[2]
            xs = [0, 1]
            means = [e0.mean(), e1.mean()]
            errs = [[means[0] - e0lo, means[1] - e1lo], [e0hi - means[0], e1hi - means[1]]]
            ax.bar(xs, means, color=[BLUE, ORANGE], width=0.55, edgecolor="white")
            ax.errorbar(xs, means, yerr=errs, fmt="none", ecolor="0.2", capsize=6, lw=2)
            ax.axhline(0, color="0.4", lw=1)
            for x, m in zip(xs, means):
                ax.text(x, m + (0.05 if m >= 0 else -0.12), f"{m:+.2f}", ha="center", fontsize=11, color="#222")
            ax.set_xticks(xs); ax.set_xticklabels(["end0", "end1\n(ハンマー入替)"])
            ax.set_ylabel(f"平均純得点 ({LA}視点)")
            ax.set_title("③ エンド別")
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"{LA} vs {LB}  2エンド自己対戦  (計{n}ゲーム)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        p = os.path.join(args.out, "selfplay2end_summary.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"\n[out] -> {args.out}/selfplay2end_aggregate.txt, selfplay2end_summary.png")


if __name__ == "__main__":
    main()
