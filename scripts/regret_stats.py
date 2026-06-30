#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regret の統計的比較 (論文グレードの評価)
================================================================================
方針 (ML標準作法 Demšar 2006「複数データセット上の手法比較」に準拠):
  - 独立単位 = 局面 (game_id, end, shot_num)。「局面」=「データセット」に対応。
  - 同一局面の複数seedは独立でないので、seed方向は平均して局面あたり1標本に集約。
    → 検定の検出力は「局面数 n」で決まる (seedは局面ごとの推定を精密化するだけ)。
  - 多手法 (>=3): Friedman 検定 (局面=ブロック) → 事後 pairwise Wilcoxon符号順位 (Holm補正)。
  - 2手法: Wilcoxon符号順位 + ブートストラップ95%CI + 局面単位の勝敗。
  - 効果量 (対応のある Cohen's d) と、d を検出するのに必要な概算 n も表示。

入力: aggregate_reinvest.py の reinvest_joined.csv
      (列: game_id,end,shot_num,seed,arm,regret,q_ref_mean,...)

使用例:
  python scripts/regret_stats.py \
      --joined reinvest_experiment/scorescreen/run1/regret/reinvest_joined.csv \
      --out    reinvest_experiment/scorescreen/run1/regret
================================================================================
"""
import argparse
import csv
import itertools
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats


def holm_correction(pairs_pvals):
    """[(key, p)] -> {key: p_adjusted} (Holm-Bonferroni)。"""
    items = sorted(pairs_pvals, key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    prev = 0.0
    for i, (key, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        a = max(a, prev)  # 単調性を強制
        adj[key] = a
        prev = a
    return adj


def bootstrap_ci(diffs, n_boot, stat=np.mean, alpha=0.05, seed=12345):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = stat(diffs[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def paired_cohens_d(diffs):
    diffs = np.asarray(diffs, dtype=float)
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else float("nan")


def n_for_power(d, power=0.8, alpha=0.05):
    """対応のあるt検定で効果量dを検出する概算n (両側)。"""
    if not d or math.isnan(d) or d == 0:
        return float("inf")
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    return ((za + zb) / abs(d)) ** 2 + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", required=True, help="reinvest_joined.csv")
    ap.add_argument("--out", default=None, help="出力ディレクトリ")
    ap.add_argument("--metric", default="regret", help="比較列 (default regret)")
    ap.add_argument("--higher-better", action="store_true",
                    help="指標が大きいほど良い場合に指定 (regretは指定しない)")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    lower_better = not args.higher_better

    rows = list(csv.DictReader(open(args.joined, newline="", encoding="utf-8")))
    by = defaultdict(lambda: defaultdict(list))  # arm -> pos -> [seed vals]
    arms_seen = []
    for r in rows:
        try:
            v = float(r[args.metric])
        except (KeyError, ValueError):
            continue
        arm = r["arm"]
        pos = (r["game_id"], r["end"], r["shot_num"])
        by[arm][pos].append(v)
        if arm not in arms_seen:
            arms_seen.append(arm)
    arms = sorted(arms_seen)
    if len(arms) < 2:
        print("[error] 2手法以上必要です")
        return

    # 全アームに共通する局面のみ (完全ブロック) を使う
    common = sorted(set.intersection(*[set(by[a].keys()) for a in arms]))
    n = len(common)
    if n == 0:
        print("[error] 全アーム共通の局面がありません")
        return

    # per-arm per-position: seed平均
    M = {a: np.array([np.mean(by[a][p]) for p in common]) for a in arms}
    n_seeds = {a: int(np.median([len(by[a][p]) for p in common])) for a in arms}

    out = []
    out.append("=" * 74)
    out.append(f"regret 統計評価   metric={args.metric}  "
               f"({'低いほど良い' if lower_better else '高いほど良い'})")
    out.append("=" * 74)
    out.append(f"独立単位(局面)数 n = {n}   (seed中央値 ~{int(np.median(list(n_seeds.values())))} を局面ごとに平均集約)")
    out.append(f"手法: {', '.join(arms)}")
    if n < 20:
        out.append(f"  ⚠ n={n} は小さい。検出力不足の可能性大 (目安: 中効果で ~30-50 局面)。")
    out.append("")

    # ---- 手法ごとの要約 (局面平均 regret + ブートストラップCI) ----
    out.append("--- 手法別 要約 (局面集約後) ---")
    out.append(f"  {'arm':<12}{'mean':>8}{'median':>8}{'95%CI(mean)':>22}{'seeds':>7}")
    for a in arms:
        lo, hi = bootstrap_ci(M[a], args.boot, np.mean)
        out.append(f"  {a:<12}{M[a].mean():>8.3f}{np.median(M[a]):>8.3f}"
                   f"   [{lo:>6.3f}, {hi:>6.3f}]   {n_seeds[a]:>5}")
    out.append("")

    # ---- 多手法: Friedman 検定 ----
    if len(arms) >= 3:
        stat_f, p_f = stats.friedmanchisquare(*[M[a] for a in arms])
        out.append("--- Friedman 検定 (全手法に差があるか) ---")
        out.append(f"  chi2 = {stat_f:.3f},  p = {p_f:.4g}"
                   f"   {'→ 有意差あり (事後検定へ)' if p_f < 0.05 else '→ 有意差なし'}")
        # 平均順位 (低regretほど良い → 昇順ランクが良い)
        ranks = np.zeros(len(arms))
        for j in range(n):
            col = np.array([M[a][j] for a in arms])
            order = stats.rankdata(col if lower_better else -col)
            ranks += order
        ranks /= n
        out.append("  平均順位 (小さいほど良い): "
                   + ", ".join(f"{a}={ranks[i]:.2f}" for i, a in enumerate(arms)))
        out.append("")

    # ---- 事後/2手法: pairwise Wilcoxon (Holm補正) ----
    out.append("--- pairwise Wilcoxon 符号順位検定 (対応あり, Holm補正) ---")
    pair_p = []
    pair_info = {}
    for a, b in itertools.combinations(arms, 2):
        da = M[a] - M[b]  # a - b (regretなら負ほど a が良い)
        nonzero = da[da != 0]
        if len(nonzero) >= 1:
            try:
                w_stat, p = stats.wilcoxon(M[a], M[b])
            except ValueError:
                p = 1.0
        else:
            p = 1.0
        # 勝敗 (局面単位): lower_better なら a<b で a の勝ち
        if lower_better:
            a_win = int(np.sum(M[a] < M[b])); b_win = int(np.sum(M[a] > M[b]))
        else:
            a_win = int(np.sum(M[a] > M[b])); b_win = int(np.sum(M[a] < M[b]))
        tie = n - a_win - b_win
        d = paired_cohens_d(da)
        lo, hi = bootstrap_ci(da, args.boot, np.mean)
        pair_p.append(((a, b), p))
        pair_info[(a, b)] = dict(mean_diff=float(da.mean()), ci=(lo, hi),
                                 a_win=a_win, b_win=b_win, tie=tie, d=d,
                                 need_n=n_for_power(d))
    adj = holm_correction(pair_p)
    for (a, b), p in pair_p:
        info = pair_info[(a, b)]
        sig = "**" if adj[(a, b)] < 0.05 else ("*" if adj[(a, b)] < 0.10 else "  ")
        out.append(f"  {a} vs {b}: Δmean({a}-{b})={info['mean_diff']:+.3f} "
                   f"CI[{info['ci'][0]:+.3f},{info['ci'][1]:+.3f}]  "
                   f"p={p:.4g} p_holm={adj[(a,b)]:.4g} {sig}")
        better = a if (info['mean_diff'] < 0) == lower_better else b
        out.append(f"      勝敗(局面): {a}={info['a_win']} {b}={info['b_win']} 分={info['tie']}"
                   f"  → 優勢={better}  効果量d={info['d']:+.2f}"
                   f"  (d検出に必要な概算n≈{info['need_n']:.0f})")
    out.append("")
    out.append("凡例: ** p_holm<0.05 (有意), * <0.10 (傾向)。Δ<0 は前者が低regret=良い。")

    report = "\n".join(out)
    print(report)

    out_dir = args.out
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "regret_stats.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")
        # ペアCSV
        with open(os.path.join(out_dir, "regret_stats_pairs.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm_a", "arm_b", "mean_diff", "ci_lo", "ci_hi",
                        "p", "p_holm", "a_win", "b_win", "tie", "cohens_d", "need_n"])
            for (a, b), p in pair_p:
                i = pair_info[(a, b)]
                w.writerow([a, b, i["mean_diff"], i["ci"][0], i["ci"][1], p, adj[(a, b)],
                            i["a_win"], i["b_win"], i["tie"], i["d"], i["need_n"]])
        print(f"\n[out] -> {out_dir}/regret_stats.txt, regret_stats_pairs.csv")


if __name__ == "__main__":
    main()
