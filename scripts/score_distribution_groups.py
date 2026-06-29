#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
得点レンジ・グルーピング (単一エンド E[score] ベース)
================================================================================
方針:
  - 各候補手の「そのエンドの純得点の期待値 E[score]」= 審判 q_ref_mean (K回ロールアウト)。
  - 候補を E[score] のレンジ(既定: 幅1点)でクラスタ化 → 「+2〜+3点クラスタ」等。
  - 各レンジ・クラスタ内のメンバーの「リスク SD」= q_ref_sd を見えるようにする。
    （最終的に MCTS 候補に入れる手を、クラスタの得点レンジ + メンバーのリスクで選べる設計）
  - 戦略タイプ(shot_type)では分けない。レンジが主軸、type は参考表示のみ。

入力: 審判の score_move_qtable*.csv
  列: game_id,end,shot_num,candidate_idx,label,shot_type,q_ref_mean,q_ref_sd,n_rollouts,resampled
  (score_hist 列があれば各候補の得点分布も表示する)

使用例:
  python scripts/score_distribution_groups.py \
      --referee-csv reinvest_experiment/0617/reinvest_referee_k200/score_move_qtable.csv \
      --bin-width 1.0 --out reinvest_experiment/0617/score_groups
================================================================================
"""
import argparse, csv, glob, math, os
from collections import defaultdict


def label_type(lbl):
    return lbl.split("(")[0].strip().strip('"') if lbl else "?"


def bin_floor(x, w):
    # x が属するレンジの下端 (幅 w)。例 w=1: +1.46 -> 1.0, -0.66 -> -1.0
    return math.floor(x / w) * w


def range_label(lo, w):
    hi = lo + w
    f = (lambda v: f"{v:+.0f}" if abs(v - round(v)) < 1e-9 else f"{v:+.1f}")
    return f"{f(lo)}〜{f(hi)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--referee-csv", required=True, help="score_move_qtable*.csv (glob 可)")
    ap.add_argument("--bin-width", type=float, default=1.0, help="得点レンジ幅 (default 1.0)")
    ap.add_argument("--out", default=None, help="出力ディレクトリ")
    ap.add_argument("--max-members", type=int, default=8, help="各クラスタで表示するメンバー数")
    args = ap.parse_args()

    files = sorted(glob.glob(args.referee_csv))
    if not files:
        print(f"[error] no referee csv matched: {args.referee_csv}")
        return
    rows = []
    for fp in files:
        with open(fp, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    has_hist = "score_hist" in rows[0] if rows else False

    by_pos = defaultdict(list)
    for r in rows:
        try:
            cand = {
                "idx": int(r["candidate_idx"]),
                "type": label_type(r.get("label", "")),
                "mean": float(r["q_ref_mean"]),
                "sd": float(r["q_ref_sd"]),
                "K": r.get("n_rollouts", "?"),
                "resampled": r.get("resampled", "?"),
                "hist": r.get("score_hist", "") if has_hist else "",
            }
        except (KeyError, ValueError):
            continue
        if cand["idx"] < 0:
            continue
        by_pos[(r["game_id"], r["end"], r["shot_num"])].append(cand)

    out_dir = args.out
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    per_cand_rows = []

    w = args.bin_width
    for k in sorted(by_pos):
        cands = by_pos[k]
        qstar = max(c["mean"] for c in cands)
        K = cands[0]["K"]; resampled = cands[0]["resampled"]
        # レンジ・クラスタへ割り当て
        bins = defaultdict(list)
        for c in cands:
            lo = bin_floor(c["mean"], w)
            bins[lo].append(c)
            per_cand_rows.append({
                "game_id": k[0], "end": k[1], "shot_num": k[2],
                "candidate_idx": c["idx"], "type": c["type"],
                "E_score": round(c["mean"], 3), "risk_sd": round(c["sd"], 3),
                "score_bin_lo": lo, "score_range": range_label(lo, w),
                "score_hist": c["hist"], "is_best_bin": "",
            })
        best_lo = bin_floor(qstar, w)
        for row in per_cand_rows:
            if (row["game_id"], row["end"], row["shot_num"]) == k:
                row["is_best_bin"] = 1 if row["score_bin_lo"] == best_lo else 0

        print(f"\n■ g{k[0]} e{k[1]} s{k[2]}  | 候補{len(cands)}  最良E[score] q*={qstar:+.2f}  "
              f"(K={K}, resampled={resampled})")
        print(f"   {'得点レンジ':>10} {'members':>7} {'リスクSD(min/中央/max)':>22}   代表メンバー idx[type] E±SD")
        for lo in sorted(bins.keys(), reverse=True):
            members = sorted(bins[lo], key=lambda c: -c["mean"])
            sds = sorted(c["sd"] for c in members)
            n = len(sds)
            sd_min = sds[0]; sd_max = sds[-1]
            sd_med = sds[n // 2] if n % 2 else (sds[n//2 - 1] + sds[n//2]) / 2
            def fmt(c):
                s = f"{c['idx']}[{c['type']}] {c['mean']:+.2f}±{c['sd']:.2f}"
                if c["hist"]:
                    s += f" {{{c['hist']}}}"   # 得点分布 (新ビルドの審判 score_hist 列があるとき)
                return s
            shown = ", ".join(fmt(c) for c in members[:args.max_members])
            more = f" …他{len(members)-args.max_members}手" if len(members) > args.max_members else ""
            star = " ★最良レンジ" if lo == best_lo else ""
            print(f"   {range_label(lo,w):>10} {n:>7} {sd_min:>6.2f}/{sd_med:>5.2f}/{sd_max:>5.2f}      {shown}{more}{star}")

    if out_dir:
        path = os.path.join(out_dir, "score_distribution_per_candidate.csv")
        cols = ["game_id","end","shot_num","candidate_idx","type","E_score","risk_sd",
                "score_bin_lo","score_range","score_hist","is_best_bin"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=cols); wtr.writeheader()
            for row in per_cand_rows: wtr.writerow(row)
        print(f"\n[out] per-candidate -> {path}")


if __name__ == "__main__":
    main()
