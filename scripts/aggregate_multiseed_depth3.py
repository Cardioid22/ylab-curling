#!/usr/bin/env python3
"""
Multi-seed depth-3 結果を集約して ε-PAC ground truth メトリクスを算出する。

入力: depth3_experiment/multiseed_<TS>/seed_<S>/depth3_results_COMBINED.csv (K 個)

出力 (同じ親ディレクトリ直下):
  - multiseed_per_state.csv       — 局面ごとの seed 横断統計
  - multiseed_summary.txt         — 全体集計
  - multiseed_pac_metrics.csv     — Proposed 評価メトリクス (ε-PAC ベース)

Usage:
  python scripts/aggregate_multiseed_depth3.py \
      --parent-dir depth3_experiment/multiseed_20260520_120000 \
      --epsilon 0.5
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required. Install with: pip install pandas")


def collect_results(parent: Path) -> "pd.DataFrame":
    """seed_*/depth3_results_COMBINED.csv を縦結合し seed カラムを付与する."""
    frames = []
    for seed_dir in sorted(parent.glob("seed_*")):
        m = re.match(r"seed_(\d+)", seed_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        csvs = list(seed_dir.glob("depth3_results_COMBINED.csv"))
        if not csvs:
            # fallback: idx 別 CSV を結合する
            parts = sorted(seed_dir.glob("depth3_results_idx*.csv"))
            if not parts:
                print(f"  [warn] no result CSV under {seed_dir}", file=sys.stderr)
                continue
            df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        else:
            df = pd.read_csv(csvs[0])
        df["seed"] = seed
        frames.append(df)
    if not frames:
        sys.exit(f"No result CSVs under {parent}")
    return pd.concat(frames, ignore_index=True)


def per_state_stats(df: "pd.DataFrame") -> "pd.DataFrame":
    """局面 (game_id, end, shot_num) ごとに seed 横断統計を計算."""
    g = df.groupby(["game_id", "end", "shot_num"])

    rows = []
    for keys, sub in g:
        n_seeds = len(sub)
        # AllGrid: top pick 安定性
        ag_idx_mode, ag_idx_count = Counter(sub.allgrid_idx).most_common(1)[0]
        ag_mode_frac = ag_idx_count / n_seeds
        ag_label_mode = sub.loc[sub.allgrid_idx == ag_idx_mode, "allgrid_label"].iloc[0]

        # AllGrid: score 推定 (mean over seeds for the SAME chosen-action's-mean)
        # 各 seed が違う action を選んでいるかも知れないので、mean は「各 seed の top-1 mean」の平均
        ag_top_mean_avg = sub.allgrid_mean.mean()
        ag_top_mean_std = sub.allgrid_mean.std(ddof=1) if n_seeds > 1 else 0.0
        ag_top_mean_se = ag_top_mean_std / math.sqrt(n_seeds) if n_seeds > 1 else 0.0

        # Proposed: 同様
        prop_idx_mode, prop_idx_count = Counter(sub.proposed_idx).most_common(1)[0]
        prop_mode_frac = prop_idx_count / n_seeds
        prop_label_mode = sub.loc[sub.proposed_idx == prop_idx_mode, "proposed_label"].iloc[0]
        prop_top_mean_avg = sub.proposed_mean.mean()
        prop_top_mean_std = sub.proposed_mean.std(ddof=1) if n_seeds > 1 else 0.0
        prop_top_mean_se = prop_top_mean_std / math.sqrt(n_seeds) if n_seeds > 1 else 0.0

        # 一致指標 (mode 同士で評価)
        exact_mode_match = int(ag_idx_mode == prop_idx_mode)

        # 全 seed × seed の AllGrid 同士の一致率 (どれくらい AllGrid が安定か)
        ag_pairwise = sum(1 for a in sub.allgrid_idx for b in sub.allgrid_idx if a == b)
        ag_self_agree = (ag_pairwise - n_seeds) / (n_seeds * (n_seeds - 1)) if n_seeds > 1 else 1.0

        rows.append({
            "game_id": keys[0], "end": keys[1], "shot_num": keys[2],
            "n_seeds": n_seeds,
            "num_candidates": int(sub.num_candidates.median()),
            "num_clusters":   int(sub.num_clusters.median()),
            "ag_mode_idx":      ag_idx_mode,
            "ag_mode_label":    ag_label_mode,
            "ag_mode_frac":     ag_mode_frac,
            "ag_self_agree":    ag_self_agree,
            "ag_top_mean":      ag_top_mean_avg,
            "ag_top_mean_sd":   ag_top_mean_std,
            "ag_top_mean_se":   ag_top_mean_se,
            "prop_mode_idx":    prop_idx_mode,
            "prop_mode_label":  prop_label_mode,
            "prop_mode_frac":   prop_mode_frac,
            "prop_top_mean":    prop_top_mean_avg,
            "prop_top_mean_sd": prop_top_mean_std,
            "prop_top_mean_se": prop_top_mean_se,
            "mode_exact_match": exact_mode_match,
        })

    return pd.DataFrame(rows)


def pac_metrics(df: "pd.DataFrame", per_state: "pd.DataFrame", epsilon: float) -> dict:
    """ε-PAC ground truth に対する Proposed の合格率を計算.

    定義:
      Ω*(s; ε) := { a : Q̄_AllGrid(a) >= max Q̄_AllGrid - ε }
                  (= top-1 と ε 以内に入る AllGrid 候補集合の seed-modal 推定)

    実運用近似: 各 seed の AllGrid top picks (K 個) と、各 seed の AllGrid top mean を
    集めて Ω* を構成。Proposed mode が Ω* に入るかチェック.
    """
    per_state_pac = []
    for keys, sub in df.groupby(["game_id", "end", "shot_num"]):
        n_seeds = len(sub)
        # ε-PAC 集合 (近似): AllGrid が選んだ top の中で、その mean が
        # max(allgrid_mean) - epsilon 以上のもの (seeds 間集合)
        ag_top = sub.allgrid_mean.max()
        omega_star = sub.loc[sub.allgrid_mean >= ag_top - epsilon, "allgrid_idx"].unique().tolist()

        # Proposed mode を 1 個取る
        prop_idx_mode = Counter(sub.proposed_idx).most_common(1)[0][0]
        prop_in_omega = int(prop_idx_mode in omega_star)

        # 各 seed の Proposed が同 seed の AllGrid と ε 以内か (=同 seed 内 score_diff <= eps)
        # ※ ここは score_diff ≤ ε で判定
        within_eps_per_seed = (sub.score_diff <= epsilon).mean()

        per_state_pac.append({
            "game_id": keys[0], "end": keys[1], "shot_num": keys[2],
            "omega_star_size": len(omega_star),
            "prop_in_omega":   prop_in_omega,
            "frac_within_eps": within_eps_per_seed,
        })
    pac_df = pd.DataFrame(per_state_pac)

    return {
        "n_states": len(per_state),
        "n_seeds_per_state": int(df.groupby(["game_id","end","shot_num"]).size().mean()),
        "ag_self_agreement_mean": per_state.ag_self_agree.mean(),
        "ag_top_mean_se_avg":     per_state.ag_top_mean_se.mean(),
        "prop_self_agreement_mean": (per_state.prop_mode_frac).mean(),  # = mode の出現率
        "prop_top_mean_se_avg":     per_state.prop_top_mean_se.mean(),
        "mode_exact_match_pct":     100.0 * per_state.mode_exact_match.mean(),
        "omega_star_size_avg":      pac_df.omega_star_size.mean(),
        "prop_in_omega_pct":        100.0 * pac_df.prop_in_omega.mean(),
        "frac_within_eps_avg":      pac_df.frac_within_eps.mean(),
        "pac_per_state":            pac_df,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-dir", type=Path, required=True,
                    help="multiseed_<TS> ディレクトリ")
    ap.add_argument("--epsilon", type=float, default=0.5,
                    help="ε-PAC の許容スコア差 (default: 0.5)")
    args = ap.parse_args()

    print(f"Aggregating from {args.parent_dir} ...", file=sys.stderr)
    df = collect_results(args.parent_dir)
    print(f"  Loaded {len(df)} rows across {df.seed.nunique()} seeds", file=sys.stderr)

    per_state = per_state_stats(df)
    metrics   = pac_metrics(df, per_state, args.epsilon)

    # 書き出し
    out_per = args.parent_dir / "multiseed_per_state.csv"
    per_state.to_csv(out_per, index=False)
    print(f"  Wrote {out_per}", file=sys.stderr)

    out_pac = args.parent_dir / "multiseed_pac_metrics.csv"
    metrics["pac_per_state"].to_csv(out_pac, index=False)
    print(f"  Wrote {out_pac}", file=sys.stderr)

    summary_path = args.parent_dir / "multiseed_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write("Multi-seed Depth-3 Summary\n")
        fh.write("===========================\n")
        fh.write(f"n_states                   = {metrics['n_states']}\n")
        fh.write(f"n_seeds_per_state          = {metrics['n_seeds_per_state']}\n")
        fh.write(f"epsilon                    = {args.epsilon}\n")
        fh.write("\n-- AllGrid (ground-truth) stability --\n")
        fh.write(f"  self-agreement (pairwise mean) = {metrics['ag_self_agreement_mean']:.3f}\n")
        fh.write(f"  top-mean SE (across seeds) avg = {metrics['ag_top_mean_se_avg']:.4f}\n")
        fh.write(f"  |Ω*(s; ε)| avg                 = {metrics['omega_star_size_avg']:.2f}\n")
        fh.write("\n-- Proposed stability --\n")
        fh.write(f"  mode pick fraction (avg)       = {metrics['prop_self_agreement_mean']:.3f}\n")
        fh.write(f"  top-mean SE (across seeds) avg = {metrics['prop_top_mean_se_avg']:.4f}\n")
        fh.write("\n-- Proposed vs AllGrid (ground-truth) --\n")
        fh.write(f"  mode-exact match %             = {metrics['mode_exact_match_pct']:.1f}\n")
        fh.write(f"  Proposed in Ω*(s; ε) %         = {metrics['prop_in_omega_pct']:.1f}\n")
        fh.write(f"  frac_within_eps (per-seed avg) = {metrics['frac_within_eps_avg']:.3f}\n")
    print(f"  Wrote {summary_path}", file=sys.stderr)

    # コンソールにも出す
    print("\n" + summary_path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
