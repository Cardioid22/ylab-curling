#!/usr/bin/env python3
"""計算再投資実験の集計 (GPW2026).

アーム選択 CSV 群 (run_reinvest.sh の出力) と審判 Q テーブル (run_referee.sh の出力) を
(game_id, end, shot_num, candidate_idx) で left join し、各選択手のリグレットを算出する。

  regret = q_best(局面) - q_ref_mean(選んだ手)        ; q_best = その局面の全候補の max
出力:
  - アームごとの平均リグレット / 平均 actual_total_sims (等予算が揃っているかの検証)
  - 主問い A3 vs A4 の (局面×seed) head-to-head 勝率 (--pair で他ペアも可)
  - 明細 CSV (--out 指定時)

ディレクトリ構成 (run_reinvest.sh):
  REINVEST_DIR/<ARM>/seed_<S>/reinvest_results.csv   (ARM = A1..A6)
審判 (run_referee.sh):
  REFEREE_DIR/score_move_qtable*.csv

Usage:
  python3 scripts/aggregate_reinvest.py \
      --reinvest-dir experiments/reinvest/run_latest \
      --referee-dir  experiments/reinvest_referee \
      [--pair A3,A4] [--budget-tol 0.10] [--out experiments/reinvest/summary]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required. Install with: pip install pandas")

KEY = ["game_id", "end", "shot_num", "candidate_idx"]
POS = ["game_id", "end", "shot_num"]


def load_referee(referee_dir: Path) -> pd.DataFrame:
    parts = sorted(referee_dir.glob("score_move_qtable*.csv"))
    if not parts:
        sys.exit(f"No referee Q table found under {referee_dir} (score_move_qtable*.csv)")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    # 同一キーが複数 (分割実行の重複) あれば平均で畳む
    df = df.groupby(KEY, as_index=False).agg(q_ref_mean=("q_ref_mean", "mean"))
    return df


def load_arms(reinvest_dir: Path) -> pd.DataFrame:
    rows = []
    for arm_dir in sorted(reinvest_dir.iterdir()):
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name  # A1..A6
        csvs = sorted(arm_dir.glob("seed_*/reinvest_results*.csv"))
        for c in csvs:
            df = pd.read_csv(c)
            if df.empty:
                continue
            df["arm"] = arm
            rows.append(df)
    if not rows:
        sys.exit(f"No arm result CSVs found under {reinvest_dir} (<ARM>/seed_*/reinvest_results*.csv)")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate reinvestment experiment results.")
    ap.add_argument("--reinvest-dir", required=True, type=Path)
    ap.add_argument("--referee-dir", required=True, type=Path)
    ap.add_argument("--pair", default="A3,A4", help="head-to-head のアームペア (default: A3,A4)")
    ap.add_argument("--budget-tol", type=float, default=0.10,
                    help="等予算判定の許容相対差 (default: 0.10 = ±10%)")
    ap.add_argument("--out", type=Path, default=None, help="集計 CSV の出力先ディレクトリ (任意)")
    args = ap.parse_args()

    ref = load_referee(args.referee_dir)
    arms = load_arms(args.reinvest_dir)

    # 局面ごとの最良 Q (理想手の価値)
    q_best = ref.groupby(POS, as_index=False).agg(q_best=("q_ref_mean", "max"))

    # 選んだ手の Q を join
    merged = arms.merge(ref, on=KEY, how="left").merge(q_best, on=POS, how="left")
    n_missing = int(merged["q_ref_mean"].isna().sum())
    if n_missing:
        print(f"[warn] {n_missing} 行が審判 Q とマッチしませんでした "
              f"(候補生成不一致 or 審判未採点)。集計から除外します。", file=sys.stderr)
    merged = merged.dropna(subset=["q_ref_mean", "q_best"]).copy()
    merged["regret"] = merged["q_best"] - merged["q_ref_mean"]

    # ---------- アームごとサマリ ----------
    g = merged.groupby("arm")
    summary = g.agg(
        method=("method", "first"),
        depth=("depth", "first"),
        playouts=("playouts", "first"),
        rollouts=("rollouts_per_visit", "first"),
        n=("regret", "size"),
        mean_regret=("regret", "mean"),
        sd_regret=("regret", "std"),
        mean_q=("q_ref_mean", "mean"),
        mean_sims=("actual_total_sims", "mean"),
        min_sims=("actual_total_sims", "min"),
        max_sims=("actual_total_sims", "max"),
        mean_time=("time_sec", "mean"),
    ).reset_index()
    summary = summary.sort_values("arm").reset_index(drop=True)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    print("\n=== Per-arm summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---------- 等予算 (B) の揃い検証 ----------
    print("\n=== Budget (actual_total_sims) check ===")
    overall = summary["mean_sims"].mean()
    print(f"  arms mean actual_total_sims (target B ≈) : {overall:,.0f}")
    bad = []
    for _, row in summary.iterrows():
        rel = abs(row["mean_sims"] - overall) / overall if overall else 0.0
        flag = "OK" if rel <= args.budget_tol else "OFF"
        if flag == "OFF":
            bad.append(row["arm"])
        print(f"  {row['arm']:>3}  mean={row['mean_sims']:>12,.0f}  "
              f"rel_diff={rel*100:5.1f}%  [{flag}]")
    if bad:
        print(f"  [warn] 予算が ±{args.budget_tol*100:.0f}% を超えてズレているアーム: {', '.join(bad)}",
              file=sys.stderr)
        print(f"         run_reinvest.sh の P_*/R_* を校正して再実行してください。", file=sys.stderr)

    # ---------- head-to-head (主問い) ----------
    pair = [p.strip() for p in args.pair.split(",")]
    if len(pair) == 2:
        a, b = pair
        ka = merged[merged["arm"] == a][POS + ["seed", "q_ref_mean"]].rename(columns={"q_ref_mean": "q_a"})
        kb = merged[merged["arm"] == b][POS + ["seed", "q_ref_mean"]].rename(columns={"q_ref_mean": "q_b"})
        h2h = ka.merge(kb, on=POS + ["seed"], how="inner")
        print(f"\n=== Head-to-head {a} vs {b} (per 局面×seed) ===")
        if h2h.empty:
            print(f"  [warn] {a} と {b} で共通の (局面×seed) がありません。", file=sys.stderr)
        else:
            wins_a = int((h2h["q_a"] > h2h["q_b"]).sum())
            wins_b = int((h2h["q_b"] > h2h["q_a"]).sum())
            ties = int((h2h["q_a"] == h2h["q_b"]).sum())
            tot = len(h2h)
            print(f"  pairs            : {tot}")
            print(f"  {a} wins         : {wins_a}  ({100*wins_a/tot:.1f}%)")
            print(f"  {b} wins         : {wins_b}  ({100*wins_b/tot:.1f}%)")
            print(f"  ties             : {ties}  ({100*ties/tot:.1f}%)")
            print(f"  mean q: {a}={h2h['q_a'].mean():.3f}  {b}={h2h['q_b'].mean():.3f}  "
                  f"(Δ = {h2h['q_a'].mean()-h2h['q_b'].mean():+.3f}, {a}−{b})")

    # ---------- 出力 ----------
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out / "reinvest_summary.csv", index=False)
        merged.to_csv(args.out / "reinvest_joined.csv", index=False)
        print(f"\nWrote {args.out/'reinvest_summary.csv'} and {args.out/'reinvest_joined.csv'}")


if __name__ == "__main__":
    main()
