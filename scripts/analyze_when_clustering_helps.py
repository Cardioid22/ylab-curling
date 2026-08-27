#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「どんな局面でクラスタリングが有効か」分析 (GPW2026 本論文)
================================================================================
position_features.py の出力 (1 局面 1 行) を入力に、

  1. 目的変数 (アーム間 regret 差・screen_loss・rho_gain・eta2_q など) と各特徴量の Spearman 相関
  2. 層別比較 (進行度 / ハンマー / 石数 / 多峰性 / 巨大クラスタ退化 / ハウス混雑) の平均±SE と Kruskal-Wallis p
  3. 図 (上位特徴の散布図、層別箱ひげ)

を出す。regret 差 `d_A_B` は「正 = A が悪い」なので、例えば d_A9_A1 < 0 の局面 = クラスタリング (A9) が
全探索 (A1) より良い局面。

Usage:
  python scripts/analyze_when_clustering_helps.py \
      --features reinvest_experiment/run50v2/features/position_features.csv \
      --out reinvest_experiment/run50v2/features/analysis \
      [--targets d_A9_A1,d_A2_A1,d_A9_A2,screen_loss,rho_gain,eta2_q] [--min-n 8] [--no-fig]
================================================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ID_COLS = {"game_id", "end", "shot_num", "team"}
GROUPS = [
    ("problem_type", "degenerate(最終ショット/巨大クラスタ) / needle(正解一意) / multi_area(別エリアの良手複数) / flat(何でも同じ)"),
    ("phase_r", None),
    ("phase_shot", None),
    ("has_hammer", None),
    ("n_stones_bin", None),
    ("no1_mine", None),
    ("best_type_is_hit", None),
    ("multimodal", "n_near_best_050 >= 3 (q* から 0.5 点以内に 3 候補以上)"),
    ("multitype", "n_types_near_best_050 >= 2 (q* 近傍に複数の戦術種別)"),
    ("degenerate", "largest_frac >= 0.5 (最大クラスタが候補の半分以上)"),
    ("crowded_house", "n_house >= 5"),
    ("center_lane_blocked", None),
]


def derive_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "n_near_best_050" in df:
        df["multimodal"] = (df["n_near_best_050"] >= 3).astype(int)
    if "n_types_near_best_050" in df:
        df["multitype"] = (df["n_types_near_best_050"] >= 2).astype(int)
    if "largest_frac" in df:
        df["degenerate"] = (df["largest_frac"] >= 0.5).astype(int)
    if "n_house" in df:
        df["crowded_house"] = (df["n_house"] >= 5).astype(int)
    # 「問題の形」4分類 (審判テーブル + クラスタ構造から):
    #   degenerate : 最終ショット or 巨大クラスタ (距離関数が潰れる)
    #   needle     : q* 近傍 (0.5点以内) の候補が 2 個以下 = 正解が一意 → 被覆 (screen_loss) で決まる
    #   multi_area : 近傍候補が 3 個以上かつ戦術種別が 2 種以上 = 別エリアの良手が複数 → 表現で決まる
    #   flat       : 近傍候補が多数で戦術種別は 1 種 = 何を選んでも同じ
    if {"n_near_best_050", "n_types_near_best_050"} <= set(df.columns):
        deg = (df.get("remaining", 99) <= 1) | (df.get("largest_frac", 0) >= 0.5)
        needle = df["n_near_best_050"] <= 2
        multi = (df["n_near_best_050"] >= 3) & (df["n_types_near_best_050"] >= 2)
        df["problem_type"] = np.select([deg, needle, multi], ["degenerate", "needle", "multi_area"], default="flat")
    return df


def interaction_section(df: pd.DataFrame, value_arms, blind_arms, base_arm, P):
    """選択規則 (価値ベース vs 価値盲目 vs 全探索) × 局面群の交互作用。regret_<arm> 列を使う。"""
    va = [a for a in value_arms if f"regret_{a}" in df.columns]
    ba = [a for a in blind_arms if f"regret_{a}" in df.columns]
    if not va or not ba or f"regret_{base_arm}" not in df.columns:
        return
    d = df.copy()
    d["value_based"] = d[[f"regret_{a}" for a in va]].mean(axis=1)
    d["value_blind"] = d[[f"regret_{a}" for a in ba]].mean(axis=1)
    d["base"] = d[f"regret_{base_arm}"]
    d["d_vb_vbl"] = d["value_based"] - d["value_blind"]
    d["d_vbl_base"] = d["value_blind"] - d["base"]
    d["d_vb_base"] = d["value_based"] - d["base"]
    P("")
    P(f"--- 選択規則 × 局面群 (価値ベース={va} / 価値盲目={ba} / 基準={base_arm}; regret の平均, 低いほど良い) ---")
    for gcol in ["crowded_house", "problem_type", "phase_r", "n_stones_bin"]:
        if gcol not in d.columns:
            continue
        P(f"  [{gcol}]")
        for lv, g in d.groupby(gcol):
            def wp(col):
                x = g[col].dropna()
                return stats.wilcoxon(x).pvalue if len(x) >= 6 and (x != 0).any() else float("nan")
            P(f"    {str(lv):<12} n={len(g):3d}  base={g['base'].mean():.3f}  blind={g['value_blind'].mean():.3f}  value={g['value_based'].mean():.3f}"
              f"  | value-blind={g['d_vb_vbl'].mean():+.3f} (p={wp('d_vb_vbl'):.3f})"
              f"  blind-base={g['d_vbl_base'].mean():+.3f} (p={wp('d_vbl_base'):.3f})"
              f"  value-base={g['d_vb_base'].mean():+.3f} (p={wp('d_vb_base'):.3f})")
        if d[gcol].nunique() == 2:
            lv = sorted(d[gcol].unique(), key=str)
            for col, nm in [("d_vb_vbl", "value-blind"), ("d_vbl_base", "blind-base"), ("d_vb_base", "value-base")]:
                a = d[d[gcol] == lv[0]][col].dropna(); b = d[d[gcol] == lv[1]][col].dropna()
                if len(a) >= 3 and len(b) >= 3:
                    P(f"    {nm}: {lv[0]} vs {lv[1]} Mann-Whitney p={stats.mannwhitneyu(a, b).pvalue:.3f}")
    for f in ["n_house", "n_cand", "n_near_best_050", "q_range", "rho_cand", "q_sd_mean"]:
        if f in d.columns:
            r, p = stats.spearmanr(d[f], d["d_vb_vbl"], nan_policy="omit")
            P(f"  spearman({f}, value-blind) = {r:+.2f} (p={p:.3f})")


def filter_positions(df: pd.DataFrame, pos_dir: Path, invert: bool) -> pd.DataFrame:
    keys = set()
    for fp in sorted(pos_dir.glob("batch_*.csv")):
        sub = pd.read_csv(fp, usecols=["match_id", "end", "shot_num"])
        keys |= set(map(tuple, sub[["match_id", "end", "shot_num"]].values))
    m = df[["game_id", "end", "shot_num"]].apply(tuple, axis=1).isin(keys)
    return df[~m] if invert else df[m]


def numeric_features(df: pd.DataFrame, targets: list) -> list:
    skip = ID_COLS | set(targets)
    out = []
    for c in df.columns:
        if c in skip or c.startswith("d_") or c.startswith("dadj_") or c.startswith("regret_") or c.startswith("n_seeds"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) >= 3:
            out.append(c)
    return out


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", default="d_A9_A1,d_A2_A1,d_A9_A2,screen_loss,rho_gain,eta2_q")
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--only-positions", type=Path, default=None, help="この batch_*.csv に含まれる局面だけを使う (例: test_positions50 = パイロット)")
    ap.add_argument("--exclude-positions", type=Path, default=None, help="この batch_*.csv の局面を除外 (例: test_positions50 → 新規150 = 検証用)")
    ap.add_argument("--value-arms", default="A9,A9P5,A9R05", help="価値ベース選択のアーム (交互作用の節)")
    ap.add_argument("--blind-arms", default="A2,A5", help="価値盲目削減のアーム")
    ap.add_argument("--base-arm", default="A1")
    args = ap.parse_args()

    df = derive_groups(pd.read_csv(args.features))
    if args.only_positions:
        df = filter_positions(df, args.only_positions, invert=False)
    if args.exclude_positions:
        df = filter_positions(df, args.exclude_positions, invert=True)
    targets = [t for t in args.targets.split(",") if t and t in df.columns]
    missing = [t for t in args.targets.split(",") if t and t not in df.columns]
    if missing:
        print(f"[warn] targets not in features (skipped): {missing}", file=sys.stderr)
    if not targets:
        sys.exit("no usable target column")
    args.out.mkdir(parents=True, exist_ok=True)
    feats = numeric_features(df, targets)
    lines = []
    P = lines.append
    P("=" * 78)
    P(f"どんな局面でクラスタリングが有効か — n={len(df)} 局面, targets={targets}")
    P("=" * 78)
    P("d_A_B = regret_A − regret_B (正 = A が悪い)。screen_loss = q* − 代表手最良 q (小さいほど被覆が良い)。")

    # ---------- 1. 相関 ----------
    corr_rows = []
    for t in targets:
        y = df[t]
        for f in feats:
            x = df[f]
            ok = ~(x.isna() | y.isna())
            n = int(ok.sum())
            if n < args.min_n or x[ok].nunique() < 3:
                continue
            r, p = stats.spearmanr(x[ok], y[ok])
            corr_rows.append(dict(target=t, feature=f, n=n, rho=r, p=p))
    corr = pd.DataFrame(corr_rows)
    if not corr.empty:
        corr["abs_rho"] = corr["rho"].abs()
        corr = corr.sort_values(["target", "abs_rho"], ascending=[True, False])
        corr.drop(columns="abs_rho").to_csv(args.out / "feature_correlations.csv", index=False)
        for t in targets:
            sub = corr[corr["target"] == t].head(12)
            P("")
            P(f"--- {t}: Spearman ρ 上位 (|ρ| 順) ---")
            P(f"  {'feature':<28} {'n':>4} {'rho':>7} {'p':>8}")
            for _, r in sub.iterrows():
                flag = "**" if r["p"] < 0.05 else "*" if r["p"] < 0.10 else ""
                P(f"  {r['feature']:<28} {int(r['n']):>4} {r['rho']:>+7.3f} {r['p']:>8.3f} {flag}")

    # ---------- 2. 層別 ----------
    grp_rows = []
    for gcol, desc in GROUPS:
        if gcol not in df.columns:
            continue
        for t in targets:
            sub = df[[gcol, t]].dropna()
            if sub.empty or sub[gcol].nunique() < 2:
                continue
            groups = [g[t].values for _, g in sub.groupby(gcol)]
            try:
                kw_p = stats.kruskal(*groups).pvalue if all(len(g) >= 2 for g in groups) else float("nan")
            except ValueError:
                kw_p = float("nan")
            for gv, g in sub.groupby(gcol):
                grp_rows.append(dict(group=gcol, level=gv, target=t, n=len(g),
                                     mean=g[t].mean(),
                                     se=g[t].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else float("nan"),
                                     median=g[t].median(), kruskal_p=kw_p))
    grp = pd.DataFrame(grp_rows)
    if not grp.empty:
        grp.to_csv(args.out / "group_comparisons.csv", index=False)
        for gcol, desc in GROUPS:
            sub = grp[grp["group"] == gcol]
            if sub.empty:
                continue
            P("")
            P(f"--- 層別: {gcol}" + (f" ({desc})" if desc else "") + " ---")
            for t in targets:
                st = sub[sub["target"] == t]
                if st.empty:
                    continue
                kp = st["kruskal_p"].iloc[0]
                flag = "**" if kp < 0.05 else "*" if kp < 0.10 else ""
                cells = "  ".join(f"{r['level']}: {r['mean']:+.3f}±{r['se']:.3f} (n={int(r['n'])})" for _, r in st.iterrows())
                P(f"  {t:<14} {cells}   [Kruskal p={kp:.3f}{flag}]")

    # ---------- 2b. 選択規則 × 局面群 の交互作用 ----------
    interaction_section(df, [a.strip() for a in args.value_arms.split(",")],
                        [a.strip() for a in args.blind_arms.split(",")], args.base_arm, P)

    # ---------- 3. 良い/悪い局面の一覧 (主目的変数) ----------
    main_t = targets[0]
    if main_t.startswith("d_"):
        P("")
        P(f"--- {main_t} が最も負 (=前者が良い) / 最も正 (=前者が悪い) の局面 ---")
        show = [c for c in ["game_id", "end", "shot_num", main_t, "n_stones", "n_house", "remaining",
                            "n_near_best_050", "largest_frac", "screen_loss", "best_type"] if c in df.columns]
        s = df.dropna(subset=[main_t]).sort_values(main_t)
        P("  [前者が良い]")
        for _, r in s.head(6).iterrows():
            P("   " + "  ".join(f"{c}={fmt(r[c])}" for c in show))
        P("  [前者が悪い]")
        for _, r in s.tail(6).iterrows():
            P("   " + "  ".join(f"{c}={fmt(r[c])}" for c in show))

    text = "\n".join(lines)
    print(text)
    (args.out / "when_clustering_helps.txt").write_text(text + "\n", encoding="utf-8")

    # ---------- 図 ----------
    if not args.no_fig:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[warn] matplotlib not available; skip figures", file=sys.stderr)
            return
        for t in targets[:3]:
            sub = corr[corr["target"] == t].head(6) if not corr.empty else pd.DataFrame()
            if sub.empty:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
            for ax, (_, r) in zip(axes.flat, sub.iterrows()):
                f = r["feature"]
                ax.scatter(df[f], df[t], s=18, alpha=0.7)
                ax.axhline(0, color="gray", lw=0.8, ls="--")
                ax.set_xlabel(f); ax.set_ylabel(t)
                ax.set_title(f"rho={r['rho']:+.2f} p={r['p']:.3f} n={int(r['n'])}", fontsize=10)
            fig.suptitle(f"{t} vs top features (Spearman)")
            fig.tight_layout()
            fig.savefig(args.out / f"scatter_{t}.png", dpi=130)
            plt.close(fig)
        # 層別箱ひげ (主目的変数)
        gcols = [g for g, _ in GROUPS if g in df.columns]
        if gcols:
            n = len(gcols)
            ncol = (n + 1) // 2
            fig, axes = plt.subplots(2, ncol, figsize=(3.2 * ncol, 7))
            for ax, g in zip(axes.flat, gcols):
                sub = df[[g, main_t]].dropna()
                levels = sorted(sub[g].unique(), key=str)
                ax.boxplot([sub[sub[g] == lv][main_t].values for lv in levels],
                           tick_labels=[f"{lv}\n(n={int((sub[g] == lv).sum())})" for lv in levels])
                ax.axhline(0, color="gray", lw=0.8, ls="--")
                ax.set_title(g, fontsize=10); ax.tick_params(axis="x", labelsize=8)
            for ax in list(axes.flat)[len(gcols):]:
                ax.axis("off")
            fig.suptitle(f"{main_t} by group (negative = former arm better)")
            fig.tight_layout()
            fig.savefig(args.out / f"groups_{main_t}.png", dpi=130)
            plt.close(fig)
        print(f"[fig] -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
