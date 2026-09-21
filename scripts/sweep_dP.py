#!/usr/bin/env python3
"""予算スイープ d(P) = regret(A9P5N) − regret(A1N) の 4 点曲線 (P=100/200/400/800).

P=200 は本走 (gpw_experiment/regret500/joined_500full11.csv の先頭200局面),
P=100/400/800 は gpw_experiment/regret500/sweep_P*/reinvest_joined.csv。
局面単位で 5 seed 平均を取り、Wilcoxon 符号順位 (片側: A9P5N < A1N) と勝敗数を出す。
r=1 (shot_num==15, 距離関数が退化する最終ショット) の除外版も併記。
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


def per_position(joined: pd.DataFrame, arms=("A9P5N", "A1N")) -> pd.DataFrame:
    df = joined[joined["arm"].isin(arms)].copy()
    pos = (
        df.groupby(["game_id", "end", "shot_num", "arm"])
        .agg(regret=("regret", "mean"), sims=("actual_total_sims", "mean"))
        .reset_index()
    )
    wide = pos.pivot_table(
        index=["game_id", "end", "shot_num"], columns="arm", values="regret"
    ).reset_index()
    wide["d"] = wide[arms[0]] - wide[arms[1]]
    wide["is_r1"] = wide["shot_num"] == 15
    return wide, pos


def summarize(wide: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for name, sub in (("all", wide), ("ex_r1", wide[~wide["is_r1"]])):
        d = sub["d"].dropna()
        wins = int((d < 0).sum())
        losses = int((d > 0).sum())
        if len(d) and (d != 0).any():
            p = stats.wilcoxon(d, alternative="less").pvalue
        else:
            p = float("nan")
        rows.append(
            dict(
                P=label,
                subset=name,
                n=len(d),
                mean_A9P5N=sub["A9P5N"].mean() if len(sub) else float("nan"),
                mean_A1N=sub["A1N"].mean() if len(sub) else float("nan"),
                d_mean=d.mean(),
                d_median=d.median(),
                p_less=p,
                wins_A9P5N=wins,
                wins_A1N=losses,
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regret-dir", type=Path, default=Path("gpw_experiment/regret500"))
    ap.add_argument("--out", type=Path, default=Path("gpw_experiment/regret500/sweep_dP.csv"))
    args = ap.parse_args()

    # 先頭200局面の game_id 集合 = sweep 走行の対象
    ref = pd.read_csv(args.regret_dir / "sweep_P100" / "reinvest_joined.csv")
    first200 = set(ref["game_id"].unique())

    all_rows = []
    sims_rows = []
    for p_label, joined_path in [
        ("100", args.regret_dir / "sweep_P100" / "reinvest_joined.csv"),
        ("200", args.regret_dir / "joined_500full11.csv"),
        ("400", args.regret_dir / "sweep_P400" / "reinvest_joined.csv"),
        ("800", args.regret_dir / "sweep_P800" / "reinvest_joined.csv"),
    ]:
        j = pd.read_csv(joined_path)
        j = j[j["game_id"].isin(first200)]
        wide, pos = per_position(j)
        assert wide["is_r1"].sum() > 0 or p_label != "200"
        all_rows += summarize(wide, p_label)
        for arm in ("A9P5N", "A1N"):
            sub = pos[pos["arm"] == arm]
            sims_rows.append(dict(P=p_label, arm=arm, mean_sims=sub["sims"].mean(), n_pos=sub["game_id"].nunique()))

    out = pd.DataFrame(all_rows)
    sims = pd.DataFrame(sims_rows)
    out.to_csv(args.out, index=False)
    sims.to_csv(args.out.with_name("sweep_sims.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(sims.to_string(index=False, float_format=lambda v: f"{v:.0f}"))


if __name__ == "__main__":
    main()
