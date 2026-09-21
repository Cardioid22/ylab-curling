#!/usr/bin/env python3
"""GPW2026 本論文・ミーティング用の図を gpw_experiment/figures/ に一括生成する。

入力 (すべて回収済みデータ):
  gpw_experiment/regret500/joined_500full11.csv  … 11アーム × 500局面
  gpw_experiment/regret500/joined_noisy7.csv     … 外乱込み7アーム × 500局面 (査読対応アブレーション)
  gpw_experiment/regret500/sweep_P{100,400,800}/reinvest_joined.csv … 予算スイープ (先頭200局面)

図:
  fig_sweep_dP.png       … d(P) 4点曲線 (ex-r1 / 全局面)
  fig_sweep_r1_split.png … d(P) を r=1 (退化32局面) と ex-r1 に分けた棒
  fig_regret500_arms.png … 全15アームの regret 序列 (500局面)
  fig_ablation_decomp.png… アブレーション分解 (価値スクリーン / クラスタ構造 / 削減)
  fig_noisy_lever.png    … 決定的→外乱込み評価の対 (全手法に効く主レバー)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

REG = Path("gpw_experiment/regret500")
OUT = Path("gpw_experiment/figures")
OUT.mkdir(exist_ok=True)

C_NOISY = "#2a78d6"   # 外乱込み評価
C_DET = "#eb6834"     # 決定的評価
C_PROP = "#134a8e"    # 提案 (A9P5N)
C_GRAY = "#8a8a8a"


def per_position(joined: pd.DataFrame) -> pd.DataFrame:
    """(game_id, arm) ごとに 5seed 平均 regret。shot_num も保持。"""
    return (
        joined.groupby(["game_id", "shot_num", "arm"])["regret"]
        .mean()
        .reset_index()
    )


def load_sweep() -> dict[int, pd.DataFrame]:
    first200 = set(
        pd.read_csv(REG / "sweep_P100" / "reinvest_joined.csv")["game_id"].unique()
    )
    out = {}
    for p, path in [
        (100, REG / "sweep_P100" / "reinvest_joined.csv"),
        (200, REG / "joined_500full11.csv"),
        (400, REG / "sweep_P400" / "reinvest_joined.csv"),
        (800, REG / "sweep_P800" / "reinvest_joined.csv"),
    ]:
        j = pd.read_csv(path)
        j = j[j["game_id"].isin(first200) & j["arm"].isin(["A9P5N", "A1N"])]
        pos = per_position(j)
        w = pos.pivot_table(index=["game_id", "shot_num"], columns="arm", values="regret").reset_index()
        w["d"] = w["A9P5N"] - w["A1N"]
        out[p] = w
    return out


def fig_sweep_dP(sweep: dict[int, pd.DataFrame]) -> None:
    ps = sorted(sweep)
    rows = []
    for p in ps:
        w = sweep[p]
        for name, sub in (("ex_r1", w[w["shot_num"] != 15]), ("all", w)):
            d = sub["d"]
            rows.append(
                dict(P=p, subset=name, mean=d.mean(), se=d.std() / np.sqrt(len(d)),
                     p_less=stats.wilcoxon(d, alternative="less").pvalue)
            )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(0, color=C_GRAY, lw=1, zorder=1)
    for name, color, label, marker in [
        ("all", C_GRAY, "全200局面", "s"),
        ("ex_r1", C_PROP, "最終ショット (r=1) 除外", "o"),
    ]:
        sub = df[df["subset"] == name]
        ax.errorbar(sub["P"], sub["mean"], yerr=sub["se"], color=color, marker=marker,
                    ms=7, lw=2, capsize=4, label=label, zorder=3)
    for _, r in df[df["subset"] == "ex_r1"].iterrows():
        txt = f"{r['mean']:+.3f}\n(p={r['p_less']:.0e})" if r["p_less"] < 0.05 else f"{r['mean']:+.3f}\n(n.s.)"
        off = (14, -38) if r["P"] == 100 else (0, -34)
        ax.annotate(txt, (r["P"], r["mean"]), textcoords="offset points", xytext=off,
                    ha="center", fontsize=9, color=C_PROP)
    ax.set_ylim(float(df["mean"].min() - df["se"].max()) - 0.035, None)
    ax.set_xscale("log", base=2)
    ax.set_xticks([100, 200, 400, 800])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("探索予算 P (root playouts; 子数 K = P/50 も連動)")
    ax.set_ylabel("d(P) = regret(提案 A9P5N) − regret(全探索 A1N)")
    ax.set_title("候補削減の利得は低予算で最大、予算増で単調に消える (先頭200局面 × 5 seed)")
    ax.text(0.44, 0.04, "下 (負) = クラスタ削減が有利", transform=ax.transAxes, fontsize=10, color=C_PROP)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sweep_dP.png", dpi=150)
    plt.close(fig)


def fig_sweep_r1_split(sweep: dict[int, pd.DataFrame]) -> None:
    ps = sorted(sweep)
    ex, r1 = [], []
    for p in ps:
        w = sweep[p]
        ex.append(w.loc[w["shot_num"] != 15, "d"].mean())
        r1.append(w.loc[w["shot_num"] == 15, "d"].mean())
    x = np.arange(len(ps))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0, color=C_GRAY, lw=1)
    b1 = ax.bar(x - 0.19, ex, width=0.38, color=C_PROP, label="通常局面 (n=168)")
    b2 = ax.bar(x + 0.19, r1, width=0.38, color=C_DET, label="最終ショット r=1 (n=32, 距離関数が退化)")
    for bars in (b1, b2):
        for rect in bars:
            v = rect.get_height()
            ax.annotate(f"{v:+.3f}", (rect.get_x() + rect.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4 if v >= 0 else -14),
                        ha="center", fontsize=9)
    ax.set_xticks(x, [f"P={p}" for p in ps])
    ax.set_ylim(min(ex) - 0.07, max(r1) + 0.06)
    ax.set_ylabel("d = regret(A9P5N) − regret(A1N)")
    ax.set_title("高予算での見かけの逆転は r=1 退化層のみが原因 (スクリーンバイパスで除去可能)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sweep_r1_split.png", dpi=150)
    plt.close(fig)


def arm_stats(pos: pd.DataFrame) -> pd.DataFrame:
    g = pos.groupby("arm")["regret"]
    return pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.count())})


def fig_regret500_arms(full11: pd.DataFrame, noisy7: pd.DataFrame) -> None:
    pos = pd.concat([
        per_position(full11),
        per_position(noisy7[~noisy7["arm"].isin(full11["arm"].unique())]),
    ])
    st = arm_stats(pos).sort_values("mean")
    noisy = {a for a in st.index if a.endswith("N") or "N4" in a}
    labels = {
        "A9P5N": "A9P5N 提案: クラスタ+価値K=4", "A8N": "A8N 価値上位K=4 (クラスタ無し)",
        "A12N": "A12N ベイズTS", "A1N": "A1N 全探索", "A5N": "A5N 乱択K≈18", "A2N": "A2N 既存法 (medoid)",
        "A5N4": "A5N4 乱択K=4", "A1": "A1 全探索", "A2": "A2 既存法 (medoid)", "A5": "A5 乱択K≈18",
        "A9": "A9 クラスタ+価値 (R_pre=3)", "A9P5": "A9P5 クラスタ+価値 (R_pre=5)",
        "A11a": "A11a PW(積極)", "A11b": "A11b PW(保守)", "A12": "A12 ベイズTS (決定的)",
    }
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    y = np.arange(len(st))[::-1]
    colors = [C_PROP if a == "A9P5N" else (C_NOISY if a in noisy else C_DET) for a in st.index]
    ax.barh(y, st["mean"], xerr=st["se"], color=colors, capsize=2.5, height=0.72)
    for yi, (a, r) in zip(y, st.iterrows()):
        ax.annotate(f"{r['mean']:.3f}", (r["mean"] + r["se"] + 0.004, yi), va="center", fontsize=9)
    ax.set_yticks(y, [labels.get(a, a) for a in st.index], fontsize=9.5)
    ax.set_xlabel("平均 regret (審判 K=200 基準; 小さいほど良い)")
    ax.set_title("全15アーム × 500局面 × 5seed の平均 regret", fontsize=12)
    ax.set_xlim(0, float(st["mean"].max()) + 0.09)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_PROP, C_NOISY, C_DET)]
    ax.legend(handles, ["提案 (A9P5N)", "外乱込み評価 (--noisy-tree)", "決定的評価"], loc="upper right", fontsize=9.5)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_regret500_arms.png", dpi=150)
    plt.close(fig)


def fig_ablation_decomp(noisy7: pd.DataFrame) -> None:
    pos = per_position(noisy7)
    st = arm_stats(pos).sort_values("mean")
    order = list(st.index)  # 良い順
    labels = {
        "A8N": "A8N: 価値上位 K=4 (クラスタ無し)", "A9P5N": "A9P5N: クラスタ+価値 K=4 (提案)",
        "A12N": "A12N: ベイズTS (全クラスタ子)", "A1N": "A1N: 全探索 (削減なし)",
        "A5N": "A5N: 乱択 K≈18", "A2N": "A2N: 既存法 medoid K≈18", "A5N4": "A5N4: 乱択 K=4",
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    y = np.arange(len(order))[::-1]
    colors = [C_PROP if a == "A9P5N" else C_NOISY for a in order]
    ax.barh(y, st.loc[order, "mean"], xerr=st.loc[order, "se"], color=colors, capsize=2.5, height=0.7)
    for yi, a in zip(y, order):
        ax.annotate(f"{st.loc[a,'mean']:.3f}", (st.loc[a, "mean"] + st.loc[a, "se"] + 0.004, yi),
                    va="center", fontsize=9.5)
    ax.set_yticks(y, [labels[a] for a in order], fontsize=10)
    ax.set_xlabel("平均 regret (外乱込み評価アームのみ, 500局面 × 5seed)")
    ax.set_title("査読対応アブレーション: 効果の分解")

    ypos = {a: yv for a, yv in zip(order, y)}
    x_base = float(st["mean"].max() + st["se"].max()) + 0.07
    x_text = x_base + 0.17

    def bracket(a1, a2, text, color, dx):
        y1, y2 = ypos[a1], ypos[a2]
        x0 = x_base + dx
        ax.plot([x0 - 0.015, x0, x0, x0 - 0.015], [y1, y1, y2, y2], color=color, lw=1.4)
        ax.text(x_text + 0.02, (y1 + y2) / 2, text, va="center", fontsize=9.5, color=color)

    d_vs = st.loc["A5N4", "mean"] - st.loc["A8N", "mean"]
    d_cl = st.loc["A9P5N", "mean"] - st.loc["A8N", "mean"]
    d_rd = st.loc["A1N", "mean"] - st.loc["A9P5N", "mean"]
    bracket("A9P5N", "A8N", f"クラスタ構造 +{d_cl:.3f}\n(p=0.41, 寄与なし)", C_GRAY, 0.0)
    bracket("A1N", "A9P5N", f"削減 > 全探索 −{d_rd:.3f}\n(Holm p=3e-4)", C_PROP, 0.06)
    bracket("A5N4", "A8N", f"価値スクリーン −{d_vs:.3f}\n(d=0.66, 効果の主因)", "#1a7a3e", 0.12)
    ax.set_xlim(0, x_text + 0.42)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation_decomp.png", dpi=150)
    plt.close(fig)


def fig_noisy_lever(full11: pd.DataFrame, noisy7: pd.DataFrame) -> None:
    pos = pd.concat([
        per_position(full11),
        per_position(noisy7[~noisy7["arm"].isin(full11["arm"].unique())]),
    ])
    st = arm_stats(pos)
    pairs = [("A9P5", "A9P5N", "クラスタ+価値 (提案)"), ("A12", "A12N", "ベイズTS"),
             ("A1", "A1N", "全探索"), ("A5", "A5N", "乱択K≈18"), ("A2", "A2N", "既存法 medoid")]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    yv = np.arange(len(pairs))[::-1]
    for y, (det, noi, name) in zip(yv, pairs):
        a, b = st.loc[det, "mean"], st.loc[noi, "mean"]
        ax.annotate("", xy=(b, y), xytext=(a, y),
                    arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.6))
        ax.plot([a], [y], "o", ms=9, color=C_DET, zorder=3)
        ax.plot([b], [y], "o", ms=9, color=C_PROP if noi == "A9P5N" else C_NOISY, zorder=3)
        ax.text((a + b) / 2, y + 0.16, f"{b - a:+.3f}", ha="center", fontsize=9.5, color="#333")
    ax.set_ylim(-0.55, len(pairs) - 1 + 0.65)
    ax.set_yticks(yv, [p[2] for p in pairs], fontsize=10.5)
    ax.set_xlabel("平均 regret (500局面 × 5seed; 左ほど良い)")
    ax.set_title("外乱込み評価 (--noisy-tree) は全手法に効く主レバー (−0.08〜−0.12)")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=c) for c in (C_DET, C_NOISY)]
    ax.legend(handles, ["決定的評価", "外乱込み評価"], loc="lower right", fontsize=9.5)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(OUT / "fig_noisy_lever.png", dpi=150)
    plt.close(fig)


def main() -> None:
    full11 = pd.read_csv(REG / "joined_500full11.csv")
    noisy7 = pd.read_csv(REG / "joined_noisy7.csv")
    sweep = load_sweep()
    fig_sweep_dP(sweep)
    fig_sweep_r1_split(sweep)
    fig_regret500_arms(full11, noisy7)
    fig_ablation_decomp(noisy7)
    fig_noisy_lever(full11, noisy7)
    for f in sorted(OUT.glob("fig_*.png")):
        print("wrote", f)


if __name__ == "__main__":
    main()
